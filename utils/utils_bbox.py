import torch
import numpy as np
from torch import nn
from torchvision.ops import nms

class DecodeBox():
    def __init__(self):
        super(DecodeBox, self).__init__()
    
    # 进行maxpool_nms抑制, 区域内仅保存最大值
    def pool_nms(self, heat, kernel = 3):
        pad = (kernel - 1) // 2  # 计算padding尺寸
        
        # 对热力图进行max_pooling操作,滤掉部分非极大值
        hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride = 1, padding=pad)
        keep = (hmax == heat).float()  # 仅保留原热力图中与最大值相等的点
        return heat * keep
    
    # 解码输出数据
    def decode_bbox(self, pred_hms, pred_whs, pred_offsets, confidence):
        pred_hms = self.pool_nms(pred_hms)  # 执行max_pool非极大值抑制
        
        # 取出数据的batch_size, 通道数, 输出高度及输出宽度
        batch_size, channel, output_h, output_w = pred_hms.shape

        detects = []  # 用于存放最终结果
        for batch in range(batch_size):

            # num_classes, 128, 128 -> 128, 128, num_classes -> 128 * 128, num_classes
            heat_map = pred_hms[batch].permute(1, 2, 0).view([-1, channel])

            # 2, 128, 128 -> 128, 128, 2 -> 128 * 128, 2
            pred_wh = pred_whs[batch].permute(1, 2, 0).view([-1, 2])

            # 2, 128, 128 -> 128, 128, 2 -> 128 * 128, 2
            pred_offsets = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])
            
            # 特征点的y坐标与x坐标
            y_cor, x_cor = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))

            x_cor, y_cor = x_cor.flatten().float(), y_cor.flatten().float()
            
            # 将坐标放置在与输入数据相同的device上
            x_cor = x_cor.to(pred_hms.device)
            y_cor = y_cor.to(pred_hms.device)

            # 获取特征点的种类置信度及种类
            # 求取每个特征点在num_class维度上的最大值，返回置信率及num_class的编号
            # class_conf, class_pred 形状为 (128, 128)
            class_conf, class_pred = torch.max(heat_map, dim=-1)
            mask = class_conf > confidence  # 筛选大于阈值的特征点

            # 取出得分筛选后对应的结果
            pred_wh_mask = pred_wh[mask]
            pred_offsets_mask = pred_offsets[mask]
            
            # 若不存在大于confidence的目标，不进行后续步骤
            if len(pred_wh_mask) == 0:
                detects.append([])
                continue

            # 调整预测后框中心 (num_objects, 1)
            x_cor_mask = torch.unsqueeze(x_cor[mask] + pred_offsets_mask[..., 0], -1)
            y_cor_mask = torch.unsqueeze(y_cor[mask] + pred_offsets_mask[..., 1], -1)

            # 计算预测框的宽与高的一半
            half_w, half_h = pred_wh_mask[..., 0:1] / 2,  pred_wh_mask[..., 1:2] / 2

            # 获得预测框的左上角与右上角
            # (num_objects, 4) 具体数据为 (xmin, ymin, xmax, ymax)
            bboxes = torch.cat([x_cor_mask - half_w, y_cor_mask - half_h, x_cor_mask + half_w, y_cor_mask + half_h], dim = 1)

            # 除以输出特征图尺寸进行归一化
            bboxes[:, [0, 2]] /= output_w
            bboxes[:, [1, 3]] /= output_h
            
            # 合并输出置信率及输出种类
            detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask], -1), torch.unsqueeze(class_pred[mask], -1).float()], dim=-1)
            detects.append(detect)
        return detects

    def centernet_correct_boxes(self, box_xy, box_wh, image_shape):

        # 调换x轴与y轴
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        image_shape = np.array(image_shape)

        # 转化为ymin, xmin, ymax, xmax的形式 
        box_mins = box_yx - (box_hw / 2.)
        box_maxs = box_yx + (box_hw / 2.)
        boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxs[..., 0:1], box_maxs[..., 1:2]], axis = -1)

        # 应该为 boxes * input_shape * image_shape / input_shape == boxes * image_shape
        boxes *= np.concatenate([image_shape, image_shape], axis = -1)

        return boxes
    
    # 进行非极大值抑制处理
    def postprocess(self, prediction, image_shape, is_nms = True, nms_thres=0.4):
        output = [None for _ in range(len(prediction))]

        for index, image_pred in enumerate(prediction):
            # image_pred 形状为(num_objects, 6)
            
            # 排除初筛后没有目标存在的情况
            if len(image_pred) == 0:
                continue

            # 获取预测框中包含的所有类别
            unique_labels = image_pred[:, -1].cpu().unique()

            if image_pred.is_cuda:
                unique_labels = unique_labels.cuda()
            
            # 单独处理每个种类
            for c in unique_labels:
                detections_class = image_pred[image_pred[:, -1] == c]
                
                if is_nms == True:
                    # 经过nms处理
                    keep = nms(
                        detections_class[:, :4], # 坐标
                        detections_class[:, 4], # 输出置信率
                        nms_thres # 过滤阈值
                    )
                    max_detections = detections_class[keep]  # nms处理后剩下的目标框
                else:
                    max_detections = detections_class
       
                # 添加到最终输出结果
                output[index] = max_detections if output[index] is None else torch.cat((output[index], max_detections))

            if output[index] is not None:
                output[index] = output[index].cpu().numpy()

                # 转化为中点及长宽的形式
                box_xy, box_wh = (output[index][:, 0:2] + output[index][:, 2:4]) / 2, output[index][:, 2:4] - output[index][:, 0:2]
                output[index][:, :4] = self.centernet_correct_boxes(box_xy, box_wh, image_shape)
        return output