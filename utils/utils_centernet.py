import os
import torch
import numpy as np
from PIL import ImageDraw, ImageFont
from utils.utils_bbox import DecodeBox
from nets.centernet import CenterNet_Resnet50
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image, ncolors


###### 功能：解析模型，生成最终结果 ######


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CenterNet(object):
    def __init__(self):
        super(CenterNet, self).__init__()

        model_path = 'model_data/centernet_weights.pth'  # 模型存储路径
        classes_path = 'model_data/voc_classes.txt'  # 类别信息存储路径

        self.input_shape = [512, 512]  # 输入尺寸

        self.confidence = 0.3  # 置信率初筛阈值
        self.is_nms = False  # 是否执行nms
        self.nms_iou = 0.3  # 非极大值抑制IOU阈值

        # 获取种类名及数量
        self.class_names, self.num_classes = get_classes(classes_path)

        # 获取框的最终颜色
        self.colors = ncolors(self.num_classes)

        # 定义解码网络输出信息类
        self.bbox_util = DecodeBox()

        # 加载模型
        model = CenterNet_Resnet50(self.num_classes)
        # 加载训练权重
        model.load_state_dict(torch.load(model_path, map_location = device))
        model = model.eval()

        self.model = model.to(device)  # 将模型放置在相应的环境中
    
    def detect_image(self, image):
        image_shape = np.array(np.shape(image))[0:2]  # 输入图像尺寸
        image = cvtColor(image)  # 将输入图片转化为RGB格式

        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]))  # 将输入图像尺寸转化为固定尺寸

        # 对输入图像进行预处理
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype = 'float32')),(2, 0, 1)), 0)

        with torch.no_grad():
            image_ = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)  # 转化为tensor形式
            image_ = image_.to(device)
        
            outputs = self.model(image_)

            # 将预测过程进行解码
            # (num_objects, 6) 具体内容为 xmin, ymin, xmax, ymax, confidence, class_id
            outputs = self.bbox_util.decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence)

            # 执行nms过滤
            # (num_objects, 6) 具体内容为 xmin, ymin, xmax, ymax, confidence, class_id
            results = self.bbox_util.postprocess(outputs, image_shape, self.is_nms, self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:,5], dtype = 'int32')  # 预测类别
            top_conf = results[0][:, 4]  # 预测置信率
            top_boxes = results[0][:, :4]  # 预测框位置 (num_bbox, (ymin, xmin, ymax, xmax))
        
        # 绘制图像上的标注框
        font_size = np.floor(2e-2 * image.size[1]).astype('int32')  # 定义字体大小
        font = ImageFont.truetype(font = 'model_data/simhei.ttf', size=font_size)  # 定义字体样式

        for index, class_id in list(enumerate(top_label)):
            predicted_class = self.class_names[int(class_id)]  # 取出预测类别名称

            box = top_boxes[index]  # 预测框的位置信息 (ymin, xmin, ymax, xmax)
            score = top_conf[index] # 预测框的置信度

            ymin, xmin, ymax, xmax = box  # 取出坐标详细信息

            # 标签内容
            label_text = '{}{:.2f}'.format(predicted_class, score)

            # 绘制图像
            draw = ImageDraw.Draw(image)

            # 获取标签区域大小
            label_size = draw.textsize(label_text, font)
            # 绘制标签包围框
            draw.rectangle((xmin, ymin - label_size[1], xmin + label_size[0], ymin), fill = self.colors[class_id])
            # 绘制目标框
            draw.rectangle((xmin, ymin, xmax, ymax), outline = self.colors[class_id], width = 3)
            # 绘制标签
            draw.text((xmin, ymin - label_size[1]), label_text, fill = (255, 255, 255), font=font)
            del draw
        return image
    
    # 将输出结果写至txt内, 便于计算mAP
    def get_map_txt(self, image_id, image, map_out_path):
        # 打开将要写入的txt文件
        # 每张图片都写入一个txt文件内
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), 'w')

        image_shape = np.array(np.shape(image)[0:2])  # 输入图像的宽和高
        image = cvtColor(image)  # 将输入图片转化为RGB形式
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]))  # 缩放图像至模型要求尺寸  
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype = 'float32')),(2, 0, 1)), 0)  # 对输入图像进行预处理

        with torch.no_grad():
            image_ = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)
            image_ = image_.to(device)
            
            outputs = self.model(image_)

            outputs = self.bbox_util.decode_bbox(outputs[0], outputs[1], outputs[2], self.confidence)
            
            # 进行非极大值抑制处理
            results = self.bbox_util.postprocess(outputs, image_shape, self.is_nms, self.nms_iou)
            
            if results[0] is None:
                return image
            
            top_label = np.array(results[0][:,5], dtype = 'int32')  # 预测类别
            top_conf = results[0][:, 4]  # 预测置信率
            top_boxes = results[0][:, :4]  # 预测框位置 (num_bbox, (ymin, xmin, ymax, xmax))

            for index, class_id in list(enumerate(top_label)):
                predicted_class = self.class_names[int(class_id)]  # 取出预测类别名称

                box = top_boxes[index]  # 预测框的位置信息 (ymin, xmin, ymax, xmax)
                score = str(top_conf[index]) # 预测框的置信度

                ymin, xmin, ymax, xmax = box  # 取出坐标详细信息

                f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(xmin)), str(int(ymin)), str(int(xmax)), str(int(ymax))))
            
            f.close()
            return 
