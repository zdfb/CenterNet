import cv2
import math
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input


###### 功能：定义数据读取部分 ######


# 产生高斯圆的半径
def gaussian_radius(det_size, min_overlap = 0.7):
    height, width = det_size  # GT框的高及宽
    
    # 第一种情况
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    
    # 第二种情况
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    
    # 第三种情况
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


# 产生二维高斯函数
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

    

# 在特征图上绘制高斯圆
def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1  # 获取高斯圆的直径
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)  # 获取圆心周围的高斯值

    x, y = int(center[0]), int(center[1])  # 将圆心转化为整数形式

    height, width = heatmap.shape[0: 2]  # 输出特征图的尺寸

    # 将高斯圆限制在特征图范围内
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top: y + bottom, x - left: x + right]  # 需要画的区域
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]  # 取得高斯值

    if min(masked_heatmap.shape) > 0 and min(masked_gaussian.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


# 定义centernnet数据读取类
class CenternetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super(CenternetDataset, self).__init__()
        self.annotation_lines = annotation_lines  # 输入的标签行

        self.input_shape = input_shape  # 512 * 512
        self.output_shape = (int(input_shape[0] / 4), int(input_shape[1] / 4))  # 512 * 512 -> 128 * 128
        self.num_classes = num_classes  # 类别数量
        self.train = train
    
    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):

        # 进行数据增强
        # 读取图片及GT框的相关信息
        image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)
        
        # 128 * 128 * num_classes  存储分类标签
        batch_hm = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        # 128 * 128 * 2 存储坐标差异标签
        batch_wh = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        # 128 * 128 * 2 存储长宽差异标签
        batch_reg = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        # 128 * 128 掩膜mask
        batch_reg_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)

        # 若图中存在GT
        if len(box) != 0:
            boxes = np.array(box[:, :4], dtype=np.float32)  # 取出框的xmin, ymin, xmax, ymax
            # 将真实框转化至下采样后的特征图尺寸内，并限制在特征图边界内
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.input_shape[1] * self.output_shape[1], 0, self.output_shape[1] - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.input_shape[0] * self.output_shape[0], 0, self.output_shape[0] - 1)
        
        # 对每个真实GT框进行处理
        for i in range(len(box)):
            bbox = boxes[i].copy()  # 复制真实框
            cls_id = int(box[i, -1])  # 提取类别index

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]  # 获取GT框的高和宽

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))  # 产生高斯圆的半径, math.ceil()向上取整
                radius = max(0, int(radius))
                
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float)  # 特征图尺度的中心点坐标
                ct_int = ct.astype(np.int32)  # 将中心点坐标转化为整数

                # 绘制高斯热力图
                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)

                # 计算宽和高的标签值
                batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h

                # 计算中心点偏移量标签值
                batch_reg[ct_int[1], ct_int[0]] = ct - ct_int

                # 将GT中心点对应位置的mask设置为1
                batch_reg_mask[ct_int[1], ct_int[0]] = 1
        
        image = np.transpose(preprocess_input(image), (2, 0, 1))  # 将图片由(h, w, c)转化为(c, h, w)

        return image, batch_hm, batch_wh, batch_reg, batch_reg_mask


    # 产生a到b的随机数
    def rand(self, a=0, b=1):
        return np.random.rand() * (b-a) + a


    # 对输入图像进行随机增强
    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):

        line = annotation_line.split()

        image = Image.open(line[0])  # 读取图像
        image = cvtColor(image)  # 转化为RGB形式 

        iw, ih = image.size  # 原始图像的宽和高
        h, w = input_shape  # 要求输入的宽和高
        
        # 读取box相关参数
        # xmin, ymin, xmax, ymax, cls_id
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1: ]])

        if not random:  

            # 处于测试状态
            scale = min(w / iw, h / ih)
            nw = int(scale * iw)
            nh = int(scale * ih)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            
            # 将图像长边缩放至目标尺寸， 短边缺少的部分用灰色填充
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            if len(box) > 0:
                # 调整框的大小
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            
                # 将超出边界的值都整合至边界内
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h

                # 计算框的宽度及长度
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]

                # 剔除宽和高小于1像素的框
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return image_data, box

        # 对图像进行缩放并进行长和宽的扭曲

        # 生成新的宽高比
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)  # 生成随机尺度
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 将图像缺少的部分补上灰边
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 翻转图像
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)  # 左右翻转

        # 色域扭曲
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        if len(box) > 0:
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy

            if flip: 
                box[:, [0, 2]] = w - box[:, [2, 0]]
            
            # 将超出边界的值都整合至边界内
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h

            # 计算框的宽度及长度
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]

            # 剔除宽和高小于1像素的框
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box


# 按照Batch进行堆叠
def centernet_dataset_collate(batch):
    # 新建list存储每张图片的数据
    imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks = [], [], [], [], []

    # 放入数据
    for img, batch_hm, batch_wh, batch_reg, batch_reg_mask in batch:
        imgs.append(img)
        batch_hms.append(batch_hm)
        batch_whs.append(batch_wh)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)
    
    # 转化为numpy形式
    imgs = np.array(imgs)
    batch_hms = np.array(batch_hms)
    batch_whs = np.array(batch_whs)
    batch_regs = np.array(batch_regs)
    batch_reg_masks = np.array(batch_reg_masks)

    return imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks