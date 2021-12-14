import random
import colorsys
import numpy as np 
from PIL import Image


###### 功能：定义部分工具代码 ######

random.seed(30)  # 设置随机数种

# 将输入图像转换成RGB图像
def cvtColor(image):
    image = image.convert('RGB')
    return image


# 将输入图像缩放至固定尺寸
def resize_image(image, size):
    w, h = size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


# 获得类别名称及其数量
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


# 预处理
def preprocess_input(image):
    image = np.array(image, dtype = np.float32)[:, :, ::-1]
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return (image / 255. - mean) / std

# 随机选取N个HLS颜色
def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step
    return hls_colors


# 随机选取N个RGB颜色
def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append((r, g, b))
    random.shuffle(rgb_colors)
    return rgb_colors