from PIL import Image
from utils.utils_centernet import CenterNet

image_path = 'Image_samples/street.jpg'  # 测试图像路径

centernet = CenterNet()

image = Image.open(image_path)
image = centernet.detect_image(image)
image.show()