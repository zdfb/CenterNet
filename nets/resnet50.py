import math
import torch.nn as nn


######  功能：搭建resnet网络  ######
######  包括：上采样部分及输出部分 ######


# 定义Block部分，包括shortcut等
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dowmsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # 改变通道
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  # 3 * 3卷积
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)  # 升维
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = dowmsample
        self.stride = stride
    
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual  # short_cut 连接
        out = self.relu(out)

        return out


# 定义Resnet主干网络
class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        
        # 512 * 512 * 512 -> 256 * 256 * 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 256 * 256 * 64 -> 128 * 128 * 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 128 * 128 * 64 -> 128 * 128 * 256
        self.layer1 = self._make_layer(block, 64, layers[0])

        # 128 * 128 * 256 -> 64 * 64 * 512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)

        # 64 * 64 * 512 -> 32 * 32 * 1024
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # 32 * 32 * 1024 -> 16 * 16 * 2048
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        print(1)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x

# 定义resnet50网络, 仅使用其特征提取层
def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    
    # 获取特征提取部分
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4])
    features = nn.Sequential(*features)  # 用Sequential合并特征提取层

    return features


# 定义resnet50的解码器部分,上采样输出特征图
class resnet50_Decoder(nn.Module):
    def __init__(self, inplanes, bn_momentum=0.1):
        super(resnet50_Decoder, self).__init__()
        self.bn_momentum = bn_momentum  
        self.inplanes = inplanes
        self.deconv_with_bias = False  # 转置卷积是否使用Bias

        # 16 * 16 * 2048 -> 32 * 32 * 256 -> 64 * 64 * 128 -> 128 * 128 * 64
        # 利用转置卷积进行上采样

        self.deconv_layers = self._make_deconv_layer(
            num_layers = 3,  # 层数
            num_filters = [256, 128, 64],  # 通道数
            num_kernels = [4, 4, 4],  # 卷积核数
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            planes = num_filters[i]

            # 转置卷积进行上菜采样
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=self.deconv_with_bias))
        
            layers.append(nn.BatchNorm2d(planes, momentum=self.bn_momentum))  # bn层
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.deconv_layers(x)


# centernet头，将上采样后的特征图输出为类别及位置回归
class  resnet50_Head(nn.Module):
    def __init__(self, num_classes=80, channel=64, bn_momentum=0.1):
        super(resnet50_Head, self).__init__()

        # 对获取到的上采样后的特征进行转化，用于产生分类热力图、中心坐标位置的回归及框宽高的回归
        # 128 * 128 * 64 -> 128 * 128 * 64 -> 128 * 128 * num_classes  分类热力图
        # 128 * 128 * 64 -> 128 * 128 * 64 -> 128 * 128 * 2  框宽高回归
        # 128 * 128 * 64 -> 128 * 128 * 64 -> 128 * 128 * 2  中心位置坐标回归

        # 分类热力图部分
        self.cls_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),  # 融合周围特征
            nn.BatchNorm2d(channel, momentum=bn_momentum),  # BN层
            nn.ReLU(inplace=True),

            nn.Conv2d(channel, num_classes, kernel_size=1, stride=1, padding=0))

        # 宽高预测部分
        self.wh_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel, momentum=bn_momentum),  # BN层
            nn.ReLU(inplace=True),

            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0))
        
        # 中心坐标位置回归
        self.reg_head = nn.Sequential(
            nn.Conv2d(64, channel, kernel_size=3, padding=1, bias=False),  # 融合周围特征
            nn.BatchNorm2d(64, momentum=0.1),  # BN层
            nn.ReLU(inplace=True),

            nn.Conv2d(channel, 2, kernel_size=1, stride=1, padding=0))
    
    def forward(self, x):
        hm = self.cls_head(x).sigmoid_()  # 分类热力图部分，使用sigmoid归一化至0～1内
        wh = self.wh_head(x)  # 框宽高回归
        offset = self.reg_head(x)  # 中心点坐标回归
        return hm, wh, offset  
        
    







