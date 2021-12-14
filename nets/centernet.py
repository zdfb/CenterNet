import torch.nn as nn
from nets.resnet50 import resnet50, resnet50_Decoder, resnet50_Head


###### 功能：定义centernet主干网络 ######


# 以Resnet50作为centernet的backbone
class CenterNet_Resnet50(nn.Module):
    def __init__(self, num_classes = 20):
        super(CenterNet_Resnet50, self).__init__()
        
        # 512 * 512 * 3 -> 16 * 16 * 2048
        self.backbone = resnet50()

        # 16 * 16 * 2048 -> 128 * 128 * 64
        self.decoder = resnet50_Decoder(2048)

        # 128 * 128 * 64 -> 128 * 128 * 64 -> 128 * 128 * num_classes  分类热力图
        # 128 * 128 * 64 -> 128 * 128 * 64 -> 128 * 128 * 2  框宽高回归
        # 128 * 128 * 64 -> 128 * 128 * 64 -> 128 * 128 * 2  中心位置回归
        self.head = resnet50_Head(channel = 64, num_classes = num_classes)
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.decoder(feat))
