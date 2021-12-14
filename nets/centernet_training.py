import torch
import torch.nn.functional as F


###### 定义网络训练过程中用到的损失函数 ######


# pred 为预测值 (bs, num_classes, 128, 128)
# target 为真实值 (bs, 128, 128, num_classes)
def focal_loss(pred, target):
    pred = pred.permute(0, 2, 3, 1)  # 转化为(bs, 128, 128, num_classes)

    pos_inds = target.eq(1).float()  # 比较真实值是否与1相等,相等的部分置为1, 其余部分为0
    neg_inds = target.lt(1).float()  # 比较真实值是否小于1, 小于的部分置为1, 其余部分为0

    neg_weights = torch.pow(1 - target, 4)  # 负样本加权, 越偏向于1的样本权重越小

    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)  # 将输出预测值限制在(1e-6, 1 - 1e-6)范围内

    # 计算focal loss, 难分类样本权重大, 易分类样本权重小
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    # 进行损失归一化
    num_pos = pos_inds.float().sum()  # 正样本数量
    pos_loss = pos_loss.sum()  # 总的正样本损失
    neg_loss = neg_loss.sum()  # 总的负样本损失

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


# 计算l1_loss
# pred 为预测值 (bs, 2, 128, 128)
# target 为真实值 (bs, 128, 128, 2)
# mask 为指示正样本所处位置的掩模, 对于位置偏移及框的大小, 仅计算正样本的损失 (bs, 128, 128)
def reg_l1_loss(pred, target, mask):
    pred = pred.permute(0, 2, 3, 1)  # (bs, 128, 128, 2)
    expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction = 'sum')  # 仅计算正样本的损失
    loss = loss / (mask.sum() + 1e-4)
    return loss



