import time
import numpy as np
import torch
from nets.centernet_training import focal_loss, reg_l1_loss


###### 功能：定义训练一个epoch的过程 ######


def fit_one_epoch(model, optimizer, train_data, test_data, device):

    start_time = time.time()  # 获取当前时间
    model.train()  # 训练模式

    loss_train_list = []
    for step, data in enumerate(train_data):

        # 将数据转化为torch.tensor形式
        with torch.no_grad():
            batch = [torch.from_numpy(ann).type(torch.FloatTensor).to(device) for ann in data]
        
        # 取出各条数据
        batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

        optimizer.zero_grad()  # 清零梯度
        
        # 前向传播
        # hm为分类热力图 128 * 128 * num_classes
        # wh为框宽高回归参数 128 * 128 * 2
        # offset为中心点坐标位置回归参数 128 * 128 * 2
        hm, wh, offset = model(batch_images)

        c_loss = focal_loss(hm, batch_hms)  # 分类热力图损失
        wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)  # 框宽高损失
        off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)  # 中心位置坐标损失

        loss = c_loss + wh_loss + off_loss  # 总的loss

        loss.backward()  # loss值反向传播
        optimizer.step()  # 优化器迭代

        loss_train_list.append(loss.item())

        # 画进度条
        rate = (step + 1) / len(train_data)
        rate = (step + 1) / len(train_data)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    model.eval()  # 测试过程

    loss_test_list = []
    for step, data in enumerate(test_data):

        # 转化为torch.tensor形式
        with torch.no_grad():
            batch = [torch.from_numpy(ann).type(torch.FloatTensor).to(device) for ann in data]

        # 取出各条数据
        batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks = batch

        hm, wh, offset = model(batch_images)

        c_loss = focal_loss(hm, batch_hms)  # 分类热力图损失
        wh_loss = 0.1 * reg_l1_loss(wh, batch_whs, batch_reg_masks)  # 框宽高损失
        off_loss = reg_l1_loss(offset, batch_regs, batch_reg_masks)  # 中心位置坐标损失

        loss = c_loss + wh_loss + off_loss  # 总的loss

        loss_test_list.append(loss.item())

        # 画进度条
        rate = (step + 1) / len(test_data)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtest loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()

    train_loss = np.mean(loss_train_list)  # 该epoch总的训练loss
    test_loss = np.mean(loss_test_list)  # 该epoch总的测试loss
    stop_time = time.time()  # 获取当前时间
    
    print('total_train_loss: %.3f, total_test_loss: %.3f, epoch_time: %.3f.'%(train_loss, test_loss, stop_time - start_time))
    return train_loss, test_loss