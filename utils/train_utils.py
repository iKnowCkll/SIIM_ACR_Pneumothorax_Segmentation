import gc
from configparser import ConfigParser

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.cuda import amp

from torch.optim import lr_scheduler
from tqdm import tqdm

config_parser = ConfigParser()
config_parser.read('/home/chenk/model_train/kaggle/SIIM_ACR_Pneumothorax_Segmentation/mysql.cfg')
cfg = config_parser['default']

# =======================loss_function=======================
JaccardLoss = smp.losses.JaccardLoss(mode='binary')
DiceLoss = smp.losses.DiceLoss(mode='binary')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss = smp.losses.LovaszLoss(mode='binary', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False)
FocalLoss = smp.losses.FocalLoss(mode='binary', gamma=2,
                                 alpha=float(cfg['sample_ratio']) / (float(cfg['sample_ratio'])+1))


def criterion(y_pred, y_true):
    return 0.5 * BCELoss(y_pred, y_true) + 0.5 * DiceLoss(y_pred, y_true)


# ===================evaluating_indicator===========================
# 如果指定dim=(2, 3)，则表示在第2和第3个维度上进行求和，即针对每个通道和每个样本的二维图像进行计算
def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


# ===============================scheduler================================
def fetch_scheduler(optimizer):
    # T_max：最大迭代次数(如果是每个epoch更新一次学习率那就是epoch次数)；eta_min：最小学习率
    if cfg['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg['T_max']),
                                                   eta_min=float(cfg['min_lr']))
        return scheduler
    # T_0指第一个周期的长度。在该周期内，学习率会从初始值降低到 eta_min
    # T_mult：指每个重启周期的长度与上一个周期长度的比例。例如，如果 T_0=10，T_mult=2，则在第一个重启周期中，周期的长度将是 10；
    # 在第二个重启周期中，周期的长度将是 102=20；在第三个重启周期中，周期的长度将是 202=40；以此类推。
    elif cfg['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(cfg['T_0']),T_mult=int(cfg['T_mult']),
                                                             eta_min=float(cfg['min_lr']))
        return scheduler
    # 'min’模式检测metric是否不再减小，'max’模式检测metric是否不再增大；触发条件后lr*=factor；patience不再减小（或增大）的累计次数；
    # threshold：阈值，用来确定metric是否不再减小（或增大）；min_lr：最小学习率
    elif cfg['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='max',
                                                   factor=0.1,
                                                   patience=2,
                                                   threshold=0.0001,
                                                   min_lr=float(cfg['min_lr']))
        return scheduler
    elif cfg['scheduler'] == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        return scheduler
    elif cfg['scheduler'] is None:
        return None


# ===============================train_one_epoch================================
def train_one_epoch(model, optimizer, scheduler, dataloader, device, n_accumulate, epoch):
    model.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    running_loss = 0.0
    epoch_loss = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Train epoch:{epoch}')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss = criterion(y_pred, masks)
            loss = loss / n_accumulate

        scaler.scale(loss).backward()
        # 这里的技巧是bs比较小，迭代次数多的话，就多迭代几次后更新优化器，bs比较大时，迭代次数少的话，就每迭代一次更新一下优化器
        if (step + 1) % n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        # epoch_loss得到的是这个epoch中所有批次的损失平均值
        epoch_loss = running_loss / dataset_size
        # 返回当前CUDA设备上已经分配的显存量
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_mem=f'{mem:0.2f} GB')
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss


# ========================valid_one_epoch==============================
def valid_one_epoch(model, dataloader, device, optimizer, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    val_scores = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Valid epoch:{epoch}')
    with torch.no_grad():
        for step, (images, masks) in pbar:
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)

            batch_size = images.size(0)

            y_pred = model(images)
            loss = criterion(y_pred, masks)

            running_loss += (loss.item() * batch_size)
            dataset_size += batch_size

            epoch_loss = running_loss / dataset_size

            y_pred = nn.Sigmoid()(y_pred)
            val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
            val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
            val_scores.append([val_dice, val_jaccard])

            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                             lr=f'{current_lr:0.5f}',
                             gpu_memory=f'{mem:0.2f} GB')
        val_scores = np.mean(val_scores, axis=0)
        torch.cuda.empty_cache()
        gc.collect()

    return epoch_loss, val_scores
