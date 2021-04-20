from config import args
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import datetime
import os

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts, LambdaLR, ExponentialLR

from sklearn.model_selection import GroupKFold
from model import SegModel
from dataset import prepare_dataloader

from pytorch_toolbelt import losses as L
from loss import FocalLoss
from lr_finder import LRFinder
import fitlog


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # True 搜索最适合的卷积算法 加快运算


def calc_dice(probabilities: torch.Tensor,
              truth: torch.Tensor,
              treshold: float = 0.5,
              eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Dice score for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: dice score aka f1.
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert(predictions.shape == truth.shape)
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        # if truth_.sum() == 0:  # 0.909536 --> 0.8393
        #     continue
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(torch.tensor(1.0).unsqueeze(dim=0).cuda())
        else:
            scores.append((intersection / (union + eps)).unsqueeze(dim=0))
    scores = torch.cat(scores)
    return scores.mean()

    num = preds.shape[0]
    preds.reshape(num, -1)
    labels.reshape(num, -1)
    dice = []
    for i in range(num):
        pred = preds[i]
        label = labels[i]

        inter = (pred * label).sum()

        dice += [2 * inter / (pred.sum() + label.sum() + eps)]

    return np.mean(dice)


def train_one_epoch(fold, epoch, model, criterion, optimizer, dataloader, device, scheduler=None):
    model.train()

    running_loss = None
    pbar = tqdm(dataloader)
    for step, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device)
        labels = labels.to(device)
        mixup = args.mixup and (random.random() < args.mixup)
        if mixup:
            lam = np.random.beta(args.alpha, args.alpha)
            index = torch.randperm(imgs.size(0)).cuda()
            imgs = lam * imgs + (1-lam) * imgs[index]

        with autocast():
            preds = model(imgs)
            if not mixup:
                loss = criterion(preds, labels)
            else:
                loss = lam * criterion(preds, labels) + \
                    (1-lam)*criterion(preds, labels[index])

        # fitlog.add_loss(loss.item(), step+epoch*len(pbar), name='train loss')  # 图画的太丑了
        scaler.scale(loss).backward()
        if ((step + 1) % args.accum_iter == 0) or ((step+1) == len(dataloader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if running_loss:
            running_loss = 0.99 * running_loss + loss.item() * 0.01
        else:
            running_loss = loss.item()

        if ((step + 1) % args.verbose == 0) or ((step + 1) == len(train_loader)):
            description = f'epoch {epoch} loss: {running_loss:.4f}'
            pbar.set_description(description)

    scheduler.step()


@torch.no_grad()
def valid_one_epoch(fold, epoch, model, criterion, optimizer, dataloader, device, scheduler=None):
    global min_loss, max_dice, save_dir
    model.eval()

    img_labels, img_preds = [], []
    total_loss, length = .0, 0

    pbar = tqdm(dataloader)
    for step, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device)
        labels = labels.to(device)

        preds = model(imgs)
        img_preds.append(preds.sigmoid())
        img_labels.append(labels)

    preds = torch.cat(img_preds)
    labels = torch.cat(img_labels).type_as(preds)

    dice = calc_dice(preds, labels, args.thersh)
    fitlog.add_metric({"val": {f"fold_{fold}_dice": dice}}, step=epoch)
    if not save_dir:
        save_dir = fitlog.get_log_folder(absolute=True)  # 将模型保存在对应fitlog文件夹下
    if dice > max_dice:
        max_dice = dice
        fitlog.add_best_metric({"val": {f"fold_{fold}_dice": max_dice}})
        torch.save(model.state_dict(
        ), f'{save_dir}/{args.structure}_{args.encoder}_fold{fold}_best.pth')

    print(f'fold {fold} epoch {epoch}, valid dice {dice:.4f}')


if __name__ == "__main__":

    seed_everything(seed=args.seed)

    save_dir = None
    group = GroupKFold(n_splits=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fitlog 初始化
    if args.find_lr:
        fitlog.debug()
    fitlog.set_log_dir('aug/')
    fitlog.add_hyper(args)
    # fitlog.add_hyper({'v2':True})
    # fitlog.commit(__file__)  #自动保存对应版本训练代码

    epochs = args.epochs

    if os.path.exists(f'{args.data_root}/train.csv'):
        train = pd.read_csv(f'{args.data_root}/train.csv')
    else:
        imgs = os.listdir(f'{args.data_root}/train/')
        train = pd.DataFrame(imgs, columns=['id'])
        train['parent'] = train['id'].apply(lambda x: x.split('_')[0])
        train.to_csv(f'{args.data_root}/train.csv', index=False)

    dice = []
    for fold, (trn_idx, val_idx) in enumerate(group.split(train, groups=train['parent'])):
        min_loss, max_dice = 1e10, 0

        trn_loader, val_loader = prepare_dataloader(
            train, trn_idx, val_idx, data_root=args.data_root, batch_size=args.bs)

        model = SegModel(encoder=args.encoder,
                         structure=args.structure).to(device)
        model = torch.nn.DataParallel(model)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
        # criterion = L.DiceLoss(mode='binary', smooth=1)  # 9056
        criterion = torch.nn.BCEWithLogitsLoss() 

        # criterion = L.JointLoss(L.DiceLoss(
        #     mode='binary', smooth=1), torch.nn.BCEWithLogitsLoss(), 0.5, 0.5) # 9060
        # criterion = L.BinaryLovaszLoss() # 9044
        # criterion = FocalLoss(gamma=2)
        # criterion = L.LovaszLoss()

        if args.find_lr:
            lr_finder = LRFinder(model, optimizer, criterion, device=device)
            lr_finder.range_test(trn_loader, start_lr=args.start_lr,
                                 end_lr=args.end_lr, num_iter=100, accumulation_steps=args.accum_iter)
            fig_name = 'lr_curve.png'
            lr_finder.plot(fig_name)
            lr_finder.reset()
            break

        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=1, eta_min=1e-6)
        scaler = GradScaler()
        for epoch in range(epochs):
            train_one_epoch(fold, epoch, model, criterion, optimizer,
                            trn_loader, device, scheduler=scheduler)
            valid_one_epoch(fold, epoch, model, criterion, optimizer,
                            val_loader, device, scheduler=None)
        print(f'fold {fold} max dice {max_dice}')
        dice.append(max_dice.cpu().numpy())

        if not args.nfold:
            break

    if args.nfold:
        fitlog.add_best_metric({"val": {"mean_dice": np.mean(dice)}})
    fitlog.finish()
