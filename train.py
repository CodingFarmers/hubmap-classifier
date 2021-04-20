from config import args
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import datetime
import os

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, CosineAnnealingWarmRestarts, LambdaLR, ExponentialLR
from torch.optim.swa_utils import AveragedModel, SWALR

from sklearn.model_selection import GroupKFold
from dataset import prepare_dataloader

import timm
from loss import FocalLoss
from lr_finder import LRFinder
import fitlog

from sklearn.metrics import roc_auc_score, log_loss, accuracy_score





def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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
    global min_loss, max_acc, save_dir
    model.eval()

    img_labels, img_preds = [], []
    total_loss, length = .0, 0

    pbar = tqdm(dataloader)
    for step, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device)
        labels = labels.to(device)

        preds = model(imgs)
        # img_preds.append(torch.argmax(preds, dim=1).detach().cpu().numpy())
        img_preds.append((preds.sigmoid() > 0.5).detach().cpu().numpy())
        img_labels.append(labels.detach().cpu().numpy())


    img_preds = np.concatenate(img_preds)
    img_labels = np.concatenate(img_labels)
    acc = (img_preds == img_labels).mean()
    fitlog.add_metric({"val": {f"fold_{fold}_acc": acc}}, step=epoch)
    save_dir = fitlog.get_log_folder(absolute=True)  
    if acc > max_acc:
        max_acc = acc
        fitlog.add_best_metric({"val": {f"fold_{fold}_acc": max_acc}})
        torch.save(model.state_dict(
        ), f'{save_dir}/{args.model}_fold{fold}_best.pth')

    print(f'fold {fold} epoch {epoch}, valid acc {acc:.4f}')


class Model(nn.Module):
    def __init__(self, model_name, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        if 'tf' in model_name:
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, n_class)
        elif 'vit' in model_name:
            num_ftrs = self.model.head.in_features
            self.model.head = nn.Linear(num_ftrs, n_class)
        else:
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, n_class)

    @autocast()
    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':

    seed_everything(args.seed)

    epochs = args.epochs
    train = pd.read_csv(f'{args.data_root}/train.csv')

    save_dir = None
    group = GroupKFold(n_splits=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fitlog 初始化
    if args.find_lr:
        fitlog.debug()
    fitlog.set_log_dir('logs/')
    fitlog.add_hyper(args)

    acc = []
    for fold, (trn_idx, val_idx) in enumerate(group.split(train, groups=train['parent'])):
        min_loss, max_acc = 1e10, 0.

        trn_loader, val_loader = prepare_dataloader(
            train, trn_idx, val_idx, data_root=args.data_root, batch_size=args.bs)

        model = Model(args.model, n_class=1, pretrained=True).to(device)
        model = torch.nn.DataParallel(model)

        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss() 

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.find_lr:
            lr_finder = LRFinder(model, optimizer, criterion, device=device)
            lr_finder.range_test(trn_loader, start_lr=args.start_lr,
                                 end_lr=args.end_lr, num_iter=100, accumulation_steps=args.accum_iter)
            fig_name = 'lr_curve.png'
            lr_finder.plot(fig_name)
            lr_finder.reset()
            break

        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=epochs, T_mult=1, eta_min=1e-6)
        scaler = GradScaler()
        for epoch in range(epochs):
            train_one_epoch(fold, epoch, model, criterion, optimizer,
                            trn_loader, device, scheduler=scheduler)
            valid_one_epoch(fold, epoch, model, criterion, optimizer,
                            val_loader, device, scheduler=None)
        print(f'fold {fold} max acc {max_acc}')
        acc.append(max_acc)

        if not args.nfold:
            break

    if args.nfold:
        fitlog.add_best_metric({"val": {"mean_acc": np.mean(acc)}})
    fitlog.finish()
