# ====================================================
# Directory settings
# ====================================================
import os
os.environ['CUDA_VISIBLFalseE_DEVICES'] = '2,3'

OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

TRAIN_PATH = 'ranzcr/train'

# ====================================================
# CFG
# ====================================================
class CFG:
    debug=False
    device='GPU' # ['TPU', 'GPU']
    nprocs=1 # [1, 8]
    print_freq=100
    num_workers=4
    model_name='resnet200d' # resnet200d
    size=640
    scheduler='CosineAnnealingLR' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    teacher='../input/ranzcr-resnet200d-3-stage-training-step1/resnet200d_320_fold0_best_loss_cpu.pth'
    student='../input/ranzcr-resnet200d-3-stage-training-step2/resnet200d_320_fold0_best_loss.pth'
    epochs=15
    factor=0.2 # ReduceLROnPlateau
    patience=4 # ReduceLROnPlateau
    eps=1e-6 # ReduceLROnPlateau
    T_max=10 # CosineAnnealingLR
    T_0=epochs # CosineAnnealingWarmRestarts
    lr=7.74e-5 # 1e-4
    min_lr=1e-6
    batch_size=32 # 64
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=416
    target_size=11
    target_cols=['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                 'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
                 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                 'Swan Ganz Catheter Present']
    n_fold=5
    trn_fold=[0] # [0, 1, 2, 3, 4]
    train=True
    

# ====================================================
# Library
# ====================================================
# import os
import ast
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip, 
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout, 
    IAAAdditiveGaussianNoise, Transpose, HueSaturationValue, CoarseDropout
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import timm

from torch.cuda.amp import autocast, GradScaler

from loss import FocalLoss

import warnings 
warnings.filterwarnings('ignore')

# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        score = roc_auc_score(y_true[:,i], y_pred[:,i])
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score, scores


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)


train = pd.read_csv('ranzcr/train.csv')
folds = pd.read_csv('train_folds.csv')
train_annotations = pd.read_csv('ranzcr/train_annotations.csv')

if CFG.debug:
    CFG.epochs = 1
    train = train.sample(n=10000, random_state=CFG.seed).reset_index(drop=True)
    folds = folds.sample(n=10000, random_state=CFG.seed).reset_index(drop=True)
# ====================================================
# Dataset
# ====================================================
class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['StudyInstanceUID'].values
        self.labels = df[CFG.target_cols].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f'{TRAIN_PATH}/{file_name}.jpg'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        label = torch.tensor(self.labels[idx]).float()
        return image, label
    
# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):
    if data == 'train':
        import albumentations
        return Compose([
           albumentations.RandomResizedCrop(CFG.size, CFG.size, scale=(0.9, 1), p=1),
           albumentations.HorizontalFlip(p=0.5),
           albumentations.ShiftScaleRotate(p=0.5),
           albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
           albumentations.RandomBrightnessContrast(brightness_limit=(-0.2,0.2), contrast_limit=(-0.2, 0.2), p=0.7),
           albumentations.CLAHE(clip_limit=(1,4), p=0.5),
           albumentations.OneOf([
               albumentations.OpticalDistortion(distort_limit=1.0),
               albumentations.GridDistortion(num_steps=5, distort_limit=1.),
               albumentations.ElasticTransform(alpha=3),
           ], p=0.2),
           albumentations.OneOf([
               albumentations.GaussNoise(var_limit=[10, 50]),
               albumentations.GaussianBlur(),
               albumentations.MotionBlur(),
               albumentations.MedianBlur(),
           ], p=0.2),
          albumentations.Resize(CFG.size, CFG.size),
          albumentations.OneOf([
              albumentations.JpegCompression(),
              albumentations.Downscale(scale_min=0.1, scale_max=0.15),
          ], p=0.2),
          albumentations.IAAPiecewiseAffine(p=0.2),
          albumentations.IAASharpen(p=0.2),
          albumentations.Cutout(max_h_size=int(CFG.size * 0.1), max_w_size=int(CFG.size * 0.1), num_holes=5, p=0.5),
          albumentations.Normalize(),
          ToTensorV2()])
    
    # if data == 'train':
    #     return Compose([
    #         # Resize(int(CFG.size * 1.25), int(CFG.size * 1.25)),
    #         #Resize(CFG.size, CFG.size),
    #         RandomResizedCrop(CFG.size, CFG.size, scale=(0.85, 1.0)),
    #         # HorizontalFlip(p=0.5),
    #         # RandomBrightnessContrast(p=0.2, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
    #         # HueSaturationValue(p=0.2),
    #         ShiftScaleRotate(p=0.2, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
    #         # # CoarseDropout(p=0.2), # max_holes=8, max_height=8, max_width=8
    #         # Cutout(p=0.2, max_h_size=16, max_w_size=16, fill_value=(0., 0., 0.), num_holes=16),
    #         Normalize(
    #             mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225],
    #         ),
    #         ToTensorV2(),
    #     ])

    elif data == 'valid':
        return Compose([
            Resize(CFG.size, CFG.size),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),

        ])

# ====================================================
# MODEL
# ====================================================
class CustomModel(nn.Module):
    def __init__(self, model_name='resnet200d_320', pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.global_pool = nn.Identity()
        
        if 'tf' in model_name:
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Identity()
        elif 'vit' in model_name:
            num_ftrs = self.model.head.in_features
            self.model.head = nn.Identity()
        else:
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_ftrs, CFG.target_size)

    @autocast()
    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return features, pooled_features, output


# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    scaler = GradScaler()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # mixup
        lam = np.random.beta(0.2, 0.2)
        index = torch.randperm(images.size(0)).cuda()
        images = lam * images + (1-lam) * images[index]
        with autocast():
            _, _, y_preds = model(images)
            # mixup loss
            loss = lam * criterion(y_preds, labels) + (1 - lam) * criterion(y_preds, labels[index])
            # loss = criterion(y_preds, labels)
            # record loss
            losses.update(loss.item(), batch_size)
            if CFG.gradient_accumulation_steps > 1:
                loss = loss / CFG.gradient_accumulation_steps
            scaler.scale(loss).backward()
            grad_norm = 0# torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            if (step + 1) % CFG.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Elapsed {remain:s} '
                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    'Grad: {grad_norm:.4f}  '
                    #'LR: {lr:.6f}  '
                    .format(
                    epoch+1, step, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses,
                    remain=timeSince(start, float(step+1)/len(train_loader)),
                    grad_norm=grad_norm,
                    #lr=scheduler.get_lr()[0],
                    ))
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    trues = []
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            _, _, y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        trues.append(labels.to('cpu').numpy())
        preds.append(y_preds.sigmoid().to('cpu').numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                    'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                    'Elapsed {remain:s} '
                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    .format(
                    step, len(valid_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses,
                    remain=timeSince(start, float(step+1)/len(valid_loader)),
                    ))
    trues = np.concatenate(trues)
    predictions = np.concatenate(preds)
    return losses.avg, predictions, trues

# ====================================================
# Train loop
# ====================================================
# def train_loop(fold, trn_idx, val_idx):
def train_loop(folds, fold):

    if CFG.device == 'GPU':
        LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values


    train_dataset = TrainDataset(train_folds, 
                                 transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, 
                                 transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset, 
                                batch_size=CFG.batch_size, 
                                shuffle=True, 
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, 
                                batch_size=CFG.batch_size * 2, 
                                shuffle=False, 
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CustomModel(CFG.model_name, pretrained=False)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(f'{CFG.model_name}_student_fold{fold}_best_score.pth', map_location=torch.device('cpu'))['model'])
    # model.load_state_dict(torch.load(f'0.9647/{CFG.model_name}_no_hflip_fold{fold}_best_score.pth', map_location=torch.device('cpu'))['model'])
    model.to(device)

    # criterion = nn.BCEWithLogitsLoss()
    criterion = FocalLoss(alpha=1, gamma=6)

    # optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    optimizer = SGD(model.parameters(), lr=1e-2, weight_decay=CFG.weight_decay, momentum=0.9)


    find_lr =  False
    if find_lr:
        from lr_finder import LRFinder
        lr_finder = LRFinder(model, optimizer, criterion, device=device)
        lr_finder.range_test(train_loader, start_lr=1e-2, end_lr=1e0, num_iter=100, accumulation_steps=1)
        
        fig_name = f'{CFG.model_name}_lr_finder.png'
        lr_finder.plot(fig_name)
        lr_finder.reset()
        return 
    scheduler = get_scheduler(optimizer)
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=1e-3)
    swa_start = 9

    # ====================================================
    # loop
    # ====================================================

    best_score = 0.
    best_loss = np.inf
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()
        
        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
                
        # eval
        # avg_val_loss, preds, _ = valid_fn(valid_loader, model, criterion, device)
        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            elif isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()
            elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()

        # scoring
        avg_val_loss, preds, _ = valid_fn(valid_loader, model, criterion, device)
        score, scores = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {np.round(scores, decimals=4)}')

        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict()},OUTPUT_DIR+f'{CFG.model_name}_no_hflip_fold{fold}_best_score.pth')
        
        # if avg_val_loss < best_loss:
        #     best_loss = avg_val_loss
        #     LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
        #     torch.save({'model': model.state_dict(), 
        #                 'preds': preds},
        #                 OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_loss.pth')
    
    torch.optim.swa_utils.update_bn(train_loader, swa_model)
    avg_val_loss, preds, _ = valid_fn(valid_loader, swa_model, criterion, device)
    score, scores = get_score(valid_labels, preds)
    LOGGER.info(f'Save swa Score: {score:.4f} Model')
    torch.save({'model': swa_model.state_dict()}, OUTPUT_DIR+f'swa_{CFG.model_name}_fold{fold}_{score:.4f}.pth')
    # if CFG.nprocs != 8:
    #     check_point = torch.load(OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best_score.pth')
    #     for c in [f'pred_{c}' for c in CFG.target_cols]:
    #         valid_folds[c] = np.nan
    #     try:
    #         valid_folds[[f'pred_{c}' for c in CFG.target_cols]] = check_point['preds']
    #     except:
    #         pass

    return 


# ====================================================
# main
# ====================================================
def main():

    """
    Prepare: 1.train  2.folds
    """

    def get_result(result_df):
        preds = result_df[[f'pred_{c}' for c in CFG.target_cols]].values
        labels = result_df[CFG.target_cols].values
        score, scores = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}  Scores: {np.round(scores, decimals=4)}')
    
    if CFG.train:
        # train 
        oof_df = pd.DataFrame()
        # group_kfold = GroupKFold(n_splits=5)
        # groups = train['PatientID'].values
        # folds = group_kfold.split(train, train.iloc[:, 1:-1], groups)

        # for fold, (trn_idx, val_idx) in enumerate(folds):
        #     if fold > 0:
        #         break
        for fold in range(CFG.n_fold):
            if fold > 0:
                break
            train_loop(folds, fold)

if __name__ == '__main__':
    main()
