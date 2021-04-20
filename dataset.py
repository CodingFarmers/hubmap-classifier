from config import args
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import random
import fitlog


def get_train_transform():
    transforms = []
    if args.hflip:
        transforms.append(A.VerticalFlip(p=args.hflip))
    if args.vflip:
        transforms.append(A.HorizontalFlip(p=args.vflip))
    if args.sharpen:
        transforms.append(A.IAASharpen(p=args.sharpen))
    if args.ssr:
        transforms.append(A.ShiftScaleRotate(p=args.ssr))
    if args.channelshuffle:
        transforms.append(A.ChannelShuffle(p=args.channelshuffle))
    # if args.clahe:
    #     transforms.append(A.CLAHE(p=args.clahe))
    if args.oneofcgb:
        transforms.append(
            A.OneOf([
                A.RandomContrast(),
                A.RandomGamma(),
                A.RandomBrightness(),
            ], p=args.oneofcgb))

    transforms.append(A.Resize(256, 256))
    transforms.append(A.Normalize())
    transforms.append(ToTensorV2(p=1))

    return A.Compose(transforms)


def get_val_transform():
    return A.Compose([
        A.Resize(256, 256, always_apply=True),
        A.Normalize(),
        # A.Normalize(mean=[0.67187779, 0.56069541, 0.70091227],
        #             std=[0.21985851, 0.27517871, 0.20858426]),
        ToTensorV2(p=1)
    ], p=1.)


class HuBMAPDataset(Dataset):
    def __init__(self, df, phase, data_root=None, mosaic=False):
        self.img_lists = df['id'].values
        self.labels = df['label'].values
        self.phase = phase
        if phase == 'train':
            self.transform = get_train_transform()
        else:
            self.transform = get_val_transform()
        self.data_root = data_root


    def __getitem__(self, idx):
        name = self.img_lists[idx]
        label = self.labels[idx]

        img = cv2.imread(f"{self.data_root}/train/{name}")
        
        img = self.transform(image=img)['image']
     
        return img, label[..., np.newaxis].astype(np.float32)

    def __len__(self):
        return len(self.img_lists)


def prepare_dataloader(df, trn_idx, val_idx, data_root='./', batch_size=32):
    train_df = df.loc[trn_idx].reset_index(drop=True)
    valid_df = df.loc[val_idx].reset_index(drop=True)

    train_dataset = HuBMAPDataset(train_df, 'train', data_root)
    valid_dataset = HuBMAPDataset(valid_df, 'valid', data_root)

    trn_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size*2,
                            shuffle=False, num_workers=4, pin_memory=True)

    return trn_loader, val_loader
