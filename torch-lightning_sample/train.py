import os
import random
import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import MNIST

from model import CustomModel
from transform import make_transform
from dataset import ImageBaseDataset
from utils import seed_fix

# Set random seed

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--dataset', type=str, default='base')
parser.add_argument('--save_path', type=str, default='saved')
parser.add_argument('--num_workers', type=int, default=0)

parser.add_argument('--label_type', type=str, default='all')

parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--test_size', type=float, default=0.2)

parser.add_argument('--model', type=str, default='tf_efficientnet_b0_ns')
parser.add_argument('--under_age', type=int, default=30)
parser.add_argument('--age', type=int, default=60)
# parser.add_argument('--f1_weight', type=float, default=0.5)
parser.add_argument('--label_smoothing', type=float, default=0.01)

parser.add_argument('--RandomBrightnessContrast', type=bool, default=False)
parser.add_argument('--HueSaturationValue', type=bool, default=False)
parser.add_argument('--RGBShift', type=bool, default=False)
parser.add_argument('--RandomGamma', type=bool, default=False)
parser.add_argument('--HorizontalFlip', type=bool, default=False)
parser.add_argument('--ImageCompression', type=bool, default=False)
parser.add_argument('--ShiftScaleRotate', type=bool, default=False)

args = parser.parse_args()


if __name__ == '__main__':
    seed_fix(args.seed)
    train_transform, test_transform = make_transform(args)
    dataset = ImageBaseDataset(data_dir='train/train/images', transform=test_transform)
    train_dataset, valid_dataset = dataset.split_dataset()

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(valid_dataset, batch_size=128, num_workers=4)

    # model
    model = CustomModel(model_name=args.model, num_classes=18)

    # training
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=2, accelerator="dp")
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)



