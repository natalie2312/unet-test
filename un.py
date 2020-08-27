import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import cv2
import glob
import math
import random
from scipy import misc
import numpy as np
from torchvision.transforms import *



save_dir = "./data"

# u-net


def conv3x3(in_channels, out_channels, stride=1, padding=1, activate="relu"):
    layers = []
    layers.append(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=padding
        )
    )
    layers.append(nn.BatchNorm2d(out_channels))
    if activate == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif activate == "sigmoid":
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def double_conv3x3(in_channels, out_channels, stride=1, padding=1, activate="relu"):
    return nn.Sequential(
        conv3x3(in_channels, out_channels, stride, padding=1, activate=activate),
        conv3x3(out_channels, out_channels, stride, padding=1, activate=activate),
    )


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            double_conv3x3(in_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpSample, self).__init__()
        self.bilinear = bilinear
        self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.net = double_conv3x3(in_channels, out_channels)

    def forward(self, front, later):
        if self.bilinear:
            later = F.interpolate(
                later, scale_factor=2, mode="bilinear", align_corners=True
            )
        else:
            later = self.conv_trans(later)
        h_diff = front.size()[2] - later.size()[2]
        w_diff = front.size()[3] - later.size()[3]
        later = F.pad(
            later,
            pad=(w_diff // 2, w_diff - w_diff // 2, h_diff // 2, h_diff - h_diff // 2),
            mode="constant",
            value=0,
        )
        x = torch.cat([front, later], dim=1)
        x = self.net(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inconv = double_conv3x3(1, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        self.down4 = DownSample(512, 512)
        self.up1 = UpSample(1024, 256)
        self.up2 = UpSample(512, 128)
        self.up3 = UpSample(256, 64)
        self.up4 = UpSample(128, 64)
        self.outconv = double_conv3x3(64, 1, activate="sigmoid")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.outconv(x)

        return x


# data loader


class Loader(Dataset):
    def __init__(self, split, save_dir):
        image_dir = os.path.join(save_dir, split, "image")
        label_dir = os.path.join(save_dir, split, "label")
        self.images, self.labels = self._read_data(image_dir, label_dir)
        self.trans = Compose(
            [
                ToPILImage(),
                RandomHorizontalFlip(0.5),
                RandomVerticalFlip(0.5),
                RandomResizedCrop(572),
                ToTensor(),
            ]
        )

    def _read_data(self, image_dir, label_dir):
        images, labels = [], []
        img_fns = os.listdir(image_dir)
        for img_fn in img_fns:
            image_path = os.path.join(image_dir, img_fn)
            label_path = os.path.join(label_dir, img_fn)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.0
            images.append(image[np.newaxis, :])
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) / 255.0
            label[label > 0.5] = 1
            label[label <= 0.5] = 0
            labels.append(label[np.newaxis, :])
        return images, labels

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if np.random.uniform(0, 1) < 0.5:
            image = image[:, ::-1, :]
            label = label[:, ::-1, :]
        if np.random.uniform(0, 1) < 0.5:
            image = image[:, :, ::-1]
            label = label[:, :, ::-1]

        image = np.ascontiguousarray(image)
        label = np.ascontiguousarray(label)
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label

    def __len__(self):
        return len(self.images)


# functions


def accuracy(logit, target, threshold=0.5):
    logit[logit > threshold] = 1
    logit[logit <= threshold] = 0
    return (logit.long() == target.long()).float().mean().item()


def adjust_lr(optimizer, lr_gamma=0.1):
    for (i, param_group) in enumerate(optimizer.param_groups):
        param_group["lr"] = param_group["lr"] * lr_gamma
    return optimizer.state_dict()["param_groups"][0]["lr"]


# train


def step(split, epoch, model, criterion, optimizer, batch_size=1, cuda=False):

    if split == "train":
        model.train()
    else:
        model.eval()

    loader = data.DataLoader(
        Loader(split, save_dir),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    epoch_loss, epoch_acc, n_batchs = 0, 0, 0
    for i, (image, label) in enumerate(loader):
        n_batchs += image.size(0)
        if cuda:
            image = image.cuda("cuda:1")
            label = label.cuda("cuda:1")
        logit = model(image)
        logit = logit.flatten()
        label = label.flatten()
        loss = criterion(logit, label)
        if split == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += accuracy(logit, label) * 100
    epoch_loss /= n_batchs
    epoch_acc /= n_batchs
    return epoch_loss, epoch_acc


batch_size = 1
pretrained = False
# set Cuda = True to enable GPU calculation
cuda = False
start_epoch = 1
end_epoch = 1000
lr_decay = 30

print("init")

criterion = nn.BCEWithLogitsLoss()
model = UNet()

optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, momentum=0.99, weight_decay=0.0005
)

print("optimizer")

if cuda:
    model = model.cuda("cuda:1")
    criterion = criterion.cuda("cuda:1")
if pretrained:
    model.load_state_dict(torch.load("model.pth"))

print("train")
train_losses, val_losses = [], []
for epoch in range(start_epoch, end_epoch):
    if epoch % lr_decay == 0:
        lr = adjust_lr(optimizer)
        print("adjust LR to {:.4f}".format(lr))
    tepoch_loss, tepoch_acc = step(
        "train", epoch, model, criterion, optimizer, batch_size, cuda=cuda
    )
    vepoch_loss, vepoch_acc = step(
        "val", epoch, model, criterion, optimizer, batch_size, cuda=cuda
    )
    train_losses.append(tepoch_loss)
    val_losses.append(vepoch_loss)
    print(
        "epoch {0:} finished, tloss:{1:.4f} [{2:.2f}%]  vloss:{3:.4f} [{4:.2f}%]".format(
            epoch, tepoch_loss, tepoch_acc, vepoch_loss, vepoch_acc
        )
    )
torch.save(model.state_dict(), "model.pth")
print("done!")
