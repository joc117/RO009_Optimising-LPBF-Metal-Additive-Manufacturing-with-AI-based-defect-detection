"""
Residual Attention U-Net with ASPP for Defect Segmentation

This script implements a residual attention U-Net architecture augmented with
Atrous Spatial Pyramid Pooling (ASPP). The training pipeline is designed for
robust segmentation under noisy annotations, using a Tversky-based loss and
stochastic weight averaging (SWA).

"""

import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from pycocotools.coco import COCO

import albumentations as A
from albumentations.pytorch import ToTensorV2


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

class Config:

    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = ROOT_DIR / "figshare"

    CHECKPOINT_DIR = Path("/Volumes/Extreme Pro/smp extension")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    BEST_MODEL_PATH = CHECKPOINT_DIR / "ResAttn_ASPP_Final.pth"
    SWA_MODEL_PATH = CHECKPOINT_DIR / "ResAttn_ASPP_SWA.pth"

    IMG_SIZE = (512, 512)
    BATCH_SIZE = 4

    LEARNING_RATE = 1e-4
    EPOCHS = 50

    TV_ALPHA = 0.3
    TV_BETA = 0.7

    DILATE_LABELS = True


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 rates=(1, 6, 12, 18)):
        super().__init__()

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=rates[1], dilation=rates[1], bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=rates[2], dilation=rates[2], bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=rates[3], dilation=rates[3], bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        ])

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [branch(x) for branch in self.branches]
        return self.project(torch.cat(features, dim=1))


class ResidualBlock(nn.Module):
    """
    Standard residual convolutional block.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x) + self.shortcut(x))


class AttentionGate(nn.Module):
    """
    Attention gate for skip-connection filtering.
    """

    def __init__(self, gating_channels: int, skip_channels: int,
                 inter_channels: int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * self.psi(torch.relu(self.W_g(g) + self.W_x(x)))


class ResAttnUNetASPP(nn.Module):
    """
    Residual Attention U-Net with ASPP bottleneck.
    """

    def __init__(self, features=(64, 128, 256, 512)):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.down1 = ResidualBlock(3, features[0])
        self.down2 = ResidualBlock(features[0], features[1])
        self.down3 = ResidualBlock(features[1], features[2])
        self.down4 = ResidualBlock(features[2], features[3])

        self.bottleneck = ASPP(features[3], features[3] * 2)

        self.up4 = nn.ConvTranspose2d(features[3] * 2, features[3], 2, stride=2)
        self.att4 = AttentionGate(features[3], features[3], features[3] // 2)
        self.dec4 = ResidualBlock(features[3] * 2, features[3])

        self.up3 = nn.ConvTranspose2d(features[3], features[2], 2, stride=2)
        self.att3 = AttentionGate(features[2], features[2], features[2] // 2)
        self.dec3 = ResidualBlock(features[2] * 2, features[2])

        self.up2 = nn.ConvTranspose2d(features[2], features[1], 2, stride=2)
        self.att2 = AttentionGate(features[1], features[1], features[1] // 2)
        self.dec2 = ResidualBlock(features[1] * 2, features[1])

        self.up1 = nn.ConvTranspose2d(features[1], features[0], 2, stride=2)
        self.att1 = AttentionGate(features[0], features[0], features[0] // 2)
        self.dec1 = ResidualBlock(features[0] * 2, features[0])

        self.out_conv = nn.Conv2d(features[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.down1(x)
        s2 = self.down2(self.pool(s1))
        s3 = self.down3(self.pool(s2))
        s4 = self.down4(self.pool(s3))

        b = self.bottleneck(self.pool(s4))

        d4 = self.dec4(torch.cat([self.up4(b), self.att4(self.up4(b), s4)], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), self.att3(self.up3(d4), s3)], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.att2(self.up2(d3), s2)], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.att1(self.up1(d2), s1)], dim=1))

        return self.out_conv(d1)

class UnifiedDataset(Dataset):

    def __init__(self, split: str, augment: bool = True):
        self.split = split
        self.img_dir = Config.DATA_DIR / split / "images"
        self.coco = COCO(str(Config.DATA_DIR / split / "_annotations.coco.json"))
        self.ids = list(self.coco.imgs.keys())

        self.transform = A.Compose([
            A.Resize(*Config.IMG_SIZE),
            A.HorizontalFlip(p=0.5) if augment else A.NoOp(),
            A.ColorJitter(brightness=0.1, contrast=0.1, p=0.3) if augment else A.NoOp(),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]

        image = np.array(
            Image.open(self.img_dir / info["file_name"]).convert("RGB")
        )

        mask = np.zeros((info["height"], info["width"]), dtype=np.uint8)
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id)):
            mask = np.maximum(mask, self.coco.annToMask(ann))

        if self.split == "train" and Config.DILATE_LABELS:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        augmented = self.transform(image=image, mask=mask)
        return augmented["image"], augmented["mask"].unsqueeze(0).float()


def train() -> None:
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = ResAttnUNetASPP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    swa_model = optim.swa_utils.AveragedModel(model)
    swa_scheduler = optim.swa_utils.SWALR(optimizer, swa_lr=5e-5)

    train_loader = DataLoader(
        UnifiedDataset("train"),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        UnifiedDataset("valid", augment=False),
        batch_size=1,
        shuffle=False
    )

    best_f1 = 0.0

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            logits = model(images)
            probs = torch.sigmoid(logits).view(logits.size(0), -1)
            targets = masks.view(masks.size(0), -1)

            tp = (probs * targets).sum(dim=1)
            fp = (probs * (1 - targets)).sum(dim=1)
            fn = ((1 - probs) * targets).sum(dim=1)

            loss = 1.0 - (
                (tp + 1e-7) /
                (tp + Config.TV_ALPHA * fp + Config.TV_BETA * fn + 1e-7)
            ).mean()

            loss.backward()
            optimizer.step()

        if epoch > int(Config.EPOCHS * 0.75):
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # Validation
        model.eval()
        tp_v = fp_v = fn_v = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                preds = (
                    torch.sigmoid(model(images.to(device))) > 0.30
                ).float().cpu().numpy()
                gt = masks.numpy()

                tp_v += (preds * gt).sum()
                fp_v += (preds * (1 - gt)).sum()
                fn_v += ((1 - preds) * gt).sum()

        f1 = 2 * tp_v / (2 * tp_v + fp_v + fn_v + 1e-8)
        print(f"Epoch {epoch:03d} | Validation F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    torch.save(swa_model.state_dict(), Config.SWA_MODEL_PATH)
    print(f"Training completed. Best validation F1 = {best_f1:.4f}")

if __name__ == "__main__":
    train()
