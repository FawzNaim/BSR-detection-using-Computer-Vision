import os
import time
import csv
import random
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# MODEL ARCHITECTURE
# ---------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, drop=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        )
    def forward(self, x): return self.layers(x)

class UNet(nn.Module):
    def __init__(self, drop=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.e1 = ConvBlock(3, 64, drop)
        self.e2 = ConvBlock(64, 128, drop)
        self.e3 = ConvBlock(128, 256, drop)
        self.e4 = ConvBlock(256, 512, drop)
        self.bot = ConvBlock(512, 512, drop)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d3  = ConvBlock(512, 256, drop)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2  = ConvBlock(256, 128, drop)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d1  = ConvBlock(128, 64, drop)

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        
        b = self.bot(self.pool(s4))

        x = self.up3(b)
        x = self.d3(torch.cat([x, s3], 1))
        x = self.up2(x)
        x = self.d2(torch.cat([x, s2], 1))
        x = self.up1(x)
        x = self.d1(torch.cat([x, s1], 1))
        return self.final(x)

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
class BSRDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.filenames = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.size = size
        
        # Pre-scan for positive pixels to handle imbalance
        self.pos_flags = []
        pos_pixels = 0
        for f in self.filenames:
            m = cv2.imread(os.path.join(mask_dir, f.replace('.png', '_label.png')), 0)
            m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
            has_pos = np.any(m > 0)
            self.pos_flags.append(has_pos)
            pos_pixels += np.sum(m > 0)
        
        self.pos_ratio = pos_pixels / (len(self.filenames) * size * size)
        print(f"Dataset Scan: {sum(self.pos_flags)}/{len(self.filenames)} imgs have BSR. Ratio: {self.pos_ratio:.6f}")

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        f = self.filenames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, f)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, f.replace('.png', '_label.png')), 0)
        
        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask > 0).float().unsqueeze(0)
        return img, mask

# ---------------------------------------------------------
# UTILS & LOSS
# ---------------------------------------------------------
def get_dice_loss(logits, target):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    union = (p + target).sum(dim=(1, 2, 3))
    return 1 - ((2. * inter + 1e-6) / (union + 1e-6)).mean()

def compute_stats(logits, target, t=0.5):
    p = (torch.sigmoid(logits) > t).float().cpu().numpy().flatten()
    y = target.cpu().numpy().flatten()
    if np.sum(y) == 0: return accuracy_score(y, p), 0.0, 0.0, 0.0
    
    acc = accuracy_score(y, p)
    prec = precision_score(y, p, zero_division=0)
    rec = recall_score(y, p, zero_division=0)
    iou = np.sum((y == 1) & (p == 1)) / (np.sum((y == 1) | (p == 1)) + 1e-7)
    return acc, prec, rec, iou

# ---------------------------------------------------------
# MAIN TRAINING LOOP
# ---------------------------------------------------------
def main():
    # Setup
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    img_path = r"" #Add path to the directory with training images
    mask_path = r"" #Add path to the directory with training masks for the images
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ds = BSRDataset(img_path, mask_path)
    
    # Oversampling logic
    weights = [5.0 if f else 1.0 for f in ds.pos_flags]
    sampler = WeightedRandomSampler(weights, len(ds), replacement=True)
    loader = DataLoader(ds, batch_size=8, sampler=sampler, num_workers=2, pin_memory=True)

    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Weight for BCE based on ratio
    pos_w = torch.tensor([(1.0 - ds.pos_ratio)/ds.pos_ratio]).to(device)
    pos_w = torch.clamp(pos_w, max=20.0)
    criterion_bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    
    scaler = GradScaler('cuda')
    log_dir = f"run_{datetime.now().strftime('%m%d_%H%M')}"
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "history.csv")
    
    best_iou = 0
    print(f"Starting training on {device}...")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'iou', 'prec', 'rec', 'lr'])

        for epoch in range(100):
            model.train()
            m_loss, m_iou, m_prec, m_rec = [], [], [], []
            t0 = time.time()

            for imgs, masks in loader:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad(set_to_none=True)

                with autocast('cuda'):
                    out = model(imgs)
                    loss = 0.5 * criterion_bce(out, masks) + 0.5 * get_dice_loss(out, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                acc, prec, rec, iou = compute_stats(out, masks)
                m_loss.append(loss.item())
                m_iou.append(iou)
                m_prec.append(prec)
                m_rec.append(rec)

            scheduler.step()
            
            # Epoch Summary
            avg_loss, avg_iou = np.mean(m_loss), np.mean(m_iou)
            avg_prec, avg_rec = np.mean(m_prec), np.mean(m_rec)
            lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | Rec: {avg_rec:.4f} | Time: {time.time()-t0:.1f}s")
            writer.writerow([epoch+1, avg_loss, avg_iou, avg_prec, avg_rec, lr])
            f.flush()

            # Save state
            torch.save(model.state_dict(), os.path.join(log_dir, f"last.pth"))
            if avg_iou > best_iou:
                best_iou = avg_iou
                torch.save(model.state_dict(), os.path.join(log_dir, "best.pth"))

    print(f"Done. Log saved to {log_dir}")

if __name__ == "__main__":
    main()
