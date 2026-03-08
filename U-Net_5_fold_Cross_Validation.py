import os
import cv2
import time
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold

# --- Globals & Setup ---
SEED = 42
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Model Blocks ---
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        # Encoder
        self.e1 = ConvBlock(3, 64)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)
        self.bot = ConvBlock(512, 512)
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d3  = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2  = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d1  = ConvBlock(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        b  = self.bot(self.pool(s4))
        
        x = self.up3(b)
        x = self.d3(torch.cat([x, s3], 1))
        x = self.up2(x)
        x = self.d2(torch.cat([x, s2], 1))
        x = self.up1(x)
        x = self.d1(torch.cat([x, s1], 1))
        return self.out(x)

# --- Dataset Logic ---
class BSRDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512):
        self.img_dir, self.mask_dir = img_dir, mask_dir
        self.size = size
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        # Meta-data for oversampling
        self.has_pos = []
        pos_px, total_px = 0, 0
        for f in self.files:
            m = cv2.imread(os.path.join(mask_dir, f.replace('.png', '_label.png')), 0)
            m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
            count = np.sum(m > 0)
            self.has_pos.append(count > 0)
            pos_px += count
            total_px += (size * size)
        
        self.pos_ratio = pos_px / (total_px + 1e-7)
        print(f"Dataset: {len(self.files)} images | Pos px: {self.pos_ratio*100:.3f}%")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, name))
        mask = cv2.imread(os.path.join(self.mask_dir, name.replace('.png', '_label.png')), 0)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask > 0).float().unsqueeze(0)
        return img, mask

# --- Training / Eval Helpers ---
def dice_loss(pred, target):
    p = torch.sigmoid(pred)
    num = 2. * (p * target).sum(dim=(1,2,3))
    den = (p + target).sum(dim=(1,2,3)) + 1e-6
    return 1. - (num / den).mean()

def compute_metrics(logits, targets, thresh=0.5):
    p = (torch.sigmoid(logits) > thresh).float().cpu().numpy().flatten()
    t = targets.cpu().numpy().flatten()
    if np.sum(t) == 0: return 0, 0, 0 # Handle empty masks
    
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return inter/(union + 1e-7), precision_score(t, p, zero_division=0), recall_score(t, p, zero_division=0)

# --- Fold Runner ---
def run_fold(fold_id, t_loader, v_loader, device, epochs=100, out_dir="cv_runs"):
    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    
    # Static weight based on dataset stats
    bce_w = torch.tensor([25.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=bce_w)
    
    best_iou = 0
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"fold_{fold_id}_log.csv")
    
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss', 'val_iou', 'val_prec', 'val_rec'])
        
        for epoch in range(epochs):
            model.train()
            losses = []
            for x, y in t_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                with torch.amp.autocast('cuda'):
                    out = model(x)
                    loss = 0.5 * criterion(out, y) + 0.5 * dice_loss(out, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
            
            # Validation
            model.eval()
            ious, precs, recs = [], [], []
            with torch.no_grad():
                for x, y in v_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    m = compute_metrics(out, y)
                    ious.append(m[0]); precs.append(m[1]); recs.append(m[2])
            
            v_iou = np.mean(ious)
            sched.step()
            writer.writerow([epoch+1, np.mean(losses), v_iou, np.mean(precs), np.mean(recs)])
            
            if v_iou > best_iou:
                best_iou = v_iou
                torch.save(model.state_dict(), os.path.join(out_dir, f"best_f{fold_id}.pth"))
            
            print(f"F{fold_id} | E{epoch+1:03d} | Loss: {np.mean(losses):.4f} | IoU: {v_iou:.4f}")
            
    return best_iou

if __name__ == "__main__":
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ds = BSRDataset(
        img_dir=r"", #Add path to directory to the training images
        mask_dir=r"" #Add path to directory to the training masks
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []
    
    for f_idx, (t_idx, v_idx) in enumerate(kf.split(np.arange(len(ds))), 1):
        print(f"\n--- Fold {f_idx} ---")
        
        # Weighted Sampler for Fold
        f_pos = [ds.has_pos[i] for i in t_idx]
        w = [5.0 if p else 1.0 for p in f_pos]
        sampler = WeightedRandomSampler(w, len(t_idx))
        
        t_ld = DataLoader(Subset(ds, t_idx), batch_size=8, sampler=sampler, num_workers=2, pin_memory=True)
        v_ld = DataLoader(Subset(ds, v_idx), batch_size=8, shuffle=False, num_workers=2)
        
        score = run_fold(f_idx, t_ld, v_ld, device)
        cv_scores.append(score)
        
        # Cleanup to prevent GPU OOM
        torch.cuda.empty_cache()

    print(f"\nMean CV IoU: {np.mean(cv_scores):.4f}")
