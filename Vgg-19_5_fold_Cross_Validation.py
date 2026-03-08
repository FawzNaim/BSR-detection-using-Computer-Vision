import os
import gc
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold
import random
import pandas as pd

from torchvision import models

# ------------------ Reproducibility ------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------ Dataset ------------------
class BSRSliceDataset(Dataset):
    """
    Loads images/masks, resizes to 512x512.
    Normalizes images to ImageNet mean/std to match VGG-19 pretraining.
    Also computes dataset stats for imbalance handling + per-image has_pos for oversampling.
    """
    def __init__(self, img_dir, mask_dir, size=512):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_list = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.png')])
        self.size = size

        total_pixels = 0
        total_pos = 0
        self.has_pos = []

        for img_name in self.img_list:
            mask_name = img_name.replace('.png', '_label.png')
            mask_path = os.path.join(mask_dir, mask_name)

            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")

            m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
            m_bin = (m > 0).astype(np.uint8)
            pos = int(m_bin.sum())
            self.has_pos.append(pos > 0)

            total_pos += pos
            total_pixels += size * size

        self.pos_frac = (total_pos / total_pixels) if total_pixels > 0 else 0.0
        print(f"📊 Dataset positive pixel fraction ≈ {self.pos_frac*100:.3f}% "
              f"({total_pos}/{total_pixels}). "
              f"{sum(self.has_pos)} / {len(self.img_list)} images contain any positives.")

        # ImageNet normalization for VGG
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)

        mask_name = img_name.replace('.png', '_label.png')
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))
        mask  = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        mask = (mask > 0).astype('float32')

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

# ------------------ VGG19-based U-Net ------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class VGG19_UNet(nn.Module):
    def __init__(self, pretrained=True, freeze_encoder=False):
        super().__init__()
        vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1 if pretrained else None)
        feats = vgg.features

        pool_ids = [6, 13, 26, 39, 52]
        self.enc1 = feats[:pool_ids[0]]
        self.pool1 = feats[pool_ids[0]]
        self.enc2 = feats[pool_ids[0]+1:pool_ids[1]]
        self.pool2 = feats[pool_ids[1]]
        self.enc3 = feats[pool_ids[1]+1:pool_ids[2]]
        self.pool3 = feats[pool_ids[2]]
        self.enc4 = feats[pool_ids[2]+1:pool_ids[3]]
        self.pool4 = feats[pool_ids[3]]
        self.enc5 = feats[pool_ids[3]+1:pool_ids[4]]

        self.bottleneck = DoubleConv(512, 512)
        self.up4 = UpBlock(512, 512, 512)
        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

        if freeze_encoder:
            for m in [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]:
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x):
        x1 = self.enc1(x); p1 = self.pool1(x1)
        x2 = self.enc2(p1); p2 = self.pool2(x2)
        x3 = self.enc3(p2); p3 = self.pool3(x3)
        x4 = self.enc4(p3); p4 = self.pool4(x4)
        x5 = self.enc5(p4)

        b = self.bottleneck(x5)
        d4 = self.up4(b, x4)
        d3 = self.up3(d4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)
        return self.out_conv(d1)

# ------------------ Losses ------------------
def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1,2,3))
    denom = (probs + targets).sum(dim=(1,2,3)) + eps
    dice = (2.0 * intersection) / denom
    return 1.0 - dice.mean()

def build_bce_logits(pos_frac, device, clip=30.0):
    pos_frac = max(1e-6, min(0.5, float(pos_frac)))
    raw = (1.0 - pos_frac) / pos_frac
    w = min(clip, raw)
    print(f"🧮 Using pos_weight={w:.2f} (raw={raw:.1f}, clip={clip}) from pos_frac={pos_frac:.6f}")
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w], device=device))

# ------------------ Metrics ------------------
def calculate_metrics_from_logits(logits, target, thresh=0.5):
    probs = torch.sigmoid(logits)
    pred = (probs > thresh).float()

    pred_np = pred.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()

    if np.sum(target_np) == 0:
        accuracy = accuracy_score(target_np, pred_np)
        return accuracy, 0.0, 0.0, 0.0

    accuracy = accuracy_score(target_np, pred_np)
    precision = precision_score(target_np, pred_np, zero_division=0)
    recall = recall_score(target_np, pred_np, zero_division=0)

    intersection = np.logical_and(target_np, pred_np)
    union = np.logical_or(target_np, pred_np)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0

    return accuracy, precision, recall, iou

# ------------------ AMP (with fallback) ------------------
try:
    from torch.amp import autocast, GradScaler
    _amp_device = 'cuda'
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    _amp_device = 'cuda'

# ------------------ Per-epoch evaluation ------------------
def evaluate_epoch(model, loader, criterion_bce, device, thresh=0.5, use_amp=True):
    model.eval()
    total_loss = total_bce = total_dice = 0.0
    total_accuracy = total_precision = total_recall = total_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)

            with autocast(_amp_device, enabled=(use_amp and device.type == 'cuda')):
                logits = model(img)
                bce = criterion_bce(logits, mask)
                dsc = dice_loss_from_logits(logits, mask)
                loss = 0.5 * bce + 0.5 * dsc

            acc, prec, rec, iou = calculate_metrics_from_logits(logits, mask, thresh=thresh)

            total_loss += loss.item()
            total_bce += bce.item()
            total_dice += dsc.item()
            total_accuracy += acc
            total_precision += prec
            total_recall += rec
            total_iou += iou
            num_batches += 1

    if num_batches == 0:
        return (0.0,)*7

    return (
        total_loss / num_batches,
        total_bce / num_batches,
        total_dice / num_batches,
        total_accuracy / num_batches,
        total_precision / num_batches,
        total_recall / num_batches,
        total_iou / num_batches
    )

# ------------------ Fold membership CSV ------------------
def write_fold_membership_csv(dataset, fold_id, train_idx, val_idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"fold_{fold_id:02d}_membership.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "split", "idx", "image_filename", "mask_filename", "has_positive"])
        for i in train_idx:
            img = dataset.img_list[i]
            w.writerow([fold_id, "train", i, img, img.replace(".png", "_label.png"), int(dataset.has_pos[i])])
        for i in val_idx:
            img = dataset.img_list[i]
            w.writerow([fold_id, "val", i, img, img.replace(".png", "_label.png"), int(dataset.has_pos[i])])
    print(f"🧾 Fold membership written: {out_path}")

# ------------------ Train one fold ------------------
def train_one_fold(dataset, fold_id, train_idx, val_idx, device,
                   num_epochs=50, lr=3e-4, weight_decay=1e-4, batch_size=8,
                   pos_multiplier=5.0, use_amp=True, eval_thresh=0.5,
                   out_dir="cv_outputs"):

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    # Oversampling only on TRAIN
    if any(dataset.has_pos[i] for i in train_idx):
        weights = [pos_multiplier if dataset.has_pos[i] else 1.0 for i in train_idx]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_idx), replacement=True)
        shuffle_train = False
        print(f"🔁 Fold {fold_id}: Oversampling ON (pos_multiplier={pos_multiplier})")
    else:
        sampler = None
        shuffle_train = True
        print(f"ℹ️ Fold {fold_id}: Oversampling OFF (no positives in train split)")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle_train,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        drop_last=False
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        drop_last=False
    )

    # Recompute pos_frac on TRAIN for pos_weight (optional, but better for CV)
    total_pos = 0
    total_pix = len(train_idx) * dataset.size * dataset.size
    for i in train_idx:
        mask_path = os.path.join(dataset.mask_dir, dataset.img_list[i].replace(".png", "_label.png"))
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (dataset.size, dataset.size), interpolation=cv2.INTER_NEAREST)
        total_pos += int((m > 0).sum())
    train_pos_frac = (total_pos / total_pix) if total_pix > 0 else dataset.pos_frac
    if train_pos_frac <= 0:
        train_pos_frac = 0.0059

    # Model (new each fold)
    model = VGG19_UNet(pretrained=True, freeze_encoder=False).to(device)
    criterion_bce = build_bce_logits(train_pos_frac, device, clip=30.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler(_amp_device, enabled=(use_amp and device.type == 'cuda'))

    log_csv = os.path.join(out_dir, f"fold_{fold_id:02d}_training_log_{ts}.csv")
    fieldnames = [
        'fold','epoch',
        'loss','bce','dice','accuracy','precision','recall','iou',
        'val_loss','val_bce','val_dice','val_accuracy','val_precision','val_recall','val_iou',
        'epoch_time','avg_time_per_batch','learning_rate'
    ]
    with open(log_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_val_iou = -1.0
    best_ckpt = None

    print(f"\n🚀 Starting Fold {fold_id} | Train={len(train_idx)} Val={len(val_idx)} | epochs={num_epochs}\n")

    for epoch in range(1, num_epochs+1):
        model.train()
        t0 = time.time()

        total_loss = total_bce = total_dice = 0.0
        total_acc = total_prec = total_rec = total_iou = 0.0
        nb = 0

        for img, mask in train_loader:
            img, mask = img.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(_amp_device, enabled=(use_amp and device.type == 'cuda')):
                logits = model(img)
                bce = criterion_bce(logits, mask)
                dsc = dice_loss_from_logits(logits, mask)
                loss = 0.5*bce + 0.5*dsc

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc, prec, rec, iou = calculate_metrics_from_logits(logits, mask, thresh=eval_thresh)
            total_loss += loss.item(); total_bce += bce.item(); total_dice += dsc.item()
            total_acc += acc; total_prec += prec; total_rec += rec; total_iou += iou
            nb += 1

        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0
        avg_time_per_batch = epoch_time / nb if nb > 0 else 0.0

        avg_loss = total_loss/nb if nb else 0.0
        avg_bce  = total_bce/nb if nb else 0.0
        avg_dice = total_dice/nb if nb else 0.0
        avg_acc  = total_acc/nb if nb else 0.0
        avg_prec = total_prec/nb if nb else 0.0
        avg_rec  = total_rec/nb if nb else 0.0
        avg_iou  = total_iou/nb if nb else 0.0
        lr_now = optimizer.param_groups[0]["lr"]

        (val_loss, val_bce, val_dice, val_acc, val_prec, val_rec, val_iou) = evaluate_epoch(
            model, val_loader, criterion_bce, device, thresh=eval_thresh, use_amp=use_amp
        )

        print(
            f"Fold {fold_id} Epoch {epoch}/{num_epochs} | "
            f"Train IoU={avg_iou:.4f} Val IoU={val_iou:.4f} | "
            f"Train Loss={avg_loss:.4f} Val Loss={val_loss:.4f} | LR={lr_now:.2e}"
        )

        with open(log_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow({
                'fold': fold_id, 'epoch': epoch,
                'loss': avg_loss, 'bce': avg_bce, 'dice': avg_dice,
                'accuracy': avg_acc, 'precision': avg_prec, 'recall': avg_rec, 'iou': avg_iou,
                'val_loss': val_loss, 'val_bce': val_bce, 'val_dice': val_dice,
                'val_accuracy': val_acc, 'val_precision': val_prec, 'val_recall': val_rec, 'val_iou': val_iou,
                'epoch_time': epoch_time, 'avg_time_per_batch': avg_time_per_batch, 'learning_rate': lr_now
            })

        # save best checkpoint (by VAL IoU)
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_ckpt = os.path.join(out_dir, f"fold_{fold_id:02d}_best.pth")
            torch.save(model.state_dict(), best_ckpt)
            print(f"💾 Fold {fold_id}: New best Val IoU={best_val_iou:.4f} -> {best_ckpt}")

    return log_csv, best_ckpt, best_val_iou

# ------------------ Main: 5-Fold CV ------------------
if __name__ == "__main__":
    set_seed(42)

    img_dir = r"" #Add path to directory with training images
    mask_dir = r"" #Add path to directory with training masks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU detected. Using CPU for training.")

    dataset = BSRSliceDataset(img_dir, mask_dir, size=512)

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    out_dir = "cv_outputs_vgg19_unet"
    os.makedirs(out_dir, exist_ok=True)

    all_indices = np.arange(len(dataset))
    summary_rows = []

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(all_indices), start=1):
        # write which images are in this fold
        write_fold_membership_csv(dataset, fold_id, train_idx, val_idx, out_dir)

        # train fold
        log_csv, best_ckpt, best_val_iou = train_one_fold(
            dataset=dataset,
            fold_id=fold_id,
            train_idx=train_idx.tolist(),
            val_idx=val_idx.tolist(),
            device=device,
            num_epochs=50,
            lr=3e-4,
            weight_decay=1e-4,
            batch_size=8,
            pos_multiplier=5.0,
            use_amp=True,
            eval_thresh=0.5,
            out_dir=out_dir
        )

        summary_rows.append({
            "fold": fold_id,
            "best_val_iou": best_val_iou,
            "best_checkpoint": best_ckpt,
            "log_csv": log_csv
        })

        # IMPORTANT: cleanup between folds
        del best_ckpt
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    # write summary CSV
    summary_csv = os.path.join(out_dir, f"cv_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"\n✅ CV complete. Summary written to: {summary_csv}")
