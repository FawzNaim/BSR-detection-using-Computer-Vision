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
import torchvision.models.segmentation as models

# --- Config & Seeds ---
SEED = 42
def setup_env():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True # Better for static input sizes

class BSRDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        # Pre-scan for sampling weights
        self.has_pos = []
        pos_pixels, total_pixels = 0, 0
        
        for f in self.files:
            m_path = os.path.join(mask_dir, f.replace('.png', '_label.png'))
            mask = cv2.imread(m_path, 0)
            if mask is None: continue
            
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
            count = np.sum(mask > 0)
            self.has_pos.append(count > 0)
            pos_pixels += count
            total_pixels += (size * size)
            
        self.pos_weight = (total_pixels - pos_pixels) / (pos_pixels + 1e-6)
        print(f"Dataset Ready: {len(self.files)} images. Pos Weight: {self.pos_weight:.2f}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, fname))
        mask = cv2.imread(os.path.join(self.mask_dir, fname.replace('.png', '_label.png')), 0)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask > 0).float().unsqueeze(0)
        return img, mask

# --- Model Factory ---
def get_segmentation_model(name="deeplabv3_resnet50"):
    if name == "deeplabv3_resnet50":
        model = models.deeplabv3_resnet50(weights='DEFAULT')
        model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, 1, 1)
        if model.aux_classifier:
            model.aux_classifier[-1] = nn.Conv2d(model.aux_classifier[-1].in_channels, 1, 1)
    else:
        model = models.fcn_resnet50(weights='DEFAULT')
        model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, 1, 1)
    return model

# --- Core Logic ---
def get_metrics(logits, targets, thresh=0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
    preds = (probs > thresh).astype(np.uint8)
    truth = targets.detach().cpu().numpy().flatten().astype(np.uint8)
    
    # Intersection over Union
    inter = np.logical_and(preds, truth).sum()
    union = np.logical_or(preds, truth).sum()
    iou = inter / (union + 1e-7)
    
    return {
        "iou": iou,
        "acc": accuracy_score(truth, preds),
        "prec": precision_score(truth, preds, zero_division=0),
        "rec": recall_score(truth, preds, zero_division=0)
    }

def train_fold(fold_idx, train_loader, val_loader, device, epochs=100):
    model = get_segmentation_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Combined Loss: BCE + Dice
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]).to(device)) 
    scaler = torch.amp.GradScaler('cuda')
    
    best_iou = 0
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = []
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                output = model(imgs)['out']
                loss = criterion(output, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss.append(loss.item())
            
        # Validation
        model.eval()
        val_ious = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)['out']
                val_ious.append(get_metrics(out, masks)['iou'])
        
        avg_val_iou = np.mean(val_ious)
        scheduler.step()
        
        print(f"Fold {fold_idx} | Epoch {epoch+1} | Loss: {np.mean(train_loss):.4f} | Val IoU: {avg_val_iou:.4f}")
        
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), f"checkpoints/best_f{fold_idx}.pth")
            
    return best_iou

if __name__ == "__main__":
    setup_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = BSRDataset(
        img_dir=r"", #Add path to the directory with training images
        mask_dir=r"" #Add path to the directory with training masks
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []

    for f_idx, (t_idx, v_idx) in enumerate(kf.split(dataset)):
        print(f"\n--- Starting Fold {f_idx+1} ---")
        
        # Sampler for class balance
        fold_has_pos = [dataset.has_pos[i] for i in t_idx]
        weights = [5.0 if x else 1.0 for x in fold_has_pos]
        sampler = WeightedRandomSampler(weights, len(t_idx))
        
        t_loader = DataLoader(Subset(dataset, t_idx), batch_size=8, sampler=sampler, num_workers=2, pin_memory=True)
        v_loader = DataLoader(Subset(dataset, v_idx), batch_size=8, shuffle=False, num_workers=2)
        
        score = train_fold(f_idx+1, t_loader, v_loader, device)
        results.append(score)
        
    print(f"\nCV Average IoU: {np.mean(results):.4f}")
