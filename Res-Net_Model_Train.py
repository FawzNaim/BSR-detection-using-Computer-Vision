import os
import time
import random
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.amp import autocast, GradScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score

# ==========================================
# SETTINGS
# ==========================================
PARAMS = {
    "img_size": 512,
    "batch_size": 8,
    "epochs": 100,
    "lr": 3e-4,
    "wd": 1e-4,
    "pos_weight": 15.0,
    "img_dir": r"", #Add path to the directory with the training images
    "mask_dir": r"", #Add path to the directory with the training image masks
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

def seed_it(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ==========================================
# DATA LOADING
# ==========================================
class BSRDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        
        # Pre-check for oversampling
        self.pos_flags = []
        for f in self.files:
            m = cv2.imread(os.path.join(mask_dir, f.replace(".png", "_label.png")), 0)
            self.pos_flags.append(np.any(m > 0))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, f))
        mask = cv2.imread(os.path.join(self.mask_dir, f.replace(".png", "_label.png")), 0)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask > 0).float().unsqueeze(0)
        return img, mask

# ==========================================
# MODEL SETUP
# ==========================================
def get_deeplab():
    # Load model and swap heads for binary segmentation (1 channel)
    model = deeplabv3_resnet50(weights='DEFAULT')
    
    # Update main classifier
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    # Update aux classifier
    if hasattr(model, 'aux_classifier') and model.aux_classifier:
        aux_in = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(aux_in, 1, kernel_size=1)
        
    return model

def freeze_bn(m):
    """Prevents batchnorm issues with small batch sizes/DeepLab architectures"""
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

# ==========================================
# TRAINING LOOP
# ==========================================
def train():
    seed_it()
    run_id = datetime.now().strftime("%m%d_%H%M")
    ckpt_path = f"checkpoints_resnet_{run_id}"
    os.makedirs(ckpt_path, exist_ok=True)

    # Data
    ds = BSRDataset(PARAMS["img_dir"], PARAMS["mask_dir"], PARAMS["img_size"])
    weights = [5.0 if x else 1.0 for x in ds.pos_flags]
    sampler = WeightedRandomSampler(weights, len(ds))
    loader = DataLoader(ds, batch_size=PARAMS["batch_size"], sampler=sampler, drop_last=True)

    # Model, Opt, Loss
    model = get_deeplab().to(PARAMS["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=PARAMS["lr"], weight_decay=PARAMS["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PARAMS["epochs"])
    scaler = GradScaler()
    
    pos_w = torch.tensor([PARAMS["pos_weight"]]).to(PARAMS["device"])
    
    history = []
    best_iou = 0.0

    

    print(f"Starting Training | Device: {PARAMS['device']}")

    for epoch in range(PARAMS["epochs"]):
        model.train()
        model.apply(freeze_bn) # Safety for DeepLab heads
        
        e_loss, e_iou = [], []
        t0 = time.time()

        for img, mask in loader:
            img, mask = img.to(PARAMS["device"]), mask.to(PARAMS["device"])
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                output = model(img)['out']
                
                # Hybrid Loss: BCE + Dice
                bce = nn.functional.binary_cross_entropy_with_logits(output, mask, pos_weight=pos_w)
                probs = torch.sigmoid(output)
                inter = (probs * mask).sum()
                dice = 1 - (2. * inter + 1) / (probs.sum() + mask.sum() + 1)
                loss = 0.5 * bce + 0.5 * dice

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Quick metrics
            e_loss.append(loss.item())
            preds = (probs > 0.5).float()
            intersection = (preds * mask).sum().item()
            union = (preds + mask).clamp(0, 1).sum().item()
            e_iou.append(intersection / (union + 1e-6))

        scheduler.step()
        
        avg_loss = np.mean(e_loss)
        avg_iou = np.mean(e_iou)
        
        print(f"Epoch [{epoch+1}/{PARAMS['epochs']}] Loss: {avg_loss:.4f} IoU: {avg_iou:.4f} Time: {time.time()-t0:.1f}s")
        
        # Save logic
        history.append({"epoch": epoch+1, "loss": avg_loss, "iou": avg_iou})
        
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), os.path.join(ckpt_path, "best_resnet_bsr.pth"))
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(ckpt_path, f"resnet_ep{epoch+1}.pth"))

    # Log results
    pd.DataFrame(history).to_csv(f"resnet_log_{run_id}.csv", index=False)
    print(f"Done. Best IoU: {best_iou:.4f}")

if __name__ == "__main__":
    train()
