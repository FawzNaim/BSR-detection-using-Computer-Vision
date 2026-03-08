import os
import time
import random
import cv2
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from torch.amp import autocast, GradScaler
from sklearn.metrics import precision_score, recall_score, accuracy_score

# --- Config & Hyperparams ---
CONFIG = {
    "img_size": 512,
    "batch_size": 8,
    "epochs": 50,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "pos_multiplier": 5.0,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "img_dir": r"" #Add path to the directory with the images,
    "mask_dir": r"" #Add path to the directory with the masks
}

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# --- Model Architecture ---

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

class BSR_VGG_UNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Using standard VGG19-BN backbone
        vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT if pretrained else None)
        features = list(vgg.features.children())
        
        # Explicit slices to define encoder stages for skip connections
        self.enc1 = nn.Sequential(*features[:7])   
        self.enc2 = nn.Sequential(*features[7:14])  
        self.enc3 = nn.Sequential(*features[14:27]) 
        self.enc4 = nn.Sequential(*features[27:40]) 
        self.enc5 = nn.Sequential(*features[40:53]) 
        
        self.center = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True))
        
        self.dec5 = DecoderBlock(512, 512, 512)
        self.dec4 = DecoderBlock(512, 512, 512)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        s5 = self.enc5(s4)
        
        out = self.center(s5)
        
        out = self.dec5(out, s5)
        out = self.dec4(out, s4)
        out = self.dec3(out, s3)
        out = self.dec2(out, s2)
        out = self.dec1(out, s1)
        
        return self.final(F.interpolate(out, size=x.shape[-2:], mode="bilinear"))

# --- Data Handling ---

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith('.png')])
        
        self.has_mask = []
        for f in self.files:
            m_path = os.path.join(mask_dir, f.replace('.png', '_label.png'))
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            self.has_mask.append(np.any(m > 0))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = cv2.imread(os.path.join(self.img_dir, name))
        mask = cv2.imread(os.path.join(self.mask_dir, name.replace('.png', '_label.png')), 0)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        
        img = img.transpose(2, 0, 1) / 255.0
        mask = (mask > 0).astype(np.float32)
        
        return torch.from_numpy(img).float(), torch.from_numpy(mask).float().unsqueeze(0)

# --- Loss & Metrics ---

def get_loss(logits, targets, bce_weight):
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=bce_weight)
    
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum()
    union = probs.sum() + targets.sum()
    dice = 1 - (2. * inter + 1e-7) / (union + 1e-7)
    
    return 0.5 * bce + 0.5 * dice

def compute_metrics(logits, targets):
    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy().flatten()
    actual = targets.cpu().numpy().flatten()
    
    if np.sum(actual) == 0:
        return accuracy_score(actual, preds), 0.0, 0.0, 0.0
        
    acc = accuracy_score(actual, preds)
    prec = precision_score(actual, preds, zero_division=0)
    rec = recall_score(actual, preds, zero_division=0)
    
    inter = np.logical_and(actual, preds).sum()
    union = np.logical_or(actual, preds).sum()
    iou = inter / union if union > 0 else 0.0
    
    return acc, prec, rec, iou

# --- Execution ---

def main():
    seed_everything()
    
    ds = SegmentationDataset(CONFIG['img_dir'], CONFIG['mask_dir'], CONFIG['img_size'])
    weights = [CONFIG['pos_multiplier'] if m else 1.0 for m in ds.has_mask]
    sampler = WeightedRandomSampler(weights, len(ds))
    
    loader = DataLoader(ds, batch_size=CONFIG['batch_size'], sampler=sampler, num_workers=2, pin_memory=True)
    
    model = BSR_VGG_UNet().to(CONFIG['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    scaler = GradScaler()
    
    bce_pos_weight = torch.tensor([15.0]).to(CONFIG['device']) 
    
    best_iou = 0
    run_id = datetime.now().strftime("%m%d_%H%M")
    os.makedirs(f"checkpoints_{run_id}", exist_ok=True)
    history = []

    print(f"Starting training on {CONFIG['device']}...")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        start = time.time()
        metrics_list = []
        epoch_losses = []

        for imgs, masks in loader:
            imgs, masks = imgs.to(CONFIG['device']), masks.to(CONFIG['device'])
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                logits = model(imgs)
                loss = get_loss(logits, masks, bce_pos_weight)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_losses.append(loss.item())
            metrics_list.append(compute_metrics(logits, masks))

        scheduler.step()
        
        avg_loss = np.mean(epoch_losses)
        avg_metrics = np.mean(metrics_list, axis=0)
        duration = time.time() - start
        
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] Loss: {avg_loss:.4f} | IoU: {avg_metrics[3]:.4f} | Time: {duration:.1f}s")
        history.append([epoch+1, avg_loss, *avg_metrics])
        
        if avg_metrics[3] > best_iou:
            best_iou = avg_metrics[3]
            torch.save(model.state_dict(), f"checkpoints_{run_id}/best_model.pth")

    df = pd.DataFrame(history, columns=['epoch', 'loss', 'acc', 'prec', 'rec', 'iou'])
    df.to_csv(f"train_log_{run_id}.csv", index=False)
    print("Training Complete.")

if __name__ == "__main__":
    main()
