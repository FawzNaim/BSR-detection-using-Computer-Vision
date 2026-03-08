import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ==========================================
# MODEL DEFINITION
# ==========================================
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

# ==========================================
# UTILS
# ==========================================
def preprocess_image(path, size=512):
    raw = cv2.imread(path)
    if raw is None: raise FileNotFoundError(f"Missing: {path}")
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size))
    
    # Prep tensor [1, 3, H, W]
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    return rgb, x.unsqueeze(0)

def apply_overlay(img, mask, color=(255, 0, 0), alpha=0.4):
    overlay = img.copy().astype(float)
    if mask is not None:
        idx = mask > 0
        overlay[idx] = alpha * np.array(color) + (1 - alpha) * overlay[idx]
    return overlay.astype(np.uint8)

# ==========================================
# MAIN INFERENCE
# ==========================================
@torch.no_grad()
def run_prediction(img_path, mask_path, ckpt, thresh=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Load Model
    model = UNet().to(device)
    state = torch.load(ckpt, map_location=device)
    # Handle module prefix from DataParallel if needed
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # Prep Data
    rgb, x = preprocess_image(img_path)
    x = x.to(device)

    # Inference
    t0 = time.time()
    logits = model(x)
    dt = time.time() - t0

    # Post-process
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    pred_bin = (prob > thresh).astype(np.uint8)

    # Stats & Metrics
    print(f"Done in {dt:.3f}s. Max prob: {prob.max():.3f}")
    
    gt_mask = None
    if mask_path and os.path.exists(mask_path):
        gt_raw = cv2.imread(mask_path, 0)
        gt_mask = (cv2.resize(gt_raw, (512, 512), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        
        y_true, y_pred = gt_mask.flatten(), pred_bin.flatten()
        iou = np.sum((y_true & y_pred)) / (np.sum((y_true | y_pred)) + 1e-7)
        print(f"Metrics -> IoU: {iou:.4f} | Prec: {precision_score(y_true, y_pred, zero_division=0):.4f}")

    # Visuals
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 4, 1); plt.imshow(rgb); plt.title("Input Image"); plt.axis('off')
    
    if gt_mask is not None:
        plt.subplot(1, 4, 2); plt.imshow(apply_overlay(rgb, gt_mask, (0, 255, 0))); plt.title("GT (Green)"); plt.axis('off')
    
    plt.subplot(1, 4, 3); plt.imshow(apply_overlay(rgb, pred_bin, (255, 0, 0))); plt.title("Pred (Red)"); plt.axis('off')
    
    plt.subplot(1, 4, 4); plt.imshow(prob, cmap='jet'); plt.title("Heatmap"); plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- UPDATE PATHS HERE ---
    IMG = r"" #Add path to the image
    MSK = r"" #Add path to the labeled mask
    CKPT = r"" #Add path to the best .pth file generated after training the model

    run_prediction(IMG, MSK, CKPT)
