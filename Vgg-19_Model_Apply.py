import os
import time
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ==========================================
# MODEL ARCHITECTURE (Must match Training)
# ==========================================
class DecBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
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
    def __init__(self):
        super().__init__()
        vgg = models.vgg19_bn(weights=None) # Weights loaded via checkpoint
        f = list(vgg.features.children())
        
        self.enc1 = nn.Sequential(*f[:7])
        self.enc2 = nn.Sequential(*f[7:14])
        self.enc3 = nn.Sequential(*f[14:27])
        self.enc4 = nn.Sequential(*f[27:40])
        self.enc5 = nn.Sequential(*f[40:53])
        
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2), 
            nn.Conv2d(512, 512, 3, padding=1), 
            nn.ReLU(inplace=True)
        )
        
        self.dec5 = DecBlock(512, 512, 512)
        self.dec4 = DecBlock(512, 512, 512)
        self.dec3 = DecBlock(512, 256, 256)
        self.dec2 = DecBlock(256, 128, 128)
        self.dec1 = DecBlock(128, 64, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x); s2 = self.enc2(s1); s3 = self.enc3(s2)
        s4 = self.enc4(s3); s5 = self.enc5(s4)
        b = self.bottleneck(s5)
        d = self.dec5(b, s5); d = self.dec4(d, s4); d = self.dec3(d, s3)
        d = self.dec2(d, s2); d = self.dec1(d, s1)
        return self.out(F.interpolate(d, size=x.shape[-2:], mode="bilinear"))

# ==========================================
# UTILS
# ==========================================
def preprocess_image(path, size=512):
    raw_bgr = cv2.imread(path)
    if raw_bgr is None:
        raise FileNotFoundError(f"Missing image at {path}")
    
    h, w = raw_bgr.shape[:2]
    rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size))
    
    # Normalization (Matches training)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    return rgb, tensor.unsqueeze(0), (h, w)

def get_overlay(img, mask, color=(255, 0, 0), alpha=0.4):
    """Applies a colored mask overlay to an image."""
    mask = mask.astype(bool)
    overlay = img.copy()
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * np.array(color)
    return overlay.astype(np.uint8)

# ==========================================
# INFERENCE LOGIC
# ==========================================
@torch.no_grad()
def predict_bsr(img_path, ckpt_path, mask_path=None, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = BSR_VGG_UNet().to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    
    # Handle potential DataParallel 'module.' prefix
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    # 2. Preprocess
    orig_rgb, input_tensor, (orig_h, orig_w) = preprocess_image(img_path)
    input_tensor = input_tensor.to(device)
    
    # 3. Forward Pass
    start_t = time.time()
    logits = model(input_tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    inf_time = time.time() - start_t
    
    # 4. Post-process (Resize back to original dimensions)
    full_probs = cv2.resize(probs, (orig_w, orig_h))
    pred_mask = (full_probs > threshold).astype(np.uint8)
    
    # 5. Visualization & Metrics
    results = {
        "time": inf_time,
        "mask": pred_mask,
        "probs": full_probs
    }

    plt.figure(figsize=(16, 6))
    
    # Original + Pred Overlay
    plt.subplot(1, 3, 1)
    plt.title("Prediction Overlay")
    plt.imshow(get_overlay(orig_rgb, pred_mask))
    plt.axis('off')

    # Probability Map
    plt.subplot(1, 3, 2)
    plt.title("Confidence Map")
    plt.imshow(full_probs, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # Ground Truth Comparison (if available)
    if mask_path:
        gt = cv2.imread(mask_path, 0)
        gt = (cv2.resize(gt, (orig_w, orig_h)) > 0).astype(np.uint8)
        
        # Simple IoU calculation
        inter = np.logical_and(gt, pred_mask).sum()
        union = np.logical_or(gt, pred_mask).sum()
        iou = inter / union if union > 0 else 0.0
        
        plt.subplot(1, 3, 3)
        plt.title(f"Ground Truth (IoU: {iou:.3f})")
        plt.imshow(get_overlay(orig_rgb, gt, color=(0, 255, 0)))
        plt.axis('off')
        results['iou'] = iou

    plt.tight_layout()
    plt.show()
    
    print(f"Inference complete in {inf_time:.3f}s")
    return results

if __name__ == "__main__":
    # Settings
    IMG_URL  = r"" #Add path to the image
    GT_URL   = r"" #Add path to the labeled mask
    MODEL  = r""   #Add path to the best .pth file generated after training the model

    res = predict_bsr(IMG_URL, WEIGHTS, mask_path=GT_URL)
