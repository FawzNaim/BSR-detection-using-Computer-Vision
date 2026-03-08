import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ==========================================
# MODEL BUILDER
# ==========================================
def load_resnet_model(model_name, ckpt_path, device, aux_loss=True):
    # Initialize base model
    if model_name == "deeplabv3_resnet50":
        model = deeplabv3_resnet50(weights=None, aux_loss=aux_loss)
    else:
        model = fcn_resnet50(weights=None, aux_loss=aux_loss)

    # Manual layer swap for binary output (1 channel)
    # Direct index access is more 'human-written' for specific architectures
    in_ch = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_ch, 1, kernel_size=1)
    
    if aux_loss and hasattr(model, 'aux_classifier'):
        aux_in = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(aux_in, 1, kernel_size=1)

    # Load weights
    print(f"Loading weights from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    
    # Handle DataParallel 'module.' prefix if it exists
    if any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

# ==========================================
# PROCESSING HELPERS
# ==========================================
def preprocess(img_path, size=512):
    bgr = cv2.imread(img_path)
    if bgr is None: raise ValueError("Image path invalid")
    
    h0, w0 = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # Standard resize/normalize match to training
    resized = cv2.resize(rgb, (size, size))
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    return rgb, (h0, w0), tensor.unsqueeze(0)

def get_overlay(img, mask, alpha=0.4, color=(255, 0, 0)):
    """Simple color overlay on RGB image"""
    overlay = img.copy().astype(float)
    if mask is not None:
        idx = mask.astype(bool)
        overlay[idx] = alpha * np.array(color) + (1 - alpha) * overlay[idx]
    return overlay.astype(np.uint8)

# ==========================================
# MAIN INFERENCE FUNCTION
# ==========================================
@torch.no_grad()
def run_prediction(image_path, mask_path=None, **cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Model
    model = load_resnet_model(
        cfg.get('model_name', 'deeplabv3_resnet50'),
        cfg['ckpt'],
        device,
        aux_loss=cfg.get('aux_loss', True)
    )

    # 2. Prep Image
    rgb_orig, (h0, w0), input_batch = preprocess(image_path, cfg.get('size', 512))
    input_batch = input_batch.to(device)

    # 3. Predict
    t0 = time.time()
    output = model(input_batch)['out']
    runtime = time.time() - t0
    
    # 4. Post-process
    prob_map = torch.sigmoid(output).squeeze().cpu().numpy()
    prob_full = cv2.resize(prob_map, (w0, h0))
    pred_mask = (prob_full > cfg.get('thresh', 0.5)).astype(np.uint8)

    # 5. Metrics & Visualization
    print(f"\nResults for: {os.path.basename(image_path)}")
    print(f"Time: {runtime:.3f}s | BSR Area: {pred_mask.mean()*100:.2f}%")

    gt_mask = None
    if mask_path:
        gt_raw = cv2.imread(mask_path, 0)
        gt_mask = (cv2.resize(gt_raw, (w0, h0), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        
        # Calculate stats
        y_true, y_pred = gt_mask.flatten(), pred_mask.flatten()
        iou = np.sum((y_true & y_pred)) / (np.sum((y_true | y_pred)) + 1e-7)
        print(f"IoU: {iou:.4f} | Prec: {precision_score(y_true, y_pred, zero_division=0):.4f}")

    # Plotting logic
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 4, 1); plt.imshow(rgb_orig); plt.title("Input"); plt.axis('off')
    
    pred_viz = get_overlay(rgb_orig, pred_mask, color=(255, 0, 0))
    plt.subplot(1, 4, 2); plt.imshow(pred_viz); plt.title("Prediction (Red)"); plt.axis('off')
    
    if gt_mask is not None:
        gt_viz = get_overlay(rgb_orig, gt_mask, color=(0, 255, 0))
        plt.subplot(1, 4, 3); plt.imshow(gt_viz); plt.title("Ground Truth (Green)"); plt.axis('off')
    
    plt.subplot(1, 4, 4); plt.imshow(prob_full, cmap='jet'); plt.title("Confidence Map"); plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# CONFIGURATION
# ==========================================
if __name__ == "__main__":
    # Update these paths manually
    CONFIG = {
        "ckpt": r"", #Add path to the best .pth file generated after training the model
        "image": r"", #Add path to the image
        "mask": r"", #Add path to the labeled mask
        "model_name": "deeplabv3_resnet50",
        "thresh": 0.5,
        "size": 512,
        "aux_loss": True
    }

    run_prediction(CONFIG["image"], mask_path=CONFIG["mask"], **CONFIG)
