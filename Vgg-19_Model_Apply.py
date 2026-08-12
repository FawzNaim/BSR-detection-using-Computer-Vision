# single_image_inference_vgg19_unet_all_images_aligned.py
# Python 3.8/3.9 compatible
#
# Aligned to your VGG19-UNet training that used ALL images (no split).
# IMPORTANT: Inference itself does not depend on whether training used a split or not.
# What matters is that the MODEL ARCH + PREPROCESSING match training.
#
# Training-aligned preprocessing (as in your dataset __getitem__):
# - BGR->RGB
# - resize to 512
# - /255.0
# - NO ImageNet mean/std normalization
#
# Output:
# - 1-channel logits -> sigmoid -> prob map in [0,1]
# - optional GT mask overlay + metrics
#
# Notes:
# - If checkpoint was saved from DataParallel, strips "module." prefixes automatically.
# - Overlays: GT = green, Pred = red

import os
import time
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

from torchvision import models


# ===========================
#  VGG19-UNet (AUTO-SKIP)
#  Matches your training model that inferred skip channels from dummy pass
# ===========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class VGG19UNet(nn.Module):
    """
    Robust VGG19-BN encoder (pool-based skips) + UNet decoder.
    Decoder channel sizes inferred automatically from a dummy forward pass.
    """
    def __init__(self, input_size=512, pretrained=False, freeze_encoder=False):
        super().__init__()
        weights = models.VGG19_BN_Weights.DEFAULT if pretrained else None
        vgg = models.vgg19_bn(weights=weights)
        self.encoder = vgg.features

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Infer skip channels with a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size, input_size)
            skips, bottom = self._encode(dummy)
            if len(skips) < 5:
                raise RuntimeError(f"Unexpected VGG structure: got {len(skips)} pool skips, expected 5.")
            s1, s2, s3, s4, s5 = skips[-5], skips[-4], skips[-3], skips[-2], skips[-1]
            c2, c3, c4, c5 = s2.shape[1], s3.shape[1], s4.shape[1], s5.shape[1]
            cb = bottom.shape[1]

        self.up4  = nn.ConvTranspose2d(cb, c5, 2, stride=2)
        self.dec4 = ConvBlock(c5 + c5, c5)

        self.up3  = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec3 = ConvBlock(c4 + c4, c4)

        self.up2  = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec2 = ConvBlock(c3 + c3, c3)

        self.up1  = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec1 = ConvBlock(c2 + c2, c2)

        self.out = nn.Conv2d(c2, 1, kernel_size=1)

    def _align(self, x, ref):
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def _encode(self, x):
        skips = []
        out = x
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                skips.append(out)
            out = layer(out)
        return skips, out

    def forward(self, x):
        skips, bottom = self._encode(x)
        s1, s2, s3, s4, s5 = skips[-5], skips[-4], skips[-3], skips[-2], skips[-1]

        u4 = self.up4(bottom)
        u4 = self._align(u4, s5)
        d4 = self.dec4(torch.cat([u4, s5], dim=1))

        u3 = self.up3(d4)
        u3 = self._align(u3, s4)
        d3 = self.dec3(torch.cat([u3, s4], dim=1))

        u2 = self.up2(d3)
        u2 = self._align(u2, s3)
        d2 = self.dec2(torch.cat([u2, s3], dim=1))

        u1 = self.up1(d2)
        u1 = self._align(u1, s2)
        d1 = self.dec1(torch.cat([u1, s2], dim=1))

        d1_up = F.interpolate(d1, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(d1_up)  # logits


# ===========================
# Helpers (training-aligned)
# ===========================
def pick_device(device_preference: Optional[str] = None):
    if device_preference:
        if device_preference == "cpu":
            return torch.device("cpu")
        if device_preference.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device_preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_image_rgb_tensor(image_path: str, size: int = 512):
    """
    TRAINING-aligned preprocessing:
    - read BGR -> RGB
    - resize to (size, size)
    - CHW float32 / 255
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb.shape[:2]

    rgb_resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = rgb_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(0)  # NCHW
    return rgb, (h0, w0), x


def load_mask_binary(mask_path: Optional[str], out_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if not mask_path:
        return None
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")
    h, w = out_hw
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def overlay_colored(rgb_uint8: np.ndarray, mask_bin_uint8: Optional[np.ndarray],
                    alpha: float = 0.35, color=(255, 0, 0)) -> np.ndarray:
    if mask_bin_uint8 is None:
        return rgb_uint8.copy()

    mask = mask_bin_uint8.astype(bool)
    out = rgb_uint8.astype(np.float32).copy()

    color_img = np.zeros_like(rgb_uint8, dtype=np.float32)
    color_img[..., 0] = color[0]
    color_img[..., 1] = color[1]
    color_img[..., 2] = color[2]

    out[mask] = alpha * color_img[mask] + (1 - alpha) * out[mask]
    return out.astype(np.uint8)


def compute_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray):
    p, g = pred_bin.flatten(), gt_bin.flatten()
    if g.sum() == 0 and p.sum() == 0:
        return dict(acc=1.0, prec=0.0, rec=0.0, iou=1.0)

    acc = accuracy_score(g, p)
    prec = precision_score(g, p, zero_division=0)
    rec = recall_score(g, p, zero_division=0)
    inter = np.logical_and(g == 1, p == 1).sum()
    uni = np.logical_or(g == 1, p == 1).sum()
    iou = (inter / uni) if uni > 0 else 0.0
    return dict(acc=acc, prec=prec, rec=rec, iou=iou)


# ===========================
# Predict & Show (single image)
# ===========================
@torch.inference_mode()
def run_vgg19_unet_inference(
    image_path: str,
    mask_path: Optional[str] = None,
    *,
    ckpt_path: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
    epoch: Optional[int] = None,
    input_size: int = 512,
    threshold: float = 0.5,
    overlay_alpha: float = 0.35,
    device_preference: Optional[str] = None,
    save_dir: Optional[str] = None,
    save_basename: Optional[str] = None,
    save_overlay: bool = True,
    save_probs: bool = True,
    show_plots: bool = True,
    use_2x2_layout: bool = True,
):
    """
    Provide ONE of:
      - ckpt_path=".../bsr_vgg19_epoch_###.pth" or ".../bsr_vgg19_best_epoch_###.pth"
      - checkpoints_dir=".../checkpoints_vgg19_YYYYMMDD_HHMMSS", epoch=##
        expects filename: bsr_vgg19_epoch_###.pth
    """
    if ckpt_path is None:
        if checkpoints_dir is None or epoch is None:
            raise ValueError("Provide either ckpt_path OR (checkpoints_dir and epoch).")
        ckpt_path = os.path.join(checkpoints_dir, f"bsr_vgg19_epoch_{int(epoch):03d}.pth")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = pick_device(device_preference)
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    # IMPORTANT:
    # - For all-images training, just load your checkpoint.
    # - pretrained=False avoids downloading ImageNet weights; checkpoint overwrites everything anyway.
    model = VGG19UNet(input_size=input_size, pretrained=True, freeze_encoder=False).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = _strip_module_prefix(ckpt)

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"[load_state_dict] missing keys: {missing}")
    if unexpected:
        print(f"[load_state_dict] unexpected keys: {unexpected}")

    model.eval()

    # --- load image (training-aligned preprocessing)
    rgb, (h0, w0), x = load_image_rgb_tensor(image_path, size=input_size)
    x = x.to(device, non_blocking=True)

    # --- inference timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    logits = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    prob_small = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)

    # Resize prob back to original
    prob_full = cv2.resize(prob_small, (w0, h0), interpolation=cv2.INTER_LINEAR)
    pred_bin = (prob_full >= threshold).astype(np.uint8)

    # --- GT (optional)
    gt_bin = load_mask_binary(mask_path, (h0, w0)) if mask_path else None

    # --- overlays: GT green, Pred red
    gt_overlay = overlay_colored(rgb, gt_bin, alpha=overlay_alpha, color=(0, 200, 0)) if gt_bin is not None else None
    pred_overlay = overlay_colored(rgb, pred_bin, alpha=overlay_alpha, color=(255, 0, 0))

    # --- summary stats
    area_pct = 100.0 * float(pred_bin.mean())
    mean_prob_in_mask = float(prob_full[pred_bin.astype(bool)].mean()) if pred_bin.any() else 0.0
    max_prob_in_mask = float(prob_full[pred_bin.astype(bool)].max()) if pred_bin.any() else 0.0
    global_mean_prob = float(prob_full.mean())

    print("\n—— Prediction summary ——")
    print(f"Inference time: {dt:.3f}s")
    print(f"Threshold: {threshold:.2f}")
    print(f"BSR area (% of image): {area_pct:.2f}%")
    print(f"Mean prob (inside pred mask): {mean_prob_in_mask:.3f}")
    print(f"Max  prob (inside pred mask): {max_prob_in_mask:.3f}")
    print(f"Global mean prob: {global_mean_prob:.3f}")

    metrics = None
    if gt_bin is not None:
        metrics = compute_metrics(pred_bin, gt_bin)
        print(f"🧪 Metrics  IoU: {metrics['iou']:.3f} | Prec: {metrics['prec']:.3f} | Rec: {metrics['rec']:.3f} | Acc: {metrics['acc']:.3f}")
    else:
        print("ℹ️ No ground-truth mask provided; metrics skipped.")

    # --- save outputs
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = save_basename if save_basename else os.path.splitext(os.path.basename(image_path))[0]

        if save_probs:
            prob_u8 = np.clip(prob_full * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, f"{base}_pred_prob.png"), prob_u8)

        cv2.imwrite(os.path.join(save_dir, f"{base}_pred_mask.png"), (pred_bin * 255).astype(np.uint8))

        if save_overlay:
            cv2.imwrite(os.path.join(save_dir, f"{base}_overlay_pred.png"),
                        cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR))
            if gt_overlay is not None:
                cv2.imwrite(os.path.join(save_dir, f"{base}_overlay_gt.png"),
                            cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR))

    # --- plot panels
    if show_plots:
        if use_2x2_layout:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
            axes = axes.ravel()
        else:
            fig, axes = plt.subplots(1, 4, figsize=(18, 6))

        axes[0].imshow(rgb); axes[0].set_title("Original"); axes[0].axis("off")
        if gt_overlay is not None:
            axes[1].imshow(gt_overlay); axes[1].set_title("GT Overlay (green)"); axes[1].axis("off")
        else:
            axes[1].imshow(rgb); axes[1].set_title("GT Overlay (none)"); axes[1].axis("off")

        axes[2].imshow(pred_overlay); axes[2].set_title(f"Pred Overlay (red) t={threshold:.2f}"); axes[2].axis("off")

        im = axes[3].imshow(prob_full, cmap="jet", vmin=0, vmax=1)
        axes[3].set_title("Probability (0–1)"); axes[3].axis("off")
        cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        cbar.set_label("BSR probability")

        plt.show()

    return {
        "rgb": rgb,
        "prob": prob_full,
        "pred_bin": pred_bin,
        "overlay_pred_rgb": pred_overlay,
        "overlay_gt_rgb": gt_overlay,
        "summary": {
            "threshold": threshold,
            "area_percent": area_pct,
            "mean_prob_inside_pred_mask": mean_prob_in_mask,
            "max_prob_inside_pred_mask": max_prob_in_mask,
            "global_mean_prob": global_mean_prob,
            "metrics": metrics,
            "inference_seconds": dt,
            "ckpt_path": ckpt_path,
        }
    }


# ===========================
# MAIN (example)
# ===========================
if __name__ == "__main__":
    IMAGE_PATH = r"/content/BSR-detection-using-Computer-Vision/Non_BSR_1_GOM.png"
    GT_MASK    = r""

    # Point to a checkpoint from your ALL-IMAGES training run folder
    CKPT_PATH  = r"/content/BSR-detection-using-Computer-Vision/models/bsr_vgg19_best_epoch_050.pth"

    res = run_vgg19_unet_inference(
        image_path=IMAGE_PATH,
        mask_path=GT_MASK,
        ckpt_path=CKPT_PATH,

        input_size=512,
        threshold=0.5,
        overlay_alpha=0.35,

        save_dir=None,
        show_plots=True,
        use_2x2_layout=True,
    )

    print("Done. Shapes — rgb:", res["rgb"].shape,
          "| prob:", res["prob"].shape,
          "| mask:", res["pred_bin"].shape)
    print("Summary:", res["summary"])
