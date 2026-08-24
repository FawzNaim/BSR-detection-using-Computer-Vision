import os
import time
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torchvision import models


# ===========================
#  VGG19-UNet (AUTO-SKIP)
#  (Matches your training model that inferred skip channels from dummy pass)
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
    Decoder channel sizes inferred automatically from a dummy forward pass
    so your checkpoints load cleanly and you avoid skip mismatches.
    """
    def __init__(self, input_size=512, pretrained=True, freeze_encoder=False):
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

        # Decoder (channels tied to inferred skips)
        self.up4 = nn.ConvTranspose2d(cb, c5, 2, stride=2)
        self.dec4 = ConvBlock(c5 + c5, c5)

        self.up3 = nn.ConvTranspose2d(c5, c4, 2, stride=2)
        self.dec3 = ConvBlock(c4 + c4, c4)

        self.up2 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec2 = ConvBlock(c3 + c3, c3)

        self.up1 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
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
# Helpers (match training)
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
    x = rgb_resized.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW
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


def overlay_colored(rgb_uint8: np.ndarray, mask_bin_uint8: np.ndarray,
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
    """
    Strict pixel-wise segmentation metrics.
    IoU here is NOT distance-tolerant.
    """
    p = pred_bin.flatten()
    g = gt_bin.flatten()

    acc = accuracy_score(g, p)
    prec = precision_score(g, p, zero_division=0)
    rec = recall_score(g, p, zero_division=0)
    f1 = f1_score(g, p, zero_division=0)

    inter = np.logical_and(g == 1, p == 1).sum()
    union = np.logical_or(g == 1, p == 1).sum()
    iou = inter / union if union > 0 else 1.0

    return dict(
        acc=acc,
        prec=prec,
        rec=rec,
        f1=f1,
        iou=iou
    )


def compute_tolerance_metrics(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    tolerance_px: int = 2
):
    """
    Distance-tolerant Precision, Recall, and F1 for thin BSR masks.

    A predicted BSR pixel is counted as correct if it lies within
    `tolerance_px` Euclidean pixels of any ground-truth BSR pixel.

    Likewise, a ground-truth BSR pixel is counted as recovered if it lies
    within `tolerance_px` pixels of any predicted BSR pixel.

    This affects evaluation only. It does NOT modify the model prediction
    and does NOT require retraining.
    """
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)

    n_pred = int(pred.sum())
    n_gt = int(gt.sum())

    if n_gt == 0 and n_pred == 0:
        return dict(
            tolerance_px=tolerance_px,
            prec_tol=1.0,
            rec_tol=1.0,
            f1_tol=1.0
        )

    if n_gt == 0 or n_pred == 0:
        return dict(
            tolerance_px=tolerance_px,
            prec_tol=0.0,
            rec_tol=0.0,
            f1_tol=0.0
        )

    # Distance from every pixel to the nearest GT BSR pixel
    dist_to_gt = cv2.distanceTransform(
        (~gt).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE
    )

    # Distance from every pixel to the nearest predicted BSR pixel
    dist_to_pred = cv2.distanceTransform(
        (~pred).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE
    )

    matched_pred = pred & (dist_to_gt <= tolerance_px)
    matched_gt = gt & (dist_to_pred <= tolerance_px)

    prec_tol = matched_pred.sum() / n_pred
    rec_tol = matched_gt.sum() / n_gt

    f1_tol = (
        2.0 * prec_tol * rec_tol / (prec_tol + rec_tol)
        if (prec_tol + rec_tol) > 0
        else 0.0
    )

    return dict(
        tolerance_px=tolerance_px,
        prec_tol=float(prec_tol),
        rec_tol=float(rec_tol),
        f1_tol=float(f1_tol)
    )


# ===========================
#  Predict & Show
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
    distance_tolerances=(2, 3, 4, 5),
    overlay_alpha: float = 0.35,
    device_preference: Optional[str] = None,
    save_dir: Optional[str] = None,
    save_basename: Optional[str] = None,
    save_overlay: bool = True,
    save_probs: bool = True,
    show_plots: bool = True,
    use_2x2_layout: bool = True,
    show_with_cv2: bool = False,
) -> Dict[str, Any]:
    """
    Provide ONE of:
      - ckpt_path=".../bsr_vgg19_epoch_043.pth" (or best_epoch file)
      - checkpoints_dir=".../checkpoints_vgg19_YYYYMMDD_HHMMSS", epoch=43
        expects filename: bsr_vgg19_epoch_###.pth (edit pattern below if needed)
    """
    if ckpt_path is None:
        if checkpoints_dir is None or epoch is None:
            raise ValueError("Provide either ckpt_path OR (checkpoints_dir and epoch).")
        # ---- adjust this pattern to match what your training script wrote ----
        ckpt_path = os.path.join(checkpoints_dir, f"bsr_vgg19_epoch_{int(epoch):03d}.pth")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = pick_device(device_preference)
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    # --- build model exactly like training (pretrained=False)
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

    # --- forward timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    logits = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    prob_small = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)  # [input_size,input_size]

    # Resize prob back to original
    prob_full = cv2.resize(prob_small, (w0, h0), interpolation=cv2.INTER_LINEAR)
    pred_bin = (prob_full >= threshold).astype(np.uint8)

    # --- GT (optional)
    gt_bin = load_mask_binary(mask_path, (h0, w0)) if mask_path else None

    # --- overlays: GT green, Pred red
    gt_overlay = overlay_colored(rgb, gt_bin, alpha=overlay_alpha, color=(0, 200, 0)) if gt_bin is not None else None
    pred_overlay = overlay_colored(rgb, pred_bin, alpha=overlay_alpha, color=(255, 0, 0))

    # --- stats
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
    tolerance_metrics = None

    if gt_bin is not None:
        metrics = compute_metrics(pred_bin, gt_bin)

        print(
            f"\n🧪 Strict metrics | "
            f"IoU: {metrics['iou']:.3f} | "
            f"F1: {metrics['f1']:.3f} | "
            f"Precision: {metrics['prec']:.3f} | "
            f"Recall: {metrics['rec']:.3f} | "
            f"Accuracy: {metrics['acc']:.3f}"
        )

        tolerance_metrics = {}

        for tol in distance_tolerances:
            tm = compute_tolerance_metrics(
                pred_bin,
                gt_bin,
                tolerance_px=tol
            )
            tolerance_metrics[tol] = tm

            print(
                f"📏 {tol}px tolerance | "
                f"Precision@{tol}px: {tm['prec_tol']:.3f} | "
                f"Recall@{tol}px: {tm['rec_tol']:.3f} | "
                f"F1@{tol}px: {tm['f1_tol']:.3f}"
            )

    else:
        print("ℹ️ No ground-truth mask provided; metrics skipped.")

    # --- save artifacts (optional)
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

    # --- Plot: Original | GT | Pred | Prob
    if show_plots:
        if use_2x2_layout:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
            axes = axes.ravel()
        else:
            fig, axes = plt.subplots(1, 4, figsize=(18, 6))

        for ax in axes:
            ax.set_aspect("equal")

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

        plt.savefig(
        "/content/BSR-detection-using-Computer-Vision/vgg19_prediction.png",
        dpi=150,
        bbox_inches="tight"
        )

        plt.show()

    # --- Optional OpenCV windows
    if show_with_cv2:
        prob_u8 = np.clip(prob_full * 255.0, 0, 255).astype(np.uint8)
        cv2.imshow("Original (BGR)", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imshow("Probability (0-255)", prob_u8)
        cv2.imshow("Binary Mask", (pred_bin * 255).astype(np.uint8))
        cv2.imshow("Pred Overlay", cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR))
        if gt_overlay is not None:
            cv2.imshow("GT Overlay", cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR))
        print("Press any key in an image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "rgb": rgb,                       # HxWx3 uint8
        "prob": prob_full,                # HxW float32 [0..1]
        "pred_bin": pred_bin,             # HxW uint8 {0,1}
        "overlay_pred_rgb": pred_overlay, # HxWx3 uint8
        "overlay_gt_rgb": gt_overlay,     # HxWx3 uint8 or None
        "summary": {
            "threshold": threshold,
            "area_percent": area_pct,
            "mean_prob_inside_pred_mask": mean_prob_in_mask,
            "max_prob_inside_pred_mask": max_prob_in_mask,
            "global_mean_prob": global_mean_prob,
            "metrics": metrics,
            "tolerance_metrics": tolerance_metrics,
            "inference_seconds": dt,
            "ckpt_path": ckpt_path,
        }
    }


# ===========================
#  MAIN (example)
# ===========================
if __name__ == "__main__":
    IMAGE_PATH = r"/content/BSR-detection-using-Computer-Vision/GOM_MC_BSR111.png"
    GT_MASK    = r"/content/BSR-detection-using-Computer-Vision/GOM_MC_BSR111.png"

    # Option A: point directly to the checkpoint you want to test
    CKPT_PATH  = r"/content/BSR-detection-using-Computer-Vision/models/bsr_vgg19_best_epoch_042.pth"

    # Option B: if you saved per-epoch checkpoints into a folder
    # CKPT_DIR = r"E:\...\checkpoints_vgg19_20260216_093343"
    # EPOCH = 43

    res = run_vgg19_unet_inference(
        image_path=IMAGE_PATH,
        mask_path=GT_MASK,
        ckpt_path=CKPT_PATH,          # or use checkpoints_dir + epoch
        # checkpoints_dir=CKPT_DIR,
        # epoch=EPOCH,

        input_size=512,
        threshold=0.5,
        distance_tolerances=(2, 3, 4, 5),
        overlay_alpha=0.35,
        device_preference=None,       # auto CUDA if available

        save_dir=None,                # set a folder path to save prob/mask/overlays
        save_basename=None,
        save_overlay=True,
        save_probs=True,

        show_plots=True,
        use_2x2_layout=True,
        show_with_cv2=False
    )

    print("Done. Shapes — rgb:", res["rgb"].shape,
          "| prob:", res["prob"].shape,
          "| mask:", res["pred_bin"].shape,
          "| overlay_pred:", None if res["overlay_pred_rgb"] is None else res["overlay_pred_rgb"].shape,
          "| overlay_gt:", None if res["overlay_gt_rgb"] is None else res["overlay_gt_rgb"].shape)
    print("Summary:", res["summary"])
