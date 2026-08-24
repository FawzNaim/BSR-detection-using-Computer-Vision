# single_image_inference_resnet_aligned.py
# Python 3.8/3.9 compatible
#
# Aligns with your ResNet training code:
# - torchvision DeepLabV3-ResNet50 or FCN-ResNet50
# - 1-channel logits (binary seg), sigmoid at inference
# - preprocessing: BGR->RGB, resize to 512, /255, CHW float32
# - masks: resize NEAREST, binarize >0
# - supports:
#     (A) ckpt_path directly, OR
#     (B) checkpoints_dir + epoch -> bsr_resnet_epoch_###.pth
#
# Notes:
# - If your training DID NOT use aux_loss=True, set aux_loss=False below.
# - If checkpoints were saved from DataParallel, we auto-strip "module." prefixes.

import os
import time
from typing import Optional, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50


# ------------------ Build ResNet segmentation (binary) ------------------
def _rewire_for_binary_output(model, model_name: str):
    """
    Modify classifier (and aux_classifier if present) to output 1 channel logits.
    """
    if "deeplabv3" in model_name:
        if hasattr(model, "classifier") and model.classifier is not None:
            last = model.classifier[-1]
            if isinstance(last, nn.Conv2d):
                model.classifier[-1] = nn.Conv2d(last.in_channels, 1, kernel_size=1)

        if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
            aux_last = model.aux_classifier[-1]
            if isinstance(aux_last, nn.Conv2d):
                model.aux_classifier[-1] = nn.Conv2d(aux_last.in_channels, 1, kernel_size=1)

    elif "fcn" in model_name:
        if hasattr(model, "classifier") and model.classifier is not None:
            last_conv_idx = None
            for i in reversed(range(len(model.classifier))):
                if isinstance(model.classifier[i], nn.Conv2d):
                    last_conv_idx = i
                    break
            if last_conv_idx is not None:
                in_ch = model.classifier[last_conv_idx].in_channels
                model.classifier[last_conv_idx] = nn.Conv2d(in_ch, 1, kernel_size=1)

        if hasattr(model, "aux_classifier") and model.aux_classifier is not None:
            last_conv_idx = None
            for i in reversed(range(len(model.aux_classifier))):
                if isinstance(model.aux_classifier[i], nn.Conv2d):
                    last_conv_idx = i
                    break
            if last_conv_idx is not None:
                in_ch = model.aux_classifier[last_conv_idx].in_channels
                model.aux_classifier[last_conv_idx] = nn.Conv2d(in_ch, 1, kernel_size=1)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model


def build_resnet_segmentation(
    model_name: str = "deeplabv3_resnet50",
    pretrained: bool = False,
    aux_loss: bool = True
):
    """
    model_name: 'deeplabv3_resnet50' or 'fcn_resnet50'
    aux_loss: MUST match how the checkpoint was trained.
              If checkpoint contains aux head weights, aux_loss must be True.
    """
    if model_name == "deeplabv3_resnet50":
        try:
            weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            model = deeplabv3_resnet50(weights=weights, aux_loss=aux_loss)
        except TypeError:
            model = deeplabv3_resnet50(pretrained=pretrained, aux_loss=aux_loss)

    elif model_name == "fcn_resnet50":
        try:
            weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT if pretrained else None
            model = fcn_resnet50(weights=weights, aux_loss=aux_loss)
        except TypeError:
            model = fcn_resnet50(pretrained=pretrained, aux_loss=aux_loss)

    else:
        raise ValueError("model_name must be 'deeplabv3_resnet50' or 'fcn_resnet50'.")

    model = _rewire_for_binary_output(model, model_name)
    return model


def _ensure_logits(output):
    """Handle torchvision segmentation outputs (dict with 'out') or plain tensors."""
    if isinstance(output, dict):
        return output.get("out", None)
    return output


# ------------------ Pre/Post helpers (match training preprocessing) ------------------
def pick_device(device_preference: Optional[str] = None):
    if device_preference:
        if device_preference == "cpu":
            return torch.device("cpu")
        if device_preference.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device_preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image_rgb_tensor(image_path: str, size: int = 512):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = rgb.shape[:2]

    rgb_resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = rgb_resized.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW float32
    x = torch.from_numpy(x).unsqueeze(0)  # NCHW

    return rgb, (h0, w0), x


def load_mask_binary(mask_path: Optional[str], out_hw: tuple):
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
    """
    Overlay a single-color mask on RGB image. color=(R,G,B).
    """
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

    return dict(acc=acc, prec=prec, rec=rec, f1=f1, iou=iou)


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

    This affects evaluation only. It does NOT modify model predictions
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

    dist_to_gt = cv2.distanceTransform(
        (~gt).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE
    )

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


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    If checkpoint was saved from DataParallel, keys start with "module." -> strip it.
    """
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


# ------------------ Main single-image inference ------------------
@torch.inference_mode()
def run_resnet_inference(
    image_path: str,
    mask_path: Optional[str] = None,
    *,
    ckpt_path: Optional[str] = None,
    checkpoints_dir: Optional[str] = None,
    epoch: Optional[int] = None,
    model_name: str = "deeplabv3_resnet50",   # or "fcn_resnet50"
    aux_loss: bool = True,                    # MUST match training
    image_size: int = 512,
    threshold: float = 0.5,
    distance_tolerances=(2, 3, 4, 5),
    overlay_alpha: float = 0.35,
    device_preference: Optional[str] = None,
    save_dir: Optional[str] = None,
    save_basename: Optional[str] = None,
    save_overlay: bool = True,
    save_probs: bool = True,
    show_plots: bool = True,
    show_with_cv2: bool = False
) -> Dict[str, Any]:
    """
    Provide ONE of:
      - ckpt_path=".../bsr_resnet_epoch_081.pth" (or best_epoch file)
      - checkpoints_dir=".../checkpoints_resnet_YYYYMMDD_HHMMSS", epoch=81
        (expects filename: bsr_resnet_epoch_###.pth; adjust below if yours differs)
    """
    if ckpt_path is None:
        if checkpoints_dir is None or epoch is None:
            raise ValueError("Provide either ckpt_path OR (checkpoints_dir and epoch).")
        ckpt_path = os.path.join(checkpoints_dir, f"bsr_resnet_epoch_{int(epoch):03d}.pth")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = pick_device(device_preference)
    print(f"Device: {device}")
    print(f"Model: {model_name} | aux_loss={aux_loss}")
    print(f"Checkpoint: {ckpt_path}")

    # Build model (binary head) and load weights
    model = build_resnet_segmentation(model_name=model_name, pretrained=False, aux_loss=aux_loss).to(device)
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

    # Load + preprocess image
    rgb, (h0, w0), x = load_image_rgb_tensor(image_path, size=image_size)
    x = x.to(device, non_blocking=True)

    # Forward timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    out = model(x)
    logits = _ensure_logits(out)
    if logits is None:
        raise RuntimeError("Model did not return logits or an 'out' tensor.")
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    # Probability at model resolution (image_size x image_size), then resize back to original
    prob_small = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()  # [H,W] in 0..1

    prob_u8_small = (prob_small * 255).astype(np.uint8)
    prob_u8 = cv2.resize(prob_u8_small, (w0, h0), interpolation=cv2.INTER_LINEAR)  # back to original size
    prob_float = prob_u8.astype(np.float32) / 255.0

    pred_bin = (prob_u8 >= int(threshold * 255)).astype(np.uint8)

    # Load GT (optional)
    gt_bin = load_mask_binary(mask_path, (h0, w0)) if mask_path else None

    # Overlays (GT green, Pred red)
    gt_overlay = overlay_colored(rgb, gt_bin, alpha=overlay_alpha, color=(0, 200, 0)) if gt_bin is not None else None
    pred_overlay = overlay_colored(rgb, pred_bin, alpha=overlay_alpha, color=(255, 0, 0))

    # Stats
    area_pct = 100.0 * float(pred_bin.mean())
    mean_prob_in_mask = float(prob_float[pred_bin.astype(bool)].mean()) if pred_bin.any() else 0.0
    max_prob_in_mask = float(prob_float[pred_bin.astype(bool)].max()) if pred_bin.any() else 0.0
    global_mean_prob = float(prob_float.mean())

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

    # Save artifacts (optional)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = save_basename if save_basename else os.path.splitext(os.path.basename(image_path))[0]

        if save_probs:
            cv2.imwrite(os.path.join(save_dir, f"{base}_pred_prob.png"), prob_u8)
        cv2.imwrite(os.path.join(save_dir, f"{base}_pred_mask.png"), (pred_bin * 255).astype(np.uint8))

        if save_overlay:
            cv2.imwrite(os.path.join(save_dir, f"{base}_overlay_pred.png"),
                        cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR))
            if gt_overlay is not None:
                cv2.imwrite(os.path.join(save_dir, f"{base}_overlay_gt.png"),
                            cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR))

    # Figure: Original | GT overlay | Pred overlay | Probability
    if show_plots:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
        axes = axes.ravel()

        axes[0].imshow(rgb); axes[0].set_title("Original"); axes[0].axis("off")

        if gt_overlay is not None:
            axes[1].imshow(gt_overlay); axes[1].set_title("GT Overlay (green)"); axes[1].axis("off")
        else:
            axes[1].imshow(rgb); axes[1].set_title("GT Overlay (none)"); axes[1].axis("off")

        axes[2].imshow(pred_overlay); axes[2].set_title(f"Pred Overlay (red) t={threshold:.2f}"); axes[2].axis("off")

        im = axes[3].imshow(prob_float, cmap="jet", vmin=0, vmax=1)
        axes[3].set_title("Probability (0–1)"); axes[3].axis("off")
        cbar = fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
        cbar.set_label("BSR probability")

        plt.savefig(
        "/content/BSR-detection-using-Computer-Vision/resnet_prediction.png",
        dpi=150,
        bbox_inches="tight"
        )

        plt.show()

    # Optional OpenCV windows
    if show_with_cv2:
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
        "prob_u8": prob_u8,               # HxW uint8 [0..255]
        "prob_float": prob_float,         # HxW float32 [0..1]
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
            "model_name": model_name,
            "aux_loss": aux_loss,
        }
    }


# ------------------ Example usage ------------------
if __name__ == "__main__":
    # Option A: checkpoint path directly (best or epoch)
    MODEL_PATH = r"/content/BSR-detection-using-Computer-Vision/models/bsr_resnet_best_epoch_094.pth"
    IMAGE_PATH = r"/content/BSR-detection-using-Computer-Vision/GOM_MC_BSR122.png"
    GT_MASK    = r"/content/BSR-detection-using-Computer-Vision/GOM_MC_BSR122.png"
    OUT_DIR    = r"D:\BSR Prediction using CV\Training images\Not used for training\GOM_MC_BSR122"

    res = run_resnet_inference(
        image_path=IMAGE_PATH,
        mask_path=GT_MASK,
        ckpt_path=MODEL_PATH,                 # or use checkpoints_dir+epoch below
        # checkpoints_dir=r"...\checkpoints_resnet_YYYYMMDD_HHMMSS",
        # epoch=81,

        model_name="deeplabv3_resnet50",      # must match training
        aux_loss=True,                        # set False if you trained without aux head

        image_size=512,
        threshold=0.5,
        distance_tolerances=(2, 3, 4, 5),
        overlay_alpha=0.35,

        device_preference=None,               # auto CUDA if available
        save_dir=OUT_DIR,                     # set None to avoid saving
        save_basename=None,
        save_overlay=True,
        save_probs=True,
        show_plots=True,
        show_with_cv2=False
    )

    print("Done. Shapes — rgb:", res["rgb"].shape,
          "| prob:", res["prob_u8"].shape,
          "| mask:", res["pred_bin"].shape,
          "| overlay_pred:", None if res["overlay_pred_rgb"] is None else res["overlay_pred_rgb"].shape,
          "| overlay_gt:", None if res["overlay_gt_rgb"] is None else res["overlay_gt_rgb"].shape)
    print("Summary:", res["summary"])
