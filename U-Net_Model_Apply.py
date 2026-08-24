import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================================================
#  U-Net definition (MUST match your 80/20 training script)
#  - Same DoubleConv blocks (Conv->BN->ReLU->Conv->BN->ReLU->Dropout/Identity)
#  - Same encoder/decoder topology (no extra bottleneck pooling)
#  - Same channel sizes
#  - Output is logits (no sigmoid)
# =========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, p_drop=0.0, debug_shapes=False):
        super().__init__()
        self.debug_shapes = debug_shapes

        self.enc1 = DoubleConv(3, 64, p_drop)
        self.enc2 = DoubleConv(64, 128, p_drop)
        self.enc3 = DoubleConv(128, 256, p_drop)
        self.enc4 = DoubleConv(256, 512, p_drop)

        # Bottleneck WITHOUT extra pooling (keeps same 1/8 resolution as enc4 output)
        self.bottleneck = DoubleConv(512, 512, p_drop)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256, p_drop)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128, p_drop)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64, p_drop)

        self.out = nn.Conv2d(64, 1, 1)  # logits

    def _assert_same(self, a, b, name_a, name_b):
        if self.debug_shapes:
            assert a.shape[2:] == b.shape[2:], f"{name_a} {a.shape} vs {name_b} {b.shape}"

    def forward(self, x):
        e1 = self.enc1(x)                 # 1×
        e2 = self.enc2(self.pool(e1))     # 1/2
        e3 = self.enc3(self.pool(e2))     # 1/4
        e4 = self.enc4(self.pool(e3))     # 1/8

        b  = self.bottleneck(e4)          # 1/8

        u3 = self.up3(b)                  # 1/4
        self._assert_same(u3, e3, "up3", "e3")
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)                 # 1/2
        self._assert_same(u2, e2, "up2", "e2")
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)                 # 1×
        self._assert_same(u1, e1, "up1", "e1")
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out(d1)               # logits


# =========================================================
#  I/O + metrics utilities (match training preprocessing)
#  - Images: BGR->RGB, resize to (size,size), normalize /255, CHW float32
#  - Masks: grayscale, resize NEAREST, binarize >0
# =========================================================
def pick_device(device_preference="cuda"):
    if device_preference == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_from_ckpt(ckpt_path, device, p_drop=0.0, debug_shapes=False):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = UNet(p_drop=p_drop, debug_shapes=debug_shapes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def load_image_for_unet(image_path, size=512):
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)

    x = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0  # CHW, float32
    x = torch.from_numpy(x).unsqueeze(0)                   # NCHW

    return rgb, x

def load_mask_binary(mask_path, size=512):
    if mask_path is None:
        return None
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)

def overlay_colored(img_rgb, mask_bin, color=(0, 200, 0), alpha=0.35):
    """
    Overlay a single-color mask on RGB image. color=(R,G,B).
    """
    if mask_bin is None:
        return img_rgb.copy()

    color_img = np.zeros_like(img_rgb, dtype=np.uint8)
    color_img[..., 0] = color[0]
    color_img[..., 1] = color[1]
    color_img[..., 2] = color[2]

    out = img_rgb.astype(np.float32).copy()
    out = np.where(mask_bin[..., None] > 0,
                   alpha * color_img + (1 - alpha) * out,
                   out)
    return out.astype(np.uint8)

def compute_metrics(pred_bin, gt_bin):
    """
    Strict pixel-wise metrics.
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


def compute_tolerance_metrics(pred_bin, gt_bin, tolerance_px=2):
    """
    Distance-tolerant Precision, Recall, and F1.

    A predicted BSR pixel is counted as correct if it lies within
    `tolerance_px` Euclidean pixels of any ground-truth BSR pixel.

    Likewise, a ground-truth BSR pixel is counted as recovered if it lies
    within `tolerance_px` pixels of any predicted BSR pixel.

    This changes evaluation only and does not require retraining.
    """
    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)

    n_pred = int(pred.sum())
    n_gt = int(gt.sum())

    if n_gt == 0 and n_pred == 0:
        return dict(tolerance_px=tolerance_px, prec_tol=1.0, rec_tol=1.0, f1_tol=1.0)

    if n_gt == 0 or n_pred == 0:
        return dict(tolerance_px=tolerance_px, prec_tol=0.0, rec_tol=0.0, f1_tol=0.0)

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


# =========================================================
#  Inference runner (supports either:
#   - direct ckpt_path, OR
#   - checkpoints_dir + epoch number)
# =========================================================
@torch.inference_mode()
def run_unet_inference(
    image_path,
    mask_path=None,
    *,
    ckpt_path=None,
    checkpoints_dir=None,
    epoch=None,
    size=512,
    thresh=0.5,
    distance_tolerances=(2, 3, 4, 5),
    device_preference="cuda",
    p_drop=0.0,
    debug_shapes=False,
    show_probability=True
):
    """
    Aligns with your 80/20 U-Net training preprocessing and architecture.

    Provide ONE of:
      - ckpt_path=".../bsr_unet_epoch_100.pth"
      - checkpoints_dir=".../checkpoints_unet_YYYYMMDD_HHMMSS", epoch=100
    """
    if ckpt_path is None:
        if checkpoints_dir is None or epoch is None:
            raise ValueError("Provide either ckpt_path OR (checkpoints_dir and epoch).")
        ckpt_path = os.path.join(checkpoints_dir, f"bsr_unet_epoch_{int(epoch):03d}.pth")

    device = pick_device(device_preference)
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    # Load model + inputs
    model = load_model_from_ckpt(ckpt_path, device, p_drop=p_drop, debug_shapes=debug_shapes)
    rgb, x = load_image_for_unet(image_path, size=size)
    x = x.to(device, non_blocking=True)

    # Forward timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    logits = model(x)  # N,1,H,W
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()   # HxW in [0,1]
    pred_bin = (prob >= thresh).astype(np.uint8)

    # Load GT (optional)
    gt_bin = None
    if mask_path is not None:
        gt_bin = load_mask_binary(mask_path, size=size)

    # Overlays
    # Keep colors explicit and consistent in titles:
    gt_overlay   = overlay_colored(rgb, gt_bin,   color=(0, 200, 0), alpha=0.35) if gt_bin is not None else rgb.copy()
    pred_overlay = overlay_colored(rgb, pred_bin, color=(255, 0, 0), alpha=0.35)  # RED prediction

    # Stats
    pos_area = 100.0 * float(pred_bin.mean())
    mean_prob = float(prob.mean())
    max_prob = float(prob.max())
    print(f"\n⏱ Inference: {dt:.3f}s | Pred area: {pos_area:.2f}% | Mean prob: {mean_prob:.3f} | Max prob: {max_prob:.3f}")

    if gt_bin is not None:
        m = compute_metrics(pred_bin, gt_bin)

        print(
            f"\n🧪 Strict metrics | "
            f"IoU: {m['iou']:.3f} | "
            f"F1: {m['f1']:.3f} | "
            f"Precision: {m['prec']:.3f} | "
            f"Recall: {m['rec']:.3f} | "
            f"Accuracy: {m['acc']:.3f}"
        )

        tolerance_results = {}
        for tol in distance_tolerances:
            tm = compute_tolerance_metrics(pred_bin, gt_bin, tolerance_px=tol)
            tolerance_results[tol] = tm

            print(
                f"📏 {tol}px tolerance | "
                f"Precision@{tol}px: {tm['prec_tol']:.3f} | "
                f"Recall@{tol}px: {tm['rec_tol']:.3f} | "
                f"F1@{tol}px: {tm['f1_tol']:.3f}"
            )
    else:
        m = None
        tolerance_results = None
        print("ℹ️ No ground-truth mask provided; metrics skipped.")

    # ---- Figure ----
    # Original | GT overlay | Pred overlay | Probability (optional)
    ncols = 4 if show_probability else 3
    fig, axs = plt.subplots(1, ncols, figsize=(18 if show_probability else 14, 4))

    axs[0].imshow(rgb); axs[0].set_title("Original"); axs[0].axis("off")

    if gt_bin is not None:
        axs[1].imshow(gt_overlay); axs[1].set_title("GT Overlay (green)"); axs[1].axis("off")
    else:
        axs[1].imshow(rgb); axs[1].set_title("GT Overlay (none)"); axs[1].axis("off")

    axs[2].imshow(pred_overlay); axs[2].set_title(f"Pred Overlay (red)  t={thresh}"); axs[2].axis("off")

    if show_probability:
        im = axs[3].imshow(prob, cmap="jet", vmin=0, vmax=1)
        axs[3].set_title("Probability (0–1)"); axs[3].axis("off")
        cbar = plt.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
        cbar.set_label("BSR probability")

    plt.tight_layout()

    plt.savefig(
    "/content/BSR-detection-using-Computer-Vision/unet_prediction.png",
    dpi=150,
    bbox_inches="tight"
    ) 

    plt.show()

    # Return useful outputs programmatically (optional)
    return {
        "prob": prob,
        "pred_bin": pred_bin,
        "gt_bin": gt_bin,
        "metrics": m,
        "tolerance_metrics": tolerance_results,
        "inference_seconds": dt,
        "ckpt_path": ckpt_path,
    }


# =========================================================
# Example usage
# =========================================================
if __name__ == "__main__":
    # Option A: give the checkpoint path directly
    run_unet_inference(
        image_path=r"/content/BSR-detection-using-Computer-Vision/GOM_MC_BSR122.png",
        mask_path=r"/content/BSR-detection-using-Computer-Vision/GOM_MC_BSR122_label.png",
        ckpt_path=r"/content/BSR-detection-using-Computer-Vision/models/bsr_unet_best_epoch_074.pth",
        size=512,
        thresh=0.5,
        distance_tolerances=(2, 3, 4, 5),
        device_preference="cuda",
        p_drop=0.0,          # MUST match training (you used 0.0)
        debug_shapes=False,
        show_probability=True
    )

    # Option B: point to the checkpoints folder and pick an epoch number
    # run_unet_inference(
    #     image_path=r"...\some_image.png",
    #     mask_path=r"...\some_image_label.png",
    #     checkpoints_dir=r"...\checkpoints_unet_20260216_093343",
    #     epoch=75,
    #     size=512,
    #     thresh=0.5,
    #     device_preference="cuda",
    # )
