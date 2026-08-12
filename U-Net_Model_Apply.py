import os
import cv2
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score


# =========================================================
#  U-Net definition (MUST match training)
#  - Same DoubleConv blocks
#  - Same topology
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

        b = self.bottleneck(e4)           # 1/8

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
#  Helpers (device, checkpoint loading, preprocessing)
# =========================================================
def pick_device(device_preference="cuda"):
    if device_preference == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _strip_module_prefix(state_dict):
    """If checkpoint was saved with DataParallel, keys start with 'module.'."""
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if isinstance(first_key, str) and first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(ckpt_obj):
    """
    Support checkpoints saved as:
      - plain state_dict
      - dict with 'state_dict'
      - dict with 'model' (sometimes)
    """
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
    return ckpt_obj


def load_model_from_ckpt(ckpt_path, device, p_drop=0.0, debug_shapes=False, strict=True):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = UNet(p_drop=p_drop, debug_shapes=debug_shapes).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = _extract_state_dict(ckpt)
    state = _strip_module_prefix(state)

    missing, unexpected = model.load_state_dict(state, strict=strict)
    if (missing or unexpected) and strict is False:
        if missing:
            print(f"[load_state_dict] missing keys: {missing}")
        if unexpected:
            print(f"[load_state_dict] unexpected keys: {unexpected}")

    model.eval()
    return model


def resolve_ckpt_path(ckpt_path=None, checkpoints_dir=None, epoch=None, prefer_best=True):
    """
    Supports:
      - ckpt_path directly
      - checkpoints_dir + epoch -> bsr_unet_epoch_###.pth
      - checkpoints_dir + epoch -> bsr_unet_best_epoch_###.pth (if prefer_best=True and exists)
    """
    if ckpt_path is not None:
        return ckpt_path

    if checkpoints_dir is None or epoch is None:
        raise ValueError("Provide either ckpt_path OR (checkpoints_dir and epoch).")

    epoch = int(epoch)
    cand_best = os.path.join(checkpoints_dir, f"bsr_unet_best_epoch_{epoch:03d}.pth")
    cand_epoch = os.path.join(checkpoints_dir, f"bsr_unet_epoch_{epoch:03d}.pth")

    if prefer_best and os.path.isfile(cand_best):
        return cand_best
    return cand_epoch


def load_image_for_unet(image_path, size=512):
    """
    TRAINING-aligned preprocessing:
      - read BGR -> RGB
      - resize to (size,size)
      - CHW float32 /255
    Returns: rgb_resized_uint8, x_tensor[N,3,H,W]
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)

    x = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(0)
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
    """Overlay a single-color mask on RGB image. color=(R,G,B)."""
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
    p = pred_bin.flatten()
    g = gt_bin.flatten()

    if g.sum() == 0:
        acc = accuracy_score(g, p)
        if p.sum() == 0:
            return dict(acc=acc, prec=0.0, rec=0.0, iou=1.0)
        return dict(acc=acc, prec=0.0, rec=0.0, iou=0.0)

    acc = accuracy_score(g, p)
    prec = precision_score(g, p, zero_division=0)
    rec = recall_score(g, p, zero_division=0)
    inter = np.logical_and(g == 1, p == 1).sum()
    union = np.logical_or(g == 1, p == 1).sum()
    iou = inter / union if union > 0 else 0.0
    return dict(acc=acc, prec=prec, rec=rec, iou=iou)


# =========================================================
#  Inference runner
# =========================================================
@torch.inference_mode()
def run_unet_inference(
    image_path,
    mask_path=None,
    *,
    ckpt_path=None,
    checkpoints_dir=None,
    epoch=None,
    prefer_best=True,
    size=512,
    thresh=0.5,
    device_preference="cuda",
    p_drop=0.0,
    debug_shapes=False,
    show_probability=True,
    strict_load=True
):
    """
    Inference is identical regardless of whether training used 80/20 split or ALL images.

    Provide ONE of:
      - ckpt_path=".../bsr_unet_epoch_100.pth" or ".../bsr_unet_best_epoch_100.pth"
      - checkpoints_dir=".../checkpoints_unet_YYYYMMDD_HHMMSS", epoch=100
        (will prefer best if prefer_best=True and file exists)
    """
    ckpt_path = resolve_ckpt_path(
        ckpt_path=ckpt_path,
        checkpoints_dir=checkpoints_dir,
        epoch=epoch,
        prefer_best=prefer_best
    )

    device = pick_device(device_preference)
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    model = load_model_from_ckpt(
        ckpt_path,
        device,
        p_drop=p_drop,
        debug_shapes=debug_shapes,
        strict=strict_load
    )

    rgb, x = load_image_for_unet(image_path, size=size)
    x = x.to(device, non_blocking=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    logits = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
    pred_bin = (prob >= thresh).astype(np.uint8)

    gt_bin = load_mask_binary(mask_path, size=size) if mask_path else None

    gt_overlay = overlay_colored(rgb, gt_bin, color=(0, 200, 0), alpha=0.35) if gt_bin is not None else rgb.copy()
    pred_overlay = overlay_colored(rgb, pred_bin, color=(255, 0, 0), alpha=0.35)

    pos_area = 100.0 * float(pred_bin.mean())
    mean_prob = float(prob.mean())
    max_prob = float(prob.max())
    print(f"\n⏱ Inference: {dt:.3f}s | Pred area: {pos_area:.2f}% | Mean prob: {mean_prob:.3f} | Max prob: {max_prob:.3f}")

    if gt_bin is not None:
        m = compute_metrics(pred_bin, gt_bin)
        print(f"🧪 Metrics  IoU: {m['iou']:.3f} | Prec: {m['prec']:.3f} | Rec: {m['rec']:.3f} | Acc: {m['acc']:.3f}")
    else:
        m = None
        print("ℹ️ No ground-truth mask provided; metrics skipped.")

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

    return {
        "prob": prob,
        "pred_bin": pred_bin,
        "gt_bin": gt_bin,
        "metrics": m,
        "inference_seconds": dt,
        "ckpt_path": ckpt_path,
    }


# =========================================================
# Example usage
# =========================================================
if __name__ == "__main__":
    run_unet_inference(
        image_path=r"/content/BSR-detection-using-Computer-Vision/Non_BSR_1_GOM.png",
        mask_path=r"",

        # Option A: direct checkpoint
        ckpt_path=r"/content/BSR-detection-using-Computer-Vision/models/bsr_unet_best_epoch_099.pth",

        # Option B: checkpoints_dir + epoch (auto prefers best if exists)
        # checkpoints_dir=r"E:\...\checkpoints_unet_all_YYYYMMDD_HHMMSS",
        # epoch=100,

        prefer_best=True,
        size=512,
        thresh=0.5,
        device_preference="cuda",
        p_drop=0.0,
        debug_shapes=False,
        show_probability=True,
        strict_load=True
    )
