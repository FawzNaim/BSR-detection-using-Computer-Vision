# ============================================================
# single_image_inference_resnet_imagenet.py
#
# DeepLabV3-ResNet50 / FCN-ResNet50 inference for binary
# BSR segmentation.
#
# Designed for a model trained using:
#
#   weights=None
#   weights_backbone=ResNet50_Weights.DEFAULT
#
# IMPORTANT:
# During inference we DO NOT reload ImageNet weights.
# We build the architecture and load the trained .pth file.
# ============================================================

import os
import time
from typing import Optional, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import torchvision

from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    fcn_resnet50
)

# ============================================================
# MODEL
# ============================================================

def _rewire_for_binary_output(model, model_name: str):
    """
    Replace the final segmentation classifier with a
    1-channel output layer for binary BSR segmentation.

    Output:
        1 channel logits

    These logits are passed through sigmoid during inference.
    """

    # --------------------------------------------------------
    # DeepLabV3
    # --------------------------------------------------------
    if "deeplabv3" in model_name:

        if hasattr(model, "classifier") and model.classifier is not None:

            last = model.classifier[-1]

            if isinstance(last, nn.Conv2d):

                model.classifier[-1] = nn.Conv2d(
                    last.in_channels,
                    1,
                    kernel_size=1
                )

        # Auxiliary classifier
        if (
            hasattr(model, "aux_classifier")
            and model.aux_classifier is not None
        ):

            aux_last = model.aux_classifier[-1]

            if isinstance(aux_last, nn.Conv2d):

                model.aux_classifier[-1] = nn.Conv2d(
                    aux_last.in_channels,
                    1,
                    kernel_size=1
                )

    # --------------------------------------------------------
    # FCN
    # --------------------------------------------------------
    elif "fcn" in model_name:

        if hasattr(model, "classifier") and model.classifier is not None:

            last_conv_idx = None

            for i in reversed(range(len(model.classifier))):

                if isinstance(model.classifier[i], nn.Conv2d):

                    last_conv_idx = i
                    break

            if last_conv_idx is not None:

                in_ch = model.classifier[last_conv_idx].in_channels

                model.classifier[last_conv_idx] = nn.Conv2d(
                    in_ch,
                    1,
                    kernel_size=1
                )

        # Auxiliary classifier
        if (
            hasattr(model, "aux_classifier")
            and model.aux_classifier is not None
        ):

            last_conv_idx = None

            for i in reversed(range(len(model.aux_classifier))):

                if isinstance(
                    model.aux_classifier[i],
                    nn.Conv2d
                ):

                    last_conv_idx = i
                    break

            if last_conv_idx is not None:

                in_ch = (
                    model.aux_classifier[last_conv_idx].in_channels
                )

                model.aux_classifier[last_conv_idx] = nn.Conv2d(
                    in_ch,
                    1,
                    kernel_size=1
                )

    else:

        raise ValueError(
            f"Unsupported model_name: {model_name}"
        )

    return model

# ============================================================
# BUILD MODEL
# ============================================================

def build_resnet_segmentation(
    model_name: str = "deeplabv3_resnet50",
    pretrained: bool = False,
    aux_loss: bool = False
):
    """
    Build a DeepLabV3-ResNet50 or FCN-ResNet50 model.

    pretrained=True:
        Use ImageNet-pretrained ResNet50 backbone.

        Intended primarily when constructing the model
        for TRAINING.

    pretrained=False:
        Do not initialize from ImageNet.

        This should normally be used during INFERENCE,
        because the trained checkpoint already contains
        the fine-tuned backbone weights.

    aux_loss MUST match the model configuration used
    during training.
    """

    backbone_weights = (
        ResNet50_Weights.DEFAULT
        if pretrained
        else None
    )

    # --------------------------------------------------------
    # DeepLabV3
    # --------------------------------------------------------

    if model_name == "deeplabv3_resnet50":

        model = deeplabv3_resnet50(

            # Do NOT load pretrained COCO segmentation weights
            weights=None,

            # ImageNet-pretrained ResNet50 backbone
            weights_backbone=backbone_weights,

            aux_loss=aux_loss
        )

    # --------------------------------------------------------
    # FCN
    # --------------------------------------------------------

    elif model_name == "fcn_resnet50":

        model = fcn_resnet50(
            weights=None,
            weights_backbone=backbone_weights,
            aux_loss=aux_loss
        )

    else:

        raise ValueError(
            "model_name must be "
            "'deeplabv3_resnet50' "
            "or 'fcn_resnet50'."
        )

    # Convert segmentation output to binary
    model = _rewire_for_binary_output(
        model,
        model_name
    )

    return model

# ============================================================
# OUTPUT HELPER
# ============================================================

def _ensure_logits(output):
    """
    Torchvision segmentation models usually return:

        {
            "out": tensor,
            "aux": tensor
        }

    We only use the main 'out' prediction.
    """

    if isinstance(output, dict):

        return output.get(
            "out",
            None
        )

    return output

# ============================================================
# DEVICE
# ============================================================

def pick_device(
    device_preference: Optional[str] = None
):

    if device_preference:

        if device_preference == "cpu":

            return torch.device("cpu")

        if (
            device_preference.startswith("cuda")
            and torch.cuda.is_available()
        ):

            return torch.device(
                device_preference
            )

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

# ============================================================
# IMAGE LOADING / PREPROCESSING
# ============================================================

def load_image_rgb_tensor(
    image_path: str,
    size: int = 512,

    # IMPORTANT:
    # Set this to True ONLY if your training code
    # also used ImageNet mean/std normalization.
    imagenet_normalization: bool = False
):
    """
    Load image and convert:

        BGR -> RGB
        resize -> size x size
        uint8 -> float32
        0-255 -> 0-1
        HWC -> CHW
        add batch dimension

    ImageNet normalization is OPTIONAL and must match
    whatever preprocessing was used during training.
    """

    bgr = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )

    if bgr is None:

        raise FileNotFoundError(
            f"Failed to read image: {image_path}"
        )
    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB
    )
    h0, w0 = rgb.shape[:2]

    # Resize
    rgb_resized = cv2.resize(
        rgb,
        (size, size),
        interpolation=cv2.INTER_LINEAR
    )

    # Convert to float32 and 0-1
    x = (
        rgb_resized.astype(np.float32)
        / 255.0
    )

    # --------------------------------------------------------
    # OPTIONAL ImageNet normalization
    # --------------------------------------------------------

    if imagenet_normalization:
        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )
        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )
        x = (
            x - mean
        ) / std
    # HWC -> CHW
    x = x.transpose(
        2,
        0,
        1
    )
    # numpy -> torch
    x = torch.from_numpy(x)

    # CHW -> NCHW
    x = x.unsqueeze(0)

    return (
        rgb,
        (h0, w0),
        x
    )

# ============================================================
# MASK LOADING
# ============================================================

def load_mask_binary(
    mask_path: str,
    out_hw: tuple
):

    m = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if m is None:

        raise FileNotFoundError(
            f"Failed to read mask: {mask_path}"
        )

    h, w = out_hw

    m = cv2.resize(
        m,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    return (
        m > 0
    ).astype(np.uint8)

# ============================================================
# OVERLAY
# ============================================================

def overlay_colored(
    rgb_uint8: np.ndarray,
    mask_bin_uint8: np.ndarray,
    alpha: float = 0.35,
    color=(255, 0, 0)
) -> np.ndarray:
    """
    Overlay a binary mask onto RGB image.

    color=(R,G,B)
    """

    if mask_bin_uint8 is None:

        return rgb_uint8.copy()

    mask = mask_bin_uint8.astype(bool)

    out = (
        rgb_uint8
        .astype(np.float32)
        .copy()
    )

    color_img = np.zeros_like(
        rgb_uint8,
        dtype=np.float32
    )

    color_img[..., 0] = color[0]
    color_img[..., 1] = color[1]
    color_img[..., 2] = color[2]

    out[mask] = (
        alpha * color_img[mask]
        + (1 - alpha) * out[mask]
    )

    return out.astype(np.uint8)

# ============================================================
# STRICT METRICS
# ============================================================

def compute_metrics(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray
):
    """
    Strict pixel-wise segmentation metrics.
    """

    p = pred_bin.flatten()
    g = gt_bin.flatten()

    acc = accuracy_score(
        g,
        p
    )

    prec = precision_score(
        g,
        p,
        zero_division=0
    )

    rec = recall_score(
        g,
        p,
        zero_division=0
    )

    f1 = f1_score(
        g,
        p,
        zero_division=0
    )

    intersection = np.logical_and(
        g == 1,
        p == 1
    ).sum()

    union = np.logical_or(
        g == 1,
        p == 1
    ).sum()

    iou = (
        intersection / union
        if union > 0
        else 1.0
    )

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "iou": iou
    }

# ============================================================
# DISTANCE-TOLERANT METRICS
# ============================================================

def compute_tolerance_metrics(
    pred_bin: np.ndarray,
    gt_bin: np.ndarray,
    tolerance_px: int = 2
):
    """
    Distance-tolerant Precision, Recall and F1.

    Useful for thin BSR reflections where a predicted
    BSR may be only a few pixels offset from the manually
    labeled BSR.

    This affects evaluation ONLY.
    """

    pred = pred_bin.astype(bool)
    gt = gt_bin.astype(bool)

    n_pred = int(pred.sum())
    n_gt = int(gt.sum())

    # Both empty
    if n_gt == 0 and n_pred == 0:

        return {
            "tolerance_px": tolerance_px,
            "prec_tol": 1.0,
            "rec_tol": 1.0,
            "f1_tol": 1.0
        }

    # Only one is empty
    if n_gt == 0 or n_pred == 0:

        return {
            "tolerance_px": tolerance_px,
            "prec_tol": 0.0,
            "rec_tol": 0.0,
            "f1_tol": 0.0
        }

    # Distance from every pixel to nearest GT pixel
    dist_to_gt = cv2.distanceTransform(

        (~gt).astype(np.uint8),

        cv2.DIST_L2,

        cv2.DIST_MASK_PRECISE
    )

    # Distance from every pixel to nearest predicted pixel
    dist_to_pred = cv2.distanceTransform(

        (~pred).astype(np.uint8),

        cv2.DIST_L2,

        cv2.DIST_MASK_PRECISE
    )

    matched_pred = (
        pred
        & (dist_to_gt <= tolerance_px)
    )

    matched_gt = (
        gt
        & (dist_to_pred <= tolerance_px)
    )

    prec_tol = (
        matched_pred.sum()
        / n_pred
    )

    rec_tol = (
        matched_gt.sum()
        / n_gt
    )

    if (
        prec_tol + rec_tol
    ) > 0:

        f1_tol = (
            2.0
            * prec_tol
            * rec_tol
            / (
                prec_tol
                + rec_tol
            )
        )

    else:

        f1_tol = 0.0

    return {

        "tolerance_px":
            tolerance_px,

        "prec_tol":
            float(prec_tol),

        "rec_tol":
            float(rec_tol),

        "f1_tol":
            float(f1_tol)
    }

# ============================================================
# DATA PARALLEL CHECKPOINT HELPER
# ============================================================

def _strip_module_prefix(
    state_dict: Dict[
        str,
        torch.Tensor
    ]
) -> Dict[str, torch.Tensor]:
    """
    Checkpoints trained using DataParallel may contain:

        module.backbone....
        module.classifier....

    This removes the 'module.' prefix.
    """

    if not state_dict:

        return state_dict

    first_key = next(
        iter(
            state_dict.keys()
        )
    )

    if first_key.startswith("module."):

        return {

            k.replace(
                "module.",
                "",
                1
            ): v

            for k, v
            in state_dict.items()
        }

    return state_dict

# ============================================================
# INFERENCE
# ============================================================

@torch.inference_mode()
def run_resnet_inference(

    image_path: str,

    mask_path: str,

    *,

    ckpt_path: Optional[str] = None,

    checkpoints_dir: Optional[str] = None,

    epoch: Optional[int] = None,

    model_name: str = "deeplabv3_resnet50",

    aux_loss: bool = False,

    image_size: int = 512,

    threshold: float = 0.5,

    distance_tolerances=(2, 3, 4, 5),

    overlay_alpha: float = 0.35,

    imagenet_normalization: bool = False,

    device_preference: Optional[str] = None,

    save_dir: Optional[str] = None,

    save_basename: Optional[str] = None,

    save_overlay: bool = True,

    save_probs: bool = True,

    show_plots: bool = True,

    show_with_cv2: bool = False

) -> Dict[str, Any]:

    """
    Run inference on one image.

    Supply either:

        ckpt_path

    OR:

        checkpoints_dir + epoch
    """

    # --------------------------------------------------------
    # Determine checkpoint
    # --------------------------------------------------------

    if ckpt_path is None:

        if (
            checkpoints_dir is None
            or epoch is None
        ):

            raise ValueError(
                "Provide either ckpt_path OR "
                "(checkpoints_dir and epoch)."
            )

        ckpt_path = os.path.join(

            checkpoints_dir,

            f"bsr_resnet_epoch_{int(epoch):03d}.pth"
        )

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = pick_device(
        device_preference
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Model: {model_name} | "
        f"aux_loss={aux_loss}"
    )

    print(
        f"Checkpoint: {ckpt_path}"
    )

    print(
        f"ImageNet normalization: "
        f"{imagenet_normalization}"
    )

    # ========================================================
    # BUILD MODEL
    # ========================================================

    # IMPORTANT:
    #
    # pretrained=False is intentional.
    #
    # ImageNet weights were used to INITIALIZE training.
    # The final .pth checkpoint contains the fine-tuned
    # backbone weights.
    #
    # Therefore inference does NOT need to download/reload
    # ImageNet weights.

    model = build_resnet_segmentation(

        model_name=model_name,

        pretrained=False,

        aux_loss=aux_loss

    ).to(device)

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    ckpt = torch.load(

        ckpt_path,

        map_location=device
    )

    # Some training scripts save:
    #
    # {
    #     "state_dict": model.state_dict(),
    #     ...
    # }

    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
    ):

        ckpt = ckpt[
            "state_dict"
        ]

    ckpt = _strip_module_prefix(
        ckpt
    )

    # --------------------------------------------------------
    # STRICT checkpoint loading
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            ckpt,
            strict=True
        )

        print(
            "Checkpoint loaded successfully "
            "with strict=True."
        )

    except RuntimeError as e:

        print(
            "\nERROR: Checkpoint architecture "
            "does not exactly match the "
            "inference architecture.\n"
        )

        raise e

    model.eval()

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    rgb, (h0, w0), x = load_image_rgb_tensor(

        image_path,

        size=image_size,

        imagenet_normalization=
            imagenet_normalization
    )

    x = x.to(

        device,

        non_blocking=True
    )

    # ========================================================
    # FORWARD PASS
    # ========================================================

    if device.type == "cuda":

        torch.cuda.synchronize()

    t0 = time.time()

    out = model(x)

    logits = _ensure_logits(
        out
    )

    if logits is None:

        raise RuntimeError(
            "Model did not return logits "
            "or an 'out' tensor."
        )

    if device.type == "cuda":

        torch.cuda.synchronize()

    inference_time = (
        time.time()
        - t0
    )

    # ========================================================
    # SIGMOID PROBABILITY
    # ========================================================

    prob_small = (

        torch.sigmoid(logits)[
            0,
            0
        ]

        .detach()

        .cpu()

        .numpy()
    )

    # Convert probability to 0-255
    prob_u8_small = (

        prob_small
        * 255

    ).astype(
        np.uint8
    )

    # Resize probability back to original image size
    prob_u8 = cv2.resize(

        prob_u8_small,

        (w0, h0),

        interpolation=cv2.INTER_LINEAR
    )

    prob_float = (

        prob_u8.astype(np.float32)

        / 255.0
    )

    # ========================================================
    # BINARY PREDICTION
    # ========================================================

    pred_bin = (

        prob_float >= threshold

    ).astype(
        np.uint8
    )

    # ========================================================
    # LOAD GROUND TRUTH
    # ========================================================

    gt_bin = load_mask_binary(mask_path, (h0, w0))

    # ========================================================
    # OVERLAYS
    # ========================================================

    gt_overlay = overlay_colored(
        rgb,
        gt_bin,
        alpha=overlay_alpha,
        color=(0, 200, 0)
    )

    pred_overlay = overlay_colored(

        rgb,

        pred_bin,

        alpha=overlay_alpha,

        color=(255, 0, 0)
    )

    # ========================================================
    # PREDICTION STATISTICS
    # ========================================================

    area_pct = (

        100.0
        * float(
            pred_bin.mean()
        )
    )

    if pred_bin.any():

        mean_prob_in_mask = float(

            prob_float[
                pred_bin.astype(bool)
            ].mean()
        )

        max_prob_in_mask = float(

            prob_float[
                pred_bin.astype(bool)
            ].max()
        )

    else:

        mean_prob_in_mask = 0.0
        max_prob_in_mask = 0.0

    global_mean_prob = float(

        prob_float.mean()
    )

    print(
        "\n"
        "==============================="
    )

    print(
        "      PREDICTION SUMMARY"
    )

    print(
        "==============================="
    )

    print(
        f"Inference time: "
        f"{inference_time:.3f} s"
    )

    print(
        f"Threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"BSR area: "
        f"{area_pct:.2f}%"
    )

    print(
        f"Mean probability "
        f"(predicted BSR): "
        f"{mean_prob_in_mask:.3f}"
    )

    print(
        f"Maximum probability "
        f"(predicted BSR): "
        f"{max_prob_in_mask:.3f}"
    )

    print(
        f"Global mean probability: "
        f"{global_mean_prob:.3f}"
    )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = None

    tolerance_metrics = None

    if gt_bin is not None:

        metrics = compute_metrics(

            pred_bin,

            gt_bin
        )

        print(
            "\n"
            "-------------------------------"
        )

        print(
            "STRICT PIXEL-WISE METRICS"
        )

        print(
            "-------------------------------"
        )

        print(
            f"IoU:       "
            f"{metrics['iou']:.4f}"
        )

        print(
            f"F1:        "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"Precision: "
            f"{metrics['prec']:.4f}"
        )

        print(
            f"Recall:    "
            f"{metrics['rec']:.4f}"
        )

        print(
            f"Accuracy:  "
            f"{metrics['acc']:.4f}"
        )

        # ----------------------------------------------------
        # TOLERANCE METRICS
        # ----------------------------------------------------

        tolerance_metrics = {}

        print(
            "\n"
            "-------------------------------"
        )

        print(
            "DISTANCE-TOLERANT METRICS"
        )

        print(
            "-------------------------------"
        )

        for tol in distance_tolerances:

            tm = compute_tolerance_metrics(

                pred_bin,

                gt_bin,

                tolerance_px=tol
            )

            tolerance_metrics[
                tol
            ] = tm

            print(
                f"{tol}px tolerance | "
                f"Precision: "
                f"{tm['prec_tol']:.4f} | "
                f"Recall: "
                f"{tm['rec_tol']:.4f} | "
                f"F1: "
                f"{tm['f1_tol']:.4f}"
            )

    else:

        print(
            "\nNo ground-truth mask "
            "provided. Metrics skipped."
        )

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    if save_dir:

        os.makedirs(
            save_dir,
            exist_ok=True
        )

        if save_basename:

            base = save_basename

        else:

            base = os.path.splitext(

                os.path.basename(
                    image_path
                )

            )[0]

        # Probability
        if save_probs:

            cv2.imwrite(

                os.path.join(
                    save_dir,
                    f"{base}_pred_prob.png"
                ),

                prob_u8
            )

        # Binary mask
        cv2.imwrite(

            os.path.join(
                save_dir,
                f"{base}_pred_mask.png"
            ),

            (
                pred_bin * 255
            ).astype(
                np.uint8
            )
        )

        # Prediction overlay
        if save_overlay:

            cv2.imwrite(

                os.path.join(
                    save_dir,
                    f"{base}_overlay_pred.png"
                ),

                cv2.cvtColor(

                    pred_overlay,

                    cv2.COLOR_RGB2BGR
                )
            )

            # GT overlay
            if gt_overlay is not None:

                cv2.imwrite(

                    os.path.join(
                        save_dir,
                        f"{base}_overlay_gt.png"
                    ),

                    cv2.cvtColor(

                        gt_overlay,

                        cv2.COLOR_RGB2BGR
                    )
                )

    # ========================================================
    # PLOT
    # ========================================================

    if show_plots:

        fig, axes = plt.subplots(

            2,
            2,

            figsize=(14, 12),

            constrained_layout=True
        )

        axes = axes.ravel()

        # Original
        axes[0].imshow(
            rgb
        )

        axes[0].set_title(
            "Original Image",
            fontsize=16
        )

        axes[0].axis(
            "off"
        )

        # Ground Truth
        if gt_overlay is not None:

            axes[1].imshow(
                gt_overlay
            )

            axes[1].set_title(
                "Ground Truth Overlay",
                fontsize=16
            )

        else:

            axes[1].imshow(
                rgb
            )

            axes[1].set_title(
                "Ground Truth: None",
                fontsize=16
            )

        axes[1].axis(
            "off"
        )

        # Prediction
        axes[2].imshow(
            pred_overlay
        )

        axes[2].set_title(

            f"Predicted BSR "
            f"(threshold = {threshold:.2f})",

            fontsize=16
        )

        axes[2].axis(
            "off"
        )

        # Probability
        im = axes[3].imshow(

            prob_float,

            cmap="jet",

            vmin=0,

            vmax=1
        )

        axes[3].set_title(
            "BSR Probability",
            fontsize=16
        )

        axes[3].axis(
            "off"
        )

        cbar = fig.colorbar(

            im,

            ax=axes[3],

            fraction=0.046,

            pad=0.04
        )

        cbar.set_label(

            "BSR Probability",

            fontsize=14
        )

        cbar.ax.tick_params(
            labelsize=12
        )

        # Save combined figure
        if save_dir:

            figure_path = os.path.join(

                save_dir,

                f"{base}_prediction_summary.png"
            )

            plt.savefig(

                figure_path,

                dpi=300,

                bbox_inches="tight"
            )

            print(
                f"\nPrediction figure saved to:\n"
                f"{figure_path}"
            )

        plt.show()

    # ========================================================
    # OPTIONAL OPENCV DISPLAY
    # ========================================================

    if show_with_cv2:

        cv2.imshow(

            "Original",

            cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR
            )
        )

        cv2.imshow(

            "Probability",

            prob_u8
        )

        cv2.imshow(

            "Binary Prediction",

            (
                pred_bin
                * 255
            ).astype(
                np.uint8
            )
        )

        cv2.imshow(

            "Prediction Overlay",

            cv2.cvtColor(

                pred_overlay,

                cv2.COLOR_RGB2BGR
            )
        )

        if gt_overlay is not None:

            cv2.imshow(

                "Ground Truth Overlay",

                cv2.cvtColor(

                    gt_overlay,

                    cv2.COLOR_RGB2BGR
                )
            )

        print(
            "Press any key in the image "
            "window to close."
        )

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {

        "rgb":
            rgb,

        "prob_u8":
            prob_u8,

        "prob_float":
            prob_float,

        "pred_bin":
            pred_bin,

        "overlay_pred_rgb":
            pred_overlay,

        "overlay_gt_rgb":
            gt_overlay,

        "summary": {

            "threshold":
                threshold,

            "area_percent":
                area_pct,

            "mean_prob_inside_pred_mask":
                mean_prob_in_mask,

            "max_prob_inside_pred_mask":
                max_prob_in_mask,

            "global_mean_prob":
                global_mean_prob,

            "metrics":
                metrics,

            "tolerance_metrics":
                tolerance_metrics,

            "inference_seconds":
                inference_time,

            "ckpt_path":
                ckpt_path,

            "model_name":
                model_name,

            "aux_loss":
                aux_loss,

            "imagenet_normalization":
                imagenet_normalization
        }
    }

# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # PATHS
    # --------------------------------------------------------

    MODEL_PATH = (
        r"/content/BSR-detection-using-Computer-Vision/"
        r"models/best_resnet_bsr.pth"
    )
    IMAGE_PATH = (
        r"/content/BSR-detection-using-Computer-Vision/"
        r"Bonaventure_BSR4.png"
    )
        # Ground-truth mask corresponding to IMAGE_PATH
    GT_MASK = (
        r"/content/BSR-detection-using-Computer-Vision/"
        r"masks/Bonaventure_BSR4_label.png"
    )

    OUT_DIR = (
        r"/content/BSR-detection-using-Computer-Vision/"
        r"prediction_outputs"
    )

    # ========================================================
    # RUN INFERENCE
    # ========================================================

    res = run_resnet_inference(

        image_path=IMAGE_PATH,

        mask_path=GT_MASK,

        ckpt_path=MODEL_PATH,

        # ----------------------------------------------------
        # Architecture
        # ----------------------------------------------------

        model_name="deeplabv3_resnet50",

        # MUST match training
        aux_loss=False,

        # ----------------------------------------------------
        # Image settings
        # ----------------------------------------------------

        image_size=512,

        threshold=0.5,

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Keep False if training only used:
        #
        #     image / 255.0
        #
        # Set True ONLY if training additionally used:
        #
        # mean = [0.485, 0.456, 0.406]
        # std  = [0.229, 0.224, 0.225]
        # ----------------------------------------------------

        imagenet_normalization=False,

        # ----------------------------------------------------
        # Distance-tolerant evaluation
        # ----------------------------------------------------

        distance_tolerances=(
            2,
            3,
            4,
            5
        ),

        overlay_alpha=0.35,

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        device_preference=None,

        # ----------------------------------------------------
        # Saving
        # ----------------------------------------------------

        save_dir=OUT_DIR,
        save_basename=None,
        save_overlay=True,
        save_probs=True,

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        show_plots=True,
        show_with_cv2=False
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\nDone."
    )
    print(
        "RGB shape:",
        res["rgb"].shape
    )
    print(
        "Probability shape:",
        res["prob_u8"].shape
    )
    print(
        "Prediction mask shape:",
        res["pred_bin"].shape
    )
    print(
        "\nSummary:"
    )
    print(
        res["summary"]
    )
