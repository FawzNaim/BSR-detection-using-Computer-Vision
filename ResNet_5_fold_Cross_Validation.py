import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
    Subset
)

from sklearn.model_selection import KFold
import torchvision.models.segmentation as models


# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42

PARAMS = {
    "img_size": 512,
    "batch_size": 8,
    "epochs": 100,
    "lr": 3e-4,
    "weight_decay": 1e-4,

    # Positive-pixel weight for BCEWithLogitsLoss
    "pos_weight": 15.0,

    # BCE and Dice contribution to total loss
    "bce_weight": 0.5,
    "dice_weight": 0.5,

    # Probability threshold for converting probabilities to masks
    "threshold": 0.5,

    # Positive images receive five times the sampling weight
    "positive_sampling_weight": 5.0,

    "num_folds": 5,
    "num_workers": 2,

    # Add your paths here
    "img_dir": r"images",
    "mask_dir": r"masks",

    "checkpoint_dir": "checkpoints"
}


# ============================================================
# REPRODUCIBILITY
# ============================================================
def setup_env(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is preferable for cross-validation
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================
class BSRDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=512):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size

        all_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(".png")
        ])

        # Retain only images with corresponding readable masks
        self.files = []
        self.has_pos = []

        pos_pixels = 0
        total_pixels = 0

        for fname in all_files:
            image_path = os.path.join(img_dir, fname)
            mask_name = os.path.splitext(fname)[0] + "_label.png"
            mask_path = os.path.join(mask_dir, mask_name)

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Warning: could not read image: {image_path}")
                continue

            if mask is None:
                print(f"Warning: could not read mask: {mask_path}")
                continue

            mask = cv2.resize(
                mask,
                (size, size),
                interpolation=cv2.INTER_NEAREST
            )

            positive_count = np.sum(mask > 0)

            self.files.append(fname)
            self.has_pos.append(positive_count > 0)

            pos_pixels += positive_count
            total_pixels += size * size

        if len(self.files) == 0:
            raise RuntimeError(
                "No valid image-mask pairs were found. "
                "Check img_dir, mask_dir, and mask filenames."
            )

        self.calculated_pos_weight = (
            (total_pixels - pos_pixels) / (pos_pixels + 1e-7)
        )

        print(
            f"Dataset ready: {len(self.files)} valid image-mask pairs\n"
            f"Calculated pixel pos_weight: "
            f"{self.calculated_pos_weight:.2f}"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        image_path = os.path.join(self.img_dir, fname)
        mask_name = os.path.splitext(fname)[0] + "_label.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (self.size, self.size),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (self.size, self.size),
            interpolation=cv2.INTER_NEAREST
        )

        # Image shape: [C, H, W], values between 0 and 1
        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        ).float() / 255.0

        # Binary mask shape: [1, H, W]
        mask = torch.from_numpy(
            (mask > 0).astype(np.float32)
        ).unsqueeze(0)

        return image, mask


# ============================================================
# MODEL
# ============================================================
def get_segmentation_model(name="deeplabv3_resnet50"):
    if name == "deeplabv3_resnet50":
        model = models.deeplabv3_resnet50(weights="DEFAULT")

        # Replace the multiclass output layer with one binary channel
        in_channels = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1
        )

        if model.aux_classifier is not None:
            aux_in_channels = model.aux_classifier[-1].in_channels
            model.aux_classifier[-1] = nn.Conv2d(
                aux_in_channels,
                1,
                kernel_size=1
            )

    elif name == "fcn_resnet50":
        model = models.fcn_resnet50(weights="DEFAULT")

        in_channels = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(
            in_channels,
            1,
            kernel_size=1
        )

        if model.aux_classifier is not None:
            aux_in_channels = model.aux_classifier[-1].in_channels
            model.aux_classifier[-1] = nn.Conv2d(
                aux_in_channels,
                1,
                kernel_size=1
            )

    else:
        raise ValueError(f"Unsupported model name: {name}")

    return model


# ============================================================
# DICE LOSS
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Convert logits to probabilities
        probabilities = torch.sigmoid(logits.float())
        targets = targets.float()

        # Calculate Dice independently for every image
        dimensions = (1, 2, 3)

        intersection = (
            probabilities * targets
        ).sum(dim=dimensions)

        denominator = (
            probabilities.sum(dim=dimensions)
            + targets.sum(dim=dimensions)
        )

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice_score.mean()


# ============================================================
# METRIC CALCULATION
# ============================================================
def update_confusion_counts(
    logits,
    targets,
    threshold=0.5
):
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold
    truth = targets >= 0.5

    true_positive = torch.logical_and(
        predictions,
        truth
    ).sum().item()

    false_positive = torch.logical_and(
        predictions,
        torch.logical_not(truth)
    ).sum().item()

    false_negative = torch.logical_and(
        torch.logical_not(predictions),
        truth
    ).sum().item()

    return true_positive, false_positive, false_negative


def calculate_metrics(tp, fp, fn, epsilon=1e-7):
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = (
        2.0 * precision * recall
        / (precision + recall + epsilon)
    )

    iou = tp / (tp + fp + fn + epsilon)

    return {
        "iou": iou,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# ============================================================
# TRAIN ONE FOLD
# ============================================================
def train_fold(
    fold_idx,
    train_loader,
    val_loader,
    device,
    epochs=100
):
    print(f"\nStarting fold {fold_idx}")

    model = get_segmentation_model(
        name="deeplabv3_resnet50"
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PARAMS["lr"],
        weight_decay=PARAMS["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    positive_weight = torch.tensor(
        [PARAMS["pos_weight"]],
        dtype=torch.float32,
        device=device
    )

    bce_loss_function = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight
    )

    dice_loss_function = DiceLoss(smooth=1.0)

    # AMP is enabled only when training on a CUDA GPU
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled
    )

    best_iou = -1.0
    best_f1 = 0.0
    best_epoch = 0

    os.makedirs(
        PARAMS["checkpoint_dir"],
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        PARAMS["checkpoint_dir"],
        f"best_f{fold_idx}.pth"
    )

    for epoch in range(epochs):
        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------
        model.train()

        train_losses = []
        train_bce_losses = []
        train_dice_losses = []

        start_time = time.time()

        for images, masks in train_loader:
            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled
            ):
                output = model(images)["out"]

                bce_loss = bce_loss_function(
                    output,
                    masks
                )

                dice_loss = dice_loss_function(
                    output,
                    masks
                )

                total_loss = (
                    PARAMS["bce_weight"] * bce_loss
                    + PARAMS["dice_weight"] * dice_loss
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(total_loss.item())
            train_bce_losses.append(bce_loss.item())
            train_dice_losses.append(dice_loss.item())

        scheduler.step()

        mean_train_loss = np.mean(train_losses)
        mean_bce_loss = np.mean(train_bce_losses)
        mean_dice_loss = np.mean(train_dice_losses)

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------
        model.eval()

        val_losses = []
        total_tp = 0
        total_fp = 0
        total_fn = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(
                    device,
                    non_blocking=True
                )

                masks = masks.to(
                    device,
                    non_blocking=True
                )

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=amp_enabled
                ):
                    output = model(images)["out"]

                    val_bce_loss = bce_loss_function(
                        output,
                        masks
                    )

                    val_dice_loss = dice_loss_function(
                        output,
                        masks
                    )

                    val_total_loss = (
                        PARAMS["bce_weight"] * val_bce_loss
                        + PARAMS["dice_weight"] * val_dice_loss
                    )

                val_losses.append(val_total_loss.item())

                tp, fp, fn = update_confusion_counts(
                    output,
                    masks,
                    threshold=PARAMS["threshold"]
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn

        mean_val_loss = np.mean(val_losses)

        val_metrics = calculate_metrics(
            total_tp,
            total_fp,
            total_fn
        )

        val_iou = val_metrics["iou"]
        val_f1 = val_metrics["f1"]
        val_precision = val_metrics["precision"]
        val_recall = val_metrics["recall"]

        elapsed_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Fold {fold_idx} | "
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Train Loss: {mean_train_loss:.4f} | "
            f"BCE: {mean_bce_loss:.4f} | "
            f"Dice: {mean_dice_loss:.4f} | "
            f"Val Loss: {mean_val_loss:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Precision: {val_precision:.4f} | "
            f"Recall: {val_recall:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed_time:.1f}s"
        )

        # Save using validation IoU
        if val_iou > best_iou:
            best_iou = val_iou
            best_f1 = val_f1
            best_epoch = epoch + 1

            torch.save(
                {
                    "fold": fold_idx,
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "validation_iou": best_iou,
                    "validation_f1": best_f1,
                    "validation_precision": val_precision,
                    "validation_recall": val_recall,
                    "parameters": PARAMS
                },
                checkpoint_path
            )

    print(
        f"\nFold {fold_idx} complete | "
        f"Best epoch: {best_epoch} | "
        f"Best IoU: {best_iou:.4f} | "
        f"F1 at best epoch: {best_f1:.4f}"
    )

    return {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "iou": best_iou,
        "f1": best_f1
    }


# ============================================================
# FIVE-FOLD CROSS-VALIDATION
# ============================================================
def main():
    setup_env()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    dataset = BSRDataset(
        img_dir=PARAMS["img_dir"],
        mask_dir=PARAMS["mask_dir"],
        size=PARAMS["img_size"]
    )

    if len(dataset) < PARAMS["num_folds"]:
        raise ValueError(
            f"The dataset contains only {len(dataset)} images, "
            f"which is insufficient for "
            f"{PARAMS['num_folds']}-fold cross-validation."
        )

    kfold = KFold(
        n_splits=PARAMS["num_folds"],
        shuffle=True,
        random_state=SEED
    )

    fold_results = []

    for fold_idx, (train_indices, val_indices) in enumerate(
        kfold.split(np.arange(len(dataset))),
        start=1
    ):
        print("\n" + "=" * 70)
        print(
            f"Fold {fold_idx}/{PARAMS['num_folds']} | "
            f"Training images: {len(train_indices)} | "
            f"Validation images: {len(val_indices)}"
        )
        print("=" * 70)

        # Oversampling weights for the training portion only
        training_positive_flags = [
            dataset.has_pos[index]
            for index in train_indices
        ]

        sampling_weights = [
            PARAMS["positive_sampling_weight"]
            if has_positive_pixels
            else 1.0
            for has_positive_pixels in training_positive_flags
        ]

        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(SEED + fold_idx)

        sampler = WeightedRandomSampler(
            weights=sampling_weights,
            num_samples=len(train_indices),
            replacement=True,
            generator=sampler_generator
        )

        train_subset = Subset(
            dataset,
            train_indices.tolist()
        )

        val_subset = Subset(
            dataset,
            val_indices.tolist()
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=PARAMS["batch_size"],
            sampler=sampler,
            num_workers=PARAMS["num_workers"],
            pin_memory=device.type == "cuda"
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=PARAMS["batch_size"],
            shuffle=False,
            num_workers=PARAMS["num_workers"],
            pin_memory=device.type == "cuda"
        )

        fold_result = train_fold(
            fold_idx=fold_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=PARAMS["epochs"]
        )

        fold_results.append(fold_result)

    # --------------------------------------------------------
    # Final cross-validation results
    # --------------------------------------------------------
    fold_ious = [
        result["iou"]
        for result in fold_results
    ]

    fold_f1_scores = [
        result["f1"]
        for result in fold_results
    ]

    print("\n" + "=" * 70)
    print("FIVE-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 70)

    for result in fold_results:
        print(
            f"Fold {result['fold']} | "
            f"Best epoch: {result['best_epoch']} | "
            f"IoU: {result['iou']:.4f} | "
            f"F1: {result['f1']:.4f}"
        )

    print("-" * 70)

    print(
        f"Mean IoU: {np.mean(fold_ious):.4f} "
        f"± {np.std(fold_ious, ddof=1):.4f}"
    )

    print(
        f"Mean F1: {np.mean(fold_f1_scores):.4f} "
        f"± {np.std(fold_f1_scores, ddof=1):.4f}"
    )


if __name__ == "__main__":
    main()
