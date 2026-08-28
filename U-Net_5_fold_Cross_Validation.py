import os
import gc
import cv2
import csv
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datetime import datetime
from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler,
    Subset
)
from sklearn.model_selection import KFold


# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42

PARAMS = {
    "img_size": 512,
    "batch_size": 8,
    "epochs": 100,
    "num_folds": 5,
    "num_workers": 2,

    "lr": 3e-4,
    "weight_decay": 1e-4,

    # Weighted BCE + Dice
    "pos_weight": 25.0,
    "bce_weight": 0.5,
    "dice_weight": 0.5,

    # Binary segmentation threshold
    "threshold": 0.5,

    # Image-level oversampling
    "positive_sampling_weight": 5.0,

    # Add your paths here
    "img_dir": r"images",
    "mask_dir": r"masks",

    # Output location
    "output_dir": "cv_outputs_unet"
}


# ============================================================
# REPRODUCIBILITY
# ============================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# U-NET MODEL
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Encoder
        self.encoder1 = ConvBlock(3, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 512)

        # Decoder level 4
        self.up4 = nn.ConvTranspose2d(
            512,
            512,
            kernel_size=2,
            stride=2
        )
        self.decoder4 = ConvBlock(
            512 + 512,
            512
        )

        # Decoder level 3
        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )
        self.decoder3 = ConvBlock(
            256 + 256,
            256
        )

        # Decoder level 2
        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )
        self.decoder2 = ConvBlock(
            128 + 128,
            128
        )

        # Decoder level 1
        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )
        self.decoder1 = ConvBlock(
            64 + 64,
            64
        )

        # One-channel binary output
        self.output_layer = nn.Conv2d(
            64,
            1,
            kernel_size=1
        )

    def forward(self, x):
        # Encoder
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        skip4 = self.encoder4(self.pool(skip3))

        # Bottleneck
        bottleneck = self.bottleneck(
            self.pool(skip4)
        )

        # Decoder
        x = self.up4(bottleneck)
        x = self.decoder4(
            torch.cat([x, skip4], dim=1)
        )

        x = self.up3(x)
        x = self.decoder3(
            torch.cat([x, skip3], dim=1)
        )

        x = self.up2(x)
        x = self.decoder2(
            torch.cat([x, skip2], dim=1)
        )

        x = self.up1(x)
        x = self.decoder1(
            torch.cat([x, skip1], dim=1)
        )

        return self.output_layer(x)


# ============================================================
# DATASET
# ============================================================
class BSRDataset(Dataset):
    def __init__(
        self,
        img_dir,
        mask_dir,
        size=512
    ):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"Image directory not found: {img_dir}"
            )

        if not os.path.isdir(mask_dir):
            raise FileNotFoundError(
                f"Mask directory not found: {mask_dir}"
            )

        candidate_files = sorted([
            filename
            for filename in os.listdir(img_dir)
            if filename.lower().endswith(".png")
        ])

        self.files = []
        self.has_pos = []

        positive_pixels = 0
        total_pixels = 0

        for filename in candidate_files:
            image_path = os.path.join(
                img_dir,
                filename
            )

            mask_filename = (
                os.path.splitext(filename)[0]
                + "_label.png"
            )

            mask_path = os.path.join(
                mask_dir,
                mask_filename
            )

            image = cv2.imread(image_path)
            mask = cv2.imread(
                mask_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                print(
                    f"Warning: unreadable image skipped: "
                    f"{image_path}"
                )
                continue

            if mask is None:
                print(
                    f"Warning: missing mask skipped: "
                    f"{mask_path}"
                )
                continue

            mask = cv2.resize(
                mask,
                (size, size),
                interpolation=cv2.INTER_NEAREST
            )

            count = int(
                np.sum(mask > 0)
            )

            self.files.append(filename)
            self.has_pos.append(count > 0)

            positive_pixels += count
            total_pixels += size * size

        if len(self.files) == 0:
            raise RuntimeError(
                "No valid image-mask pairs were found."
            )

        self.positive_pixel_fraction = (
            positive_pixels
            / (total_pixels + 1e-7)
        )

        print(
            f"\nDataset: {len(self.files)} images"
        )

        print(
            f"Positive images: "
            f"{sum(self.has_pos)}/{len(self.has_pos)}"
        )

        print(
            f"Positive pixels: "
            f"{self.positive_pixel_fraction * 100:.4f}%"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]

        image_path = os.path.join(
            self.img_dir,
            filename
        )

        mask_filename = (
            os.path.splitext(filename)[0]
            + "_label.png"
        )

        mask_path = os.path.join(
            self.mask_dir,
            mask_filename
        )

        image = cv2.imread(image_path)

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise FileNotFoundError(
                f"Image could not be read: {image_path}"
            )

        if mask is None:
            raise FileNotFoundError(
                f"Mask could not be read: {mask_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

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

        image = (
            image.transpose(2, 0, 1)
            .astype(np.float32)
            / 255.0
        )

        mask = (
            mask > 0
        ).astype(np.float32)

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        ).unsqueeze(0)

        return image, mask


# ============================================================
# BCE + DICE LOSS
# ============================================================
def dice_loss_from_logits(
    logits,
    targets,
    smooth=1.0
):
    probabilities = torch.sigmoid(
        logits.float()
    )

    targets = targets.float()

    dimensions = (1, 2, 3)

    intersection = (
        probabilities * targets
    ).sum(dim=dimensions)

    denominator = (
        probabilities.sum(dim=dimensions)
        + targets.sum(dim=dimensions)
    )

    dice_score = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return 1.0 - dice_score.mean()


# ============================================================
# METRICS
# ============================================================
def get_confusion_counts(
    logits,
    targets,
    threshold=0.5
):
    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    )

    truth = (
        targets >= 0.5
    )

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

    return (
        true_positive,
        false_positive,
        false_negative
    )


def calculate_metrics(
    true_positive,
    false_positive,
    false_negative,
    epsilon=1e-7
):
    precision = true_positive / (
        true_positive
        + false_positive
        + epsilon
    )

    recall = true_positive / (
        true_positive
        + false_negative
        + epsilon
    )

    f1 = (
        2.0 * true_positive
    ) / (
        2.0 * true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    iou = true_positive / (
        true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "iou": iou
    }


# ============================================================
# FOLD MEMBERSHIP CSV
# ============================================================
def write_fold_membership_csv(
    dataset,
    fold_id,
    train_indices,
    val_indices,
    output_dir
):
    membership_path = os.path.join(
        output_dir,
        f"fold_{fold_id:02d}_membership.csv"
    )

    fieldnames = [
        "fold",
        "split",
        "dataset_index",
        "image_filename",
        "mask_filename",
        "has_positive"
    ]

    with open(
        membership_path,
        "w",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for index in train_indices:
            image_filename = (
                dataset.files[index]
            )

            writer.writerow({
                "fold": fold_id,
                "split": "train",
                "dataset_index": index,
                "image_filename": image_filename,
                "mask_filename": (
                    os.path.splitext(image_filename)[0]
                    + "_label.png"
                ),
                "has_positive": int(
                    dataset.has_pos[index]
                )
            })

        for index in val_indices:
            image_filename = (
                dataset.files[index]
            )

            writer.writerow({
                "fold": fold_id,
                "split": "validation",
                "dataset_index": index,
                "image_filename": image_filename,
                "mask_filename": (
                    os.path.splitext(image_filename)[0]
                    + "_label.png"
                ),
                "has_positive": int(
                    dataset.has_pos[index]
                )
            })

    return membership_path


# ============================================================
# EVALUATION
# ============================================================
def evaluate_epoch(
    model,
    data_loader,
    criterion,
    device,
    threshold=0.5,
    use_amp=True
):
    model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0
    number_of_batches = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    amp_enabled = (
        use_amp
        and device.type == "cuda"
    )

    with torch.no_grad():
        for images, masks in data_loader:
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
                logits = model(images)

                bce_loss = criterion(
                    logits,
                    masks
                )

                dice_loss = (
                    dice_loss_from_logits(
                        logits,
                        masks
                    )
                )

                combined_loss = (
                    PARAMS["bce_weight"]
                    * bce_loss
                    + PARAMS["dice_weight"]
                    * dice_loss
                )

            total_loss += combined_loss.item()
            total_bce += bce_loss.item()
            total_dice += dice_loss.item()
            number_of_batches += 1

            tp, fp, fn = get_confusion_counts(
                logits,
                masks,
                threshold=threshold
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

    if number_of_batches == 0:
        return {
            "loss": 0.0,
            "bce": 0.0,
            "dice": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "iou": 0.0
        }

    metrics = calculate_metrics(
        total_tp,
        total_fp,
        total_fn
    )

    return {
        "loss": total_loss / number_of_batches,
        "bce": total_bce / number_of_batches,
        "dice": total_dice / number_of_batches,
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "iou": metrics["iou"]
    }


# ============================================================
# TRAIN ONE FOLD
# ============================================================
def run_fold(
    fold_id,
    train_loader,
    val_loader,
    device,
    epochs=100,
    output_dir="cv_outputs_unet"
):
    model = UNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=PARAMS["lr"],
        weight_decay=PARAMS["weight_decay"]
    )

    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=epochs
        )
    )

    positive_weight = torch.tensor(
        [PARAMS["pos_weight"]],
        dtype=torch.float32,
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight
    )

    amp_enabled = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = os.path.join(
        output_dir,
        (
            f"fold_{fold_id:02d}_"
            f"training_log_{timestamp}.csv"
        )
    )

    checkpoint_path = os.path.join(
        output_dir,
        f"fold_{fold_id:02d}_best.pth"
    )

    fieldnames = [
        "fold",
        "epoch",

        "loss",
        "bce",
        "dice",
        "f1",
        "precision",
        "recall",
        "iou",

        "val_loss",
        "val_bce",
        "val_dice",
        "val_f1",
        "val_precision",
        "val_recall",
        "val_iou",

        "epoch_time_seconds",
        "average_time_per_train_batch",
        "learning_rate"
    ]

    with open(
        log_path,
        "w",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

    best_val_iou = -1.0
    best_val_f1 = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(
        1,
        epochs + 1
    ):
        model.train()

        epoch_start = time.time()

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        total_train_loss = 0.0
        total_train_bce = 0.0
        total_train_dice = 0.0
        number_of_batches = 0

        total_tp = 0
        total_fp = 0
        total_fn = 0

        for images, masks in train_loader:
            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled
            ):
                logits = model(images)

                bce_loss = criterion(
                    logits,
                    masks
                )

                dice_loss = (
                    dice_loss_from_logits(
                        logits,
                        masks
                    )
                )

                combined_loss = (
                    PARAMS["bce_weight"]
                    * bce_loss
                    + PARAMS["dice_weight"]
                    * dice_loss
                )

            scaler.scale(
                combined_loss
            ).backward()

            scaler.step(optimizer)
            scaler.update()

            total_train_loss += (
                combined_loss.item()
            )

            total_train_bce += (
                bce_loss.item()
            )

            total_train_dice += (
                dice_loss.item()
            )

            number_of_batches += 1

            tp, fp, fn = get_confusion_counts(
                logits,
                masks,
                threshold=PARAMS["threshold"]
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

        average_train_loss = (
            total_train_loss
            / number_of_batches
        )

        average_train_bce = (
            total_train_bce
            / number_of_batches
        )

        average_train_dice = (
            total_train_dice
            / number_of_batches
        )

        train_metrics = calculate_metrics(
            total_tp,
            total_fp,
            total_fn
        )

        validation_metrics = evaluate_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=PARAMS["threshold"],
            use_amp=True
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = (
            time.time() - epoch_start
        )

        average_time_per_batch = (
            epoch_time / number_of_batches
        )

        print(
            f"Fold {fold_id} | "
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Val Loss: {validation_metrics['loss']:.4f} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val F1: {validation_metrics['f1']:.4f} | "
            f"Train IoU: {train_metrics['iou']:.4f} | "
            f"Val IoU: {validation_metrics['iou']:.4f}"
        )

        epoch_row = {
            "fold": fold_id,
            "epoch": epoch,

            "loss": average_train_loss,
            "bce": average_train_bce,
            "dice": average_train_dice,
            "f1": train_metrics["f1"],
            "precision": train_metrics["precision"],
            "recall": train_metrics["recall"],
            "iou": train_metrics["iou"],

            "val_loss": validation_metrics["loss"],
            "val_bce": validation_metrics["bce"],
            "val_dice": validation_metrics["dice"],
            "val_f1": validation_metrics["f1"],
            "val_precision": (
                validation_metrics["precision"]
            ),
            "val_recall": validation_metrics["recall"],
            "val_iou": validation_metrics["iou"],

            "epoch_time_seconds": epoch_time,
            "average_time_per_train_batch": (
                average_time_per_batch
            ),
            "learning_rate": current_learning_rate
        }

        with open(
            log_path,
            "a",
            newline=""
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writerow(epoch_row)

        # Save best checkpoint using validation IoU
        if (
            validation_metrics["iou"]
            > best_val_iou
        ):
            best_val_iou = (
                validation_metrics["iou"]
            )

            best_val_f1 = (
                validation_metrics["f1"]
            )

            best_val_precision = (
                validation_metrics["precision"]
            )

            best_val_recall = (
                validation_metrics["recall"]
            )

            best_val_loss = (
                validation_metrics["loss"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "fold": fold_id,
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": (
                        optimizer.state_dict()
                    ),
                    "validation_loss": best_val_loss,
                    "validation_iou": best_val_iou,
                    "validation_f1": best_val_f1,
                    "validation_precision": (
                        best_val_precision
                    ),
                    "validation_recall": (
                        best_val_recall
                    ),
                    "parameters": PARAMS
                },
                checkpoint_path
            )

            print(
                f"New best checkpoint | "
                f"Epoch {best_epoch} | "
                f"Val IoU: {best_val_iou:.4f} | "
                f"Val F1: {best_val_f1:.4f}"
            )

        scheduler.step()

    result = {
        "fold": fold_id,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_iou": best_val_iou,
        "best_val_f1": best_val_f1,
        "best_val_precision": best_val_precision,
        "best_val_recall": best_val_recall,
        "best_checkpoint": checkpoint_path,
        "training_log_csv": log_path
    }

    del model
    del optimizer
    del scheduler
    del scaler

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()

    return result


# ============================================================
# MAIN: FIVE-FOLD CROSS-VALIDATION
# ============================================================
def main():
    seed_everything(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    os.makedirs(
        PARAMS["output_dir"],
        exist_ok=True
    )

    dataset = BSRDataset(
        img_dir=PARAMS["img_dir"],
        mask_dir=PARAMS["mask_dir"],
        size=PARAMS["img_size"]
    )

    kfold = KFold(
        n_splits=PARAMS["num_folds"],
        shuffle=True,
        random_state=SEED
    )

    all_indices = np.arange(
        len(dataset)
    )

    summary_rows = []

    for fold_id, (
        train_indices,
        val_indices
    ) in enumerate(
        kfold.split(all_indices),
        start=1
    ):
        print("\n" + "=" * 70)
        print(
            f"STARTING FOLD "
            f"{fold_id}/{PARAMS['num_folds']}"
        )
        print("=" * 70)

        train_indices = train_indices.tolist()
        val_indices = val_indices.tolist()

        membership_csv = (
            write_fold_membership_csv(
                dataset=dataset,
                fold_id=fold_id,
                train_indices=train_indices,
                val_indices=val_indices,
                output_dir=PARAMS["output_dir"]
            )
        )

        fold_positive_flags = [
            dataset.has_pos[index]
            for index in train_indices
        ]

        sampling_weights = [
            (
                PARAMS["positive_sampling_weight"]
                if has_positive
                else 1.0
            )
            for has_positive in fold_positive_flags
        ]

        sampler_generator = torch.Generator()

        sampler_generator.manual_seed(
            SEED + fold_id
        )

        sampler = WeightedRandomSampler(
            weights=sampling_weights,
            num_samples=len(train_indices),
            replacement=True,
            generator=sampler_generator
        )

        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=PARAMS["batch_size"],
            sampler=sampler,
            shuffle=False,
            num_workers=PARAMS["num_workers"],
            pin_memory=(device.type == "cuda"),
            drop_last=False
        )

        validation_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=PARAMS["batch_size"],
            shuffle=False,
            num_workers=PARAMS["num_workers"],
            pin_memory=(device.type == "cuda"),
            drop_last=False
        )

        fold_result = run_fold(
            fold_id=fold_id,
            train_loader=train_loader,
            val_loader=validation_loader,
            device=device,
            epochs=PARAMS["epochs"],
            output_dir=PARAMS["output_dir"]
        )

        fold_result["membership_csv"] = (
            membership_csv
        )

        summary_rows.append(fold_result)

        del train_loader
        del validation_loader

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    # --------------------------------------------------------
    # Save overall CV summary
    # --------------------------------------------------------
    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    summary_path = os.path.join(
        PARAMS["output_dir"],
        f"cv_summary_{timestamp}.csv"
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False
    )

    mean_iou = summary_dataframe[
        "best_val_iou"
    ].mean()

    std_iou = summary_dataframe[
        "best_val_iou"
    ].std(ddof=1)

    mean_f1 = summary_dataframe[
        "best_val_f1"
    ].mean()

    std_f1 = summary_dataframe[
        "best_val_f1"
    ].std(ddof=1)

    print("\n" + "=" * 70)
    print("FIVE-FOLD CROSS-VALIDATION COMPLETE")
    print("=" * 70)

    for result in summary_rows:
        print(
            f"Fold {result['fold']} | "
            f"Best epoch: {result['best_epoch']} | "
            f"Val IoU: {result['best_val_iou']:.4f} | "
            f"Val F1: {result['best_val_f1']:.4f} | "
            f"Precision: "
            f"{result['best_val_precision']:.4f} | "
            f"Recall: "
            f"{result['best_val_recall']:.4f}"
        )

    print("-" * 70)

    print(
        f"Mean validation IoU: "
        f"{mean_iou:.4f} ± {std_iou:.4f}"
    )

    print(
        f"Mean validation F1: "
        f"{mean_f1:.4f} ± {std_f1:.4f}"
    )

    print(
        f"Summary CSV saved to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()
