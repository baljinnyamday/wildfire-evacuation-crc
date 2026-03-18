"""Phase 3: train a Tiny U-Net spatial baseline and export probability heatmaps."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fire.data.constants import DEFAULT_SEED, DEFAULT_TARGET_FEATURE
from fire.data.pipeline import NDWSPipeline


class ConvBlock(nn.Module):
    """Two-layer conv block used across encoder/decoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    """Shallow 2-downsampling U-Net with sigmoid probability output."""

    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.head(d1)
        return torch.sigmoid(logits)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "tfrecord", "tif"],
        help="Source file type.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/splits/ndws_seed42_70_15_15.json"),
        help="Deterministic split manifest generated during Phase 1.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/predictions/spatial_baseline"),
        help="Where checkpoint and split probability heatmaps are saved.",
    )
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _train_one_epoch(
    model: TinyUNet,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_valid_pixels = 0
    for x, y, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        probs = model(x)
        pixel_loss = criterion(probs, y)
        masked_loss = (pixel_loss * mask).sum()
        n_valid = mask.sum()
        if n_valid > 0:
            avg_loss = masked_loss / n_valid
        else:
            avg_loss = masked_loss
        avg_loss.backward()
        optimizer.step()

        total_loss += float(masked_loss.item())
        total_valid_pixels += int(n_valid.item())

    if total_valid_pixels == 0:
        raise RuntimeError("Train dataloader produced zero valid pixels.")
    return total_loss / total_valid_pixels


@torch.no_grad()
def _validate(
    model: TinyUNet,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_valid_pixels = 0
    for x, y, mask in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        probs = model(x)
        pixel_loss = criterion(probs, y)
        masked_loss = (pixel_loss * mask).sum()
        n_valid = mask.sum()

        total_loss += float(masked_loss.item())
        total_valid_pixels += int(n_valid.item())

    if total_valid_pixels == 0:
        raise RuntimeError("Validation dataloader produced zero valid pixels.")
    return total_loss / total_valid_pixels


def _export_split_probability_heatmaps(
    model: TinyUNet,
    loader,
    split_name: str,
    output_dir: Path,
    device: torch.device,
) -> dict[str, int]:
    model.eval()
    sample_ids: list[str] = []
    probs_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []

    mask_batches: list[np.ndarray] = []

    with torch.no_grad():
        for x, y, mask, batch_sample_ids in loader:
            x = x.to(device, non_blocking=True)
            probs = model(x).cpu().numpy()[:, 0, :, :].astype(np.float32)
            targets = y.numpy()[:, 0, :, :]

            probs_batches.append(probs)
            target_batches.append(targets.astype(np.uint8))
            mask_batches.append(mask.numpy()[:, 0, :, :].astype(np.uint8))
            sample_ids.extend([str(sid) for sid in batch_sample_ids])

    if not probs_batches:
        raise RuntimeError(f"{split_name} dataloader produced zero batches.")

    prob_maps = np.concatenate(probs_batches, axis=0)
    target_maps = np.concatenate(target_batches, axis=0)
    valid_masks = np.concatenate(mask_batches, axis=0)
    max_len = max(len(sample_id) for sample_id in sample_ids)
    output_path = output_dir / f"{split_name}_probability_heatmaps.npz"
    np.savez_compressed(
        output_path,
        split=split_name,
        sample_ids=np.asarray(sample_ids, dtype=f"<U{max_len}"),
        targets=target_maps,
        valid_masks=valid_masks,
        probabilities=prob_maps,
    )
    return {
        "samples": int(prob_maps.shape[0]),
        "height": int(prob_maps.shape[1]),
        "width": int(prob_maps.shape[2]),
    }


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    device = _resolve_device(args.device)

    pipeline = NDWSPipeline(
        data_root=args.data_root,
        input_format=args.input_format,
        seed=args.seed,
        split_manifest_path=args.split_manifest,
    )
    feature_names = list(pipeline.input_features or [])

    model = TinyUNet(in_channels=len(feature_names), base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCELoss(reduction="none")

    train_loader = pipeline.build_spatial_dataloader(
        split="train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = pipeline.build_spatial_dataloader(
        split="calibration",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_state: dict | None = None
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = _validate(model, val_loader, criterion, device)
        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} - "
            f"train_bce: {train_loss:.6f}  val_bce: {val_loss:.6f}  lr: {lr_now:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ↳ New best val_bce: {val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    print(f"Restoring best model (val_bce={best_val_loss:.6f}) for export.")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": args.seed,
            "input_features": feature_names,
            "target_name": DEFAULT_TARGET_FEATURE,
            "base_channels": args.base_channels,
            "best_val_loss": best_val_loss,
            "epochs_trained": args.epochs,
        },
        output_dir / "tiny_unet_checkpoint.pt",
    )

    split_export_shapes: dict[str, dict[str, int]] = {}
    for split_name in ("calibration", "test"):
        eval_loader = pipeline.build_spatial_dataloader(
            split=split_name,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            return_sample_id=True,
        )
        split_export_shapes[split_name] = _export_split_probability_heatmaps(
            model=model,
            loader=eval_loader,
            split_name=split_name,
            output_dir=output_dir,
            device=device,
        )
        print(f"Saved {split_name} probability heatmaps.")

    metadata = {
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "base_channels": args.base_channels,
        "scheduler": "CosineAnnealingLR",
        "best_val_loss": best_val_loss,
        "input_features": feature_names,
        "target_name": DEFAULT_TARGET_FEATURE,
        "split_sizes": pipeline.split_sizes(),
        "exports": split_export_shapes,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved spatial baseline artifacts to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
