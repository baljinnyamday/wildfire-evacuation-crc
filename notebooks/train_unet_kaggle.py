"""
Tiny U-Net Training Script for Kaggle / Colab
==============================================
Self-contained — no local fire package needed.
Downloads the dataset automatically via Kaggle API.

Usage on Kaggle:
  1. Create a new Kaggle notebook
  2. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
  3. Paste this entire script into a code cell and run
  4. Download the output files from /kaggle/working/

Usage on Colab:
  1. Upload your kaggle.json to Colab (or set KAGGLE_USERNAME/KAGGLE_KEY)
  2. Enable GPU (Runtime → Change runtime type → T4)
  3. Run the script
"""

# ──────────────────────────────────────────────
# CONFIG — edit these if needed
# ──────────────────────────────────────────────
import os
import subprocess

KAGGLE_DATASET = "fantineh/next-day-wildfire-spread"
OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "outputs"

# ──────────────────────────────────────────────
# AUTO-DOWNLOAD DATASET
# ──────────────────────────────────────────────
# Try Kaggle default mount first, then download via API
KAGGLE_MOUNT = "/kaggle/input/next-day-wildfire-spread"
DOWNLOAD_DIR = os.path.join(OUTPUT_DIR, "ndws_data")

def _has_tfrecords(path: str) -> bool:
    """Check if path or any subdirectory contains TFRecord files."""
    from pathlib import Path
    p = Path(path)
    return p.exists() and (
        any(p.rglob("*.tfrecord")) or any(p.rglob("*.tfrecords"))
    )

if _has_tfrecords(KAGGLE_MOUNT):
    DATA_ROOT = KAGGLE_MOUNT
    print(f"Using Kaggle-mounted dataset at {DATA_ROOT}")
else:
    print(f"Dataset not mounted. Downloading '{KAGGLE_DATASET}' via Kaggle API...")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Install kaggle if needed (Colab)
    try:
        import kaggle  # noqa: F401
    except ImportError:
        subprocess.check_call(["pip", "install", "-q", "kaggle"])

    subprocess.check_call([
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", DOWNLOAD_DIR,
        "--unzip",
    ])
    DATA_ROOT = DOWNLOAD_DIR
    print(f"Downloaded dataset to {DATA_ROOT}")

SEED = 42
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BASE_CHANNELS = 32
TRAIN_FRACTION = 0.70
CALIBRATION_FRACTION = 0.15

INPUT_FEATURES = [
    "elevation", "th", "vs", "tmmn", "tmmx", "sph",
    "pr", "pdsi", "NDVI", "population", "erc", "PrevFireMask",
]
TARGET_FEATURE = "FireMask"
SAMPLE_SIZE = 64

# ──────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────
# SEED
# ──────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# ──────────────────────────────────────────────
# DEVICE
# ──────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ──────────────────────────────────────────────
# DATA LOADING — reads TFRecords into numpy arrays
# ──────────────────────────────────────────────
import tensorflow as tf

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

def parse_tfrecord(serialized: tf.Tensor) -> dict[str, tf.Tensor]:
    """Parse a single NDWS TFRecord example."""
    all_features = INPUT_FEATURES + [TARGET_FEATURE]
    feature_spec = {
        name: tf.io.FixedLenFeature([SAMPLE_SIZE * SAMPLE_SIZE], tf.float32)
        for name in all_features
    }
    parsed = tf.io.parse_single_example(serialized, feature_spec)
    return {
        name: tf.reshape(parsed[name], [SAMPLE_SIZE, SAMPLE_SIZE])
        for name in all_features
    }


def load_all_samples(data_root: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all TFRecord files into numpy arrays.

    Returns:
        inputs: (N, 64, 64, 12) float32
        targets: (N, 64, 64) float32
        sample_ids: list of str
    """
    root = Path(data_root)
    tfrecord_files = sorted(
        [str(p) for p in root.rglob("*.tfrecord")]
        + [str(p) for p in root.rglob("*.tfrecords")]
    )
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files in {root}")
    print(f"Found {len(tfrecord_files)} TFRecord files in {root}")

    all_inputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_ids: list[str] = []

    for file_idx, filepath in enumerate(tfrecord_files):
        filename = Path(filepath).name
        ds = tf.data.TFRecordDataset(filepath).map(
            parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE
        )
        record_idx = 0
        for record in ds:
            inp = np.stack(
                [record[feat].numpy() for feat in INPUT_FEATURES], axis=-1
            )
            tgt = record[TARGET_FEATURE].numpy()
            all_inputs.append(inp)
            all_targets.append(tgt)
            all_ids.append(f"{filename}:{record_idx}")
            record_idx += 1

        print(f"  [{file_idx + 1}/{len(tfrecord_files)}] {filename}: {record_idx} samples")

    inputs = np.stack(all_inputs, axis=0).astype(np.float32)
    targets = np.stack(all_targets, axis=0).astype(np.float32)
    print(f"Loaded {inputs.shape[0]} total samples. Shape: inputs={inputs.shape}, targets={targets.shape}")
    return inputs, targets, all_ids


print("Loading dataset...")
t0 = time.time()
all_inputs, all_targets, all_sample_ids = load_all_samples(DATA_ROOT)
print(f"Data loaded in {time.time() - t0:.1f}s")

# ──────────────────────────────────────────────
# DETERMINISTIC SPLITS (must match local pipeline)
# ──────────────────────────────────────────────
n_samples = len(all_sample_ids)
rng = np.random.default_rng(SEED)
indices = np.arange(n_samples, dtype=np.int64)
rng.shuffle(indices)

n_train = int(n_samples * TRAIN_FRACTION)
n_cal = int(n_samples * CALIBRATION_FRACTION)

train_idx = np.sort(indices[:n_train])
cal_idx = np.sort(indices[n_train:n_train + n_cal])
test_idx = np.sort(indices[n_train + n_cal:])

print(f"Splits: train={len(train_idx)}, calibration={len(cal_idx)}, test={len(test_idx)}")

# ──────────────────────────────────────────────
# PYTORCH DATASET
# ──────────────────────────────────────────────
class WildfireDataset(Dataset):
    """In-memory dataset for 64x64 wildfire maps."""

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        sample_ids: list[str],
        split_indices: np.ndarray,
        return_sample_id: bool = False,
    ) -> None:
        self.inputs = inputs[split_indices]   # (N, 64, 64, 12)
        self.targets = targets[split_indices]  # (N, 64, 64)
        self.ids = [sample_ids[i] for i in split_indices]
        self.return_sample_id = return_sample_id

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        inp = self.inputs[idx]   # (64, 64, 12)
        tgt = self.targets[idx]  # (64, 64)

        valid_mask = (tgt >= 0).astype(np.float32)
        tgt = np.clip(tgt, 0.0, 1.0)

        x = torch.from_numpy(np.transpose(inp, (2, 0, 1)))  # (12, 64, 64)
        y = torch.from_numpy(tgt[np.newaxis, ...])           # (1, 64, 64)
        m = torch.from_numpy(valid_mask[np.newaxis, ...])    # (1, 64, 64)

        if self.return_sample_id:
            return x, y, m, self.ids[idx]
        return x, y, m


train_ds = WildfireDataset(all_inputs, all_targets, all_sample_ids, train_idx)
val_ds = WildfireDataset(all_inputs, all_targets, all_sample_ids, cal_idx)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.head(d1))


model = TinyUNet(in_channels=len(INPUT_FEATURES), base_channels=BASE_CHANNELS).to(DEVICE)
param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count:,}")

# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.BCELoss(reduction="none")

best_val_loss = float("inf")
best_state = None
history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}

print(f"\nTraining for {EPOCHS} epochs on {DEVICE}...\n")

for epoch in range(1, EPOCHS + 1):
    t_start = time.time()

    # --- Train ---
    model.train()
    train_loss_sum = 0.0
    train_pixels = 0
    for x, y, mask in train_loader:
        x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        probs = model(x)
        pixel_loss = criterion(probs, y)
        masked = (pixel_loss * mask).sum()
        n_valid = mask.sum()
        (masked / n_valid).backward()
        optimizer.step()
        train_loss_sum += masked.item()
        train_pixels += int(n_valid.item())
    train_bce = train_loss_sum / train_pixels

    # --- Validate ---
    model.eval()
    val_loss_sum = 0.0
    val_pixels = 0
    with torch.no_grad():
        for x, y, mask in val_loader:
            x, y, mask = x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            probs = model(x)
            pixel_loss = criterion(probs, y)
            masked = (pixel_loss * mask).sum()
            n_valid = mask.sum()
            val_loss_sum += masked.item()
            val_pixels += int(n_valid.item())
    val_bce = val_loss_sum / val_pixels

    lr_now = scheduler.get_last_lr()[0]
    scheduler.step()
    elapsed = time.time() - t_start

    history["train_loss"].append(train_bce)
    history["val_loss"].append(val_bce)
    history["lr"].append(lr_now)

    marker = ""
    if val_bce < best_val_loss:
        best_val_loss = val_bce
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        marker = " ★"

    print(
        f"Epoch {epoch:03d}/{EPOCHS:03d} | "
        f"train: {train_bce:.6f} | val: {val_bce:.6f} | "
        f"lr: {lr_now:.2e} | {elapsed:.1f}s{marker}"
    )

print(f"\nBest val_bce: {best_val_loss:.6f}")

# Restore best model
if best_state is not None:
    model.load_state_dict(best_state)
    model.to(DEVICE)

# ──────────────────────────────────────────────
# EXPORT PROBABILITY HEATMAPS
# ──────────────────────────────────────────────
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

# Save checkpoint
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "seed": SEED,
        "input_features": INPUT_FEATURES,
        "target_name": TARGET_FEATURE,
        "base_channels": BASE_CHANNELS,
        "best_val_loss": best_val_loss,
        "epochs_trained": EPOCHS,
    },
    output_dir / "tiny_unet_checkpoint.pt",
)
print(f"Saved checkpoint to {output_dir / 'tiny_unet_checkpoint.pt'}")


def export_heatmaps(split_name: str, split_indices: np.ndarray) -> dict[str, int]:
    """Export probability heatmaps for a split."""
    ds = WildfireDataset(
        all_inputs, all_targets, all_sample_ids, split_indices, return_sample_id=True
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model.eval()
    prob_list, tgt_list, mask_list, id_list = [], [], [], []

    with torch.no_grad():
        for x, y, mask, sids in loader:
            x = x.to(DEVICE)
            probs = model(x).cpu().numpy()[:, 0, :, :]
            prob_list.append(probs.astype(np.float32))
            tgt_list.append(y.numpy()[:, 0, :, :].astype(np.uint8))
            mask_list.append(mask.numpy()[:, 0, :, :].astype(np.uint8))
            id_list.extend(list(sids))

    prob_maps = np.concatenate(prob_list, axis=0)
    tgt_maps = np.concatenate(tgt_list, axis=0)
    valid_masks = np.concatenate(mask_list, axis=0)
    max_len = max(len(s) for s in id_list)

    out_path = output_dir / f"{split_name}_probability_heatmaps.npz"
    np.savez_compressed(
        out_path,
        split=split_name,
        sample_ids=np.asarray(id_list, dtype=f"<U{max_len}"),
        targets=tgt_maps,
        valid_masks=valid_masks,
        probabilities=prob_maps,
    )
    print(f"Saved {split_name} heatmaps: {prob_maps.shape} → {out_path}")
    return {"samples": prob_maps.shape[0], "height": prob_maps.shape[1], "width": prob_maps.shape[2]}


export_shapes = {}
for name, idx in [("calibration", cal_idx), ("test", test_idx)]:
    export_shapes[name] = export_heatmaps(name, idx)

# Save training history and metadata
metadata = {
    "seed": SEED,
    "device": str(DEVICE),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "base_channels": BASE_CHANNELS,
    "scheduler": "CosineAnnealingLR",
    "best_val_loss": float(best_val_loss),
    "input_features": INPUT_FEATURES,
    "target_name": TARGET_FEATURE,
    "split_sizes": {
        "train": len(train_idx),
        "calibration": len(cal_idx),
        "test": len(test_idx),
    },
    "exports": export_shapes,
}
(output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

# Save training curves as JSON (for plotting later)
(output_dir / "training_history.json").write_text(json.dumps(history, indent=2))

print(f"\nAll artifacts saved to {output_dir.resolve()}")
print("Files to download:")
for f in sorted(output_dir.iterdir()):
    size_mb = f.stat().st_size / 1024 / 1024
    print(f"  {f.name} ({size_mb:.1f} MB)")
