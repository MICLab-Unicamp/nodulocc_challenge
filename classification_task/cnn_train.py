import argparse
import math
import random
import warnings
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

BEST_METRIC_CHOICES = ["roc_auc", "pr_auc", "balanced_accuracy", "f1"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DINOv2 lung nodule classifier")

    # ── Paths ──────────────────────────────────────────────────────────────
    p.add_argument("--data_root",  type=str, default="./data/nodulocc",
                   help="Root dir with classification_labels.csv + image folders")
    p.add_argument("--output_dir", type=str, default="./checkpoints/dinov2_nodule")

    # ── Split files (optional; falls back to auto-stratified if omitted) ───
    p.add_argument("--train_split", type=str, default=None,
                   help="Text file with training image filenames (one per line)")
    p.add_argument("--val_split",   type=str, default=None,
                   help="Text file with validation image filenames")
    p.add_argument("--test_split",  type=str, default=None,
                   help="Text file with test image filenames")

    # ── Auto-split ratios (only used when split files are not provided) ────
    p.add_argument("--train_size", type=float, default=0.80)
    p.add_argument("--val_size",   type=float, default=0.05)
    p.add_argument("--test_size",  type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)

    # ── Model ──────────────────────────────────────────────────────────────
    p.add_argument("--img_size", type=int, default=512,
                     help="Resize input images to this size (square). "
                            "If not set, uses the model's default resolution.")
    p.add_argument("--model_name", type=str,
                   default="convnext_base.dinov3_lvd1689m",
                   help="timm model name. "
                        "Use 'vit_base_patch16_dinov3.lvd1689m' for DINOv3.")
    p.add_argument("--pretrained",       action="store_true", default=True)
    p.add_argument("--freeze_backbone",  action="store_true",
                   help="Freeze backbone; train only the classification head")

    # ── Training ───────────────────────────────────────────────────────────
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--eval_every",   type=int,   default=1,
                   help="Run validation every N epochs")

    # ── Resume / eval-only ─────────────────────────────────────────────────
    p.add_argument(
        "--resume", type=str, default=None, metavar="CHECKPOINT",
        help=(
            "Path to a checkpoint (.pth) to resume training from. "
            "The file must have been saved with --save_epoch_checkpoints "
            "(i.e. contain 'epoch', 'model_state_dict', 'optimizer_state_dict', "
            "and 'scheduler_state_dict' keys). "
            "Training continues from the saved epoch; all other CLI flags "
            "override the saved args unless --resume_args is also set."
        ),
    )
    p.add_argument(
        "--resume_args", action="store_true",
        help=(
            "When resuming, restore ALL training hyper-parameters (lr, "
            "weight_decay, batch_size, epochs, best_metric, …) from the "
            "checkpoint rather than from the current CLI flags. "
            "Paths (--data_root, --output_dir, split files) are always "
            "taken from the CLI."
        ),
    )
    p.add_argument(
        "--eval_only", action="store_true",
        help=(
            "Skip training entirely. Load the model weights specified by "
            "--resume (required) and run the full validation + test "
            "evaluation pipeline. "
            "If --resume points to an epoch checkpoint (dict with "
            "'model_state_dict') its weights are loaded; if it points to a "
            "bare state-dict (best_model_*.pth) that is loaded directly."
        ),
    )

    # ── Mixed precision ────────────────────────────────────────────────────
    p.add_argument(
        "--precision", type=str, default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help=(
            "Floating-point precision for the forward pass.\n"
            "  fp32  – full precision (default, no autocast)\n"
            "  fp16  – half precision with GradScaler (fastest on Ampere+)\n"
            "  bf16  – bfloat16 (no GradScaler needed; stable on Ampere+)"
        ),
    )

    # ── Gradient accumulation ──────────────────────────────────────────────
    p.add_argument(
        "--grad_accum_steps", type=int, default=1,
        help=(
            "Accumulate gradients over this many micro-batches before "
            "stepping the optimiser. Effective batch size = "
            "batch_size × grad_accum_steps."
        ),
    )

    # ── Loss ───────────────────────────────────────────────────────────────
    p.add_argument("--use_focal_loss", action="store_true",
                   help="Use focal loss modulated by (1-p)^gamma")
    p.add_argument("--focal_gamma",  type=float, default=2.0)
    p.add_argument("--pos_weight",   type=float, default=20.0,
                   help="Weight for the positive (nodule) class")

    # ── Sampler ────────────────────────────────────────────────────────────
    p.add_argument("--use_balanced_sampler", action="store_true",
                   help="BalancedEpochSampler: ~50/50 per-epoch batches")

    # ── Checkpoint metric ──────────────────────────────────────────────────
    p.add_argument(
        "--best_metric", type=str, default="roc_auc",
        choices=BEST_METRIC_CHOICES,
        help=(
            "Validation metric used to select the best checkpoint "
            "(higher is always better for all choices).\n"
            "  roc_auc          – threshold-free ranking; good all-rounder\n"
            "  pr_auc           – focuses on the positive class; "
            "recommended for heavy imbalance (95/5)\n"
            "  balanced_accuracy – mean of sensitivity + specificity at thresh=0.5\n"
            "  f1               – harmonic mean of precision + recall at thresh=0.5"
        ),
    )

    p.add_argument(
        "--save_epoch_checkpoints", action="store_true",
        help=(
            "Save an additional checkpoint at the end of each epoch under "
            "output_dir/epochs/ (epoch_XXX.pth)."
        ),
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device: torch.device) -> dict:
    """
    Load a checkpoint file and return its contents as a dict.

    Handles two formats:
      • Full checkpoint (epoch dict): saved by --save_epoch_checkpoints.
        Keys: epoch, model_state_dict, optimizer_state_dict,
              scheduler_state_dict, best_metric_name, best_metric_value, args
      • Bare state-dict: saved as best_model_*.pth.
        Wrapped into {'model_state_dict': <state_dict>} for uniform access.
    """
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw                                  # full epoch checkpoint
    # Bare state-dict (best_model_*.pth)
    return {"model_state_dict": raw, "epoch": 0}


def apply_checkpoint_args(ckpt: dict, args: argparse.Namespace) -> None:
    """
    Overwrite training hyper-parameters in *args* with those stored in *ckpt*.
    Path-related arguments (data_root, output_dir, split files) are preserved
    from the CLI so the user can redirect data/output without re-training.
    """
    saved = ckpt.get("args")
    if not saved:
        print("[resume] No 'args' found in checkpoint; keeping CLI flags.")
        return

    PATH_KEYS = {
        "data_root", "output_dir",
        "train_split", "val_split", "test_split",
    }
    restored = []
    for k, v in saved.items():
        if k in PATH_KEYS:
            continue                               # always use CLI paths
        if hasattr(args, k) and getattr(args, k) != v:
            setattr(args, k, v)
            restored.append(f"{k}={v!r}")

    if restored:
        print(f"[resume] Restored from checkpoint: {', '.join(restored)}")


# ---------------------------------------------------------------------------
# LIDC preprocessing
# ---------------------------------------------------------------------------

def to_uint8_percentile_clip(arr: np.ndarray,
                              low_p: float = 0.5,
                              high_p: float = 95.5) -> np.ndarray:
    arr = np.asarray(arr)
    lo, hi = np.percentile(arr, [low_p, high_p])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    clipped = np.clip(arr, lo, hi).astype(np.float32)
    return np.round((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)


def preprocess_lidc_16bit_to_uint8(input_dir: Path, output_dir: Path,
                                    file_names: list[str],
                                    overwrite: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fn in tqdm(file_names, desc="Converting LIDC 16-bit → uint8"):
        out = output_dir / fn
        if out.exists() and not overwrite:
            continue
        img = Image.open(input_dir / fn)
        arr = np.array(img)
        result = arr if arr.dtype == np.uint8 else to_uint8_percentile_clip(arr)
        Image.fromarray(result, mode="L").save(out)


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

LABEL_MAP = {"No Finding": 0, "Nodule": 1}


def resolve_image_path(file_name: str,
                       nih_dir: Path,
                       lidc_u8_dir: Path,
                       lidc_16_dir: Path) -> str:
    for d in [nih_dir, lidc_u8_dir, lidc_16_dir]:
        p = d / file_name
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Cannot find image: {file_name}")


def load_split_file(path: str) -> list[str]:
    """Read a split text file; skip blank lines and # comments."""
    with open(path) as f:
        return [ln.strip() for ln in f
                if ln.strip() and not ln.strip().startswith("#")]


def build_records(df: pd.DataFrame,
                  filenames: list[str],
                  nih_dir: Path,
                  lidc_u8_dir: Path,
                  lidc_16_dir: Path) -> list[dict]:
    """Return [{path, label}, …] for the given filenames."""
    fn_to_label = dict(zip(df["file_name"].astype(str), df["label_id"]))
    records, missing_label, missing_file = [], [], []
    for fn in filenames:
        if fn not in fn_to_label:
            missing_label.append(fn)
            continue
        try:
            path = resolve_image_path(fn, nih_dir, lidc_u8_dir, lidc_16_dir)
            records.append({"path": path, "label": int(fn_to_label[fn])})
        except FileNotFoundError:
            missing_file.append(fn)
    if missing_label:
        print(f"  [warn] {len(missing_label)} filenames had no label in CSV")
    if missing_file:
        print(f"  [warn] {len(missing_file)} filenames not found on disk")
    return records


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class NoduleDataset(Dataset):
    def __init__(self, records: list[dict], transform: A.Compose):
        self.records   = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec    = self.records[idx]
        img    = np.array(Image.open(rec["path"]).convert("RGB"))
        tensor = self.transform(image=img)["image"]   # float32 CHW
        return tensor, rec["label"]


# ---------------------------------------------------------------------------
# Albumentations pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        # ── Spatial ──────────────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10,
                           rotate_limit=10,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.ElasticTransform(alpha=40, sigma=5, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        # ── Intensity / texture ───────────────────────────────────────────
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                   contrast_limit=0.15, p=0.6),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4),
                        hole_height_range=(8, 32),
                        hole_width_range=(8, 32),
                        fill=0, p=0.2),
        # ── Normalise + tensor ────────────────────────────────────────────
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_test_transforms(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ---------------------------------------------------------------------------
# Balanced epoch sampler
# ---------------------------------------------------------------------------

class BalancedEpochSampler(torch.utils.data.Sampler):
    """
    Each epoch: all positives + an equal-size rotating chunk of negatives,
    giving ~50/50 batches. The negative pool cycles so every negative is
    eventually seen over multiple epochs.
    """

    def __init__(self, labels: list[int], seed: int = 0):
        self.seed    = seed
        arr          = np.array(labels)
        self.pos_idx = np.where(arr == 1)[0].tolist()
        self.neg_idx = np.where(arr == 0)[0].tolist()
        if not self.pos_idx or not self.neg_idx:
            raise ValueError("Need ≥1 positive and ≥1 negative sample.")
        self.n_pos = len(self.pos_idx)
        self.n_neg = len(self.neg_idx)
        self._epoch = 0

    def _negatives_for_epoch(self, epoch: int) -> list[int]:
        need, out    = self.n_pos, []
        global_start = epoch * self.n_pos
        cycle        = global_start // self.n_neg
        offset       = global_start % self.n_neg
        while need > 0:
            perm = np.random.default_rng(self.seed + cycle).permutation(
                self.neg_idx).tolist()
            take = min(need, self.n_neg - offset)
            out.extend(perm[offset: offset + take])
            need -= take; cycle += 1; offset = 0
        return out

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        indices = self.pos_idx + self._negatives_for_epoch(self._epoch)
        np.random.default_rng(self.seed + 10_000 + self._epoch).shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.n_pos * 2


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class FocalCrossEntropyLoss(nn.Module):
    """Focal loss with optional per-class weights. gamma=0 → weighted CE."""

    def __init__(self, gamma: float = 2.0,
                 weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        w  = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, targets, weight=w, reduction="none")
        pt = (torch.softmax(logits, dim=-1)
              .gather(1, targets.view(-1, 1)).squeeze(1)
              .clamp(1e-6, 1.0))
        return (((1.0 - pt) ** self.gamma) * ce).mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DINOv2Classifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 2,
                 pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained,
                                          num_classes=0)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        feat_dim  = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _safe(a, b):
    return float(a) / float(b) if b else 0.0


def compute_full_metrics(p_pos: np.ndarray, refs: np.ndarray,
                         threshold: float = 0.5) -> dict:
    preds = (p_pos >= threshold).astype(int)
    tp = int(((preds == 1) & (refs == 1)).sum())
    tn = int(((preds == 0) & (refs == 0)).sum())
    fp = int(((preds == 1) & (refs == 0)).sum())
    fn = int(((preds == 0) & (refs == 1)).sum())

    sensitivity  = _safe(tp, tp + fn)
    specificity  = _safe(tn, tn + fp)
    precision    = _safe(tp, tp + fp)
    f1           = _safe(2 * precision * sensitivity, precision + sensitivity)
    accuracy     = _safe(tp + tn, tp + tn + fp + fn)
    balanced_acc = 0.5 * (sensitivity + specificity)

    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc   = (tp * tn - fp * fn) / math.sqrt(denom) if denom > 0 else 0.0

    pr_auc = roc_auc = float("nan")
    if len(np.unique(refs)) == 2:
        pr_auc  = average_precision_score(refs, p_pos)
        roc_auc = roc_auc_score(refs, p_pos)

    brier = float(np.mean((p_pos - refs) ** 2))

    return dict(
        threshold=threshold,
        accuracy=accuracy, balanced_accuracy=balanced_acc,
        sensitivity=sensitivity, specificity=specificity,
        precision=precision, f1=f1, mcc=float(mcc),
        roc_auc=roc_auc, pr_auc=pr_auc, brier=brier,
        tp=tp, tn=tn, fp=fp, fn=fn,
    )


def sensitivity_at_fixed_specificity(p_pos: np.ndarray, refs: np.ndarray,
                                     target_spec: float = 0.99):
    fpr, tpr, thresholds = roc_curve(refs, p_pos)
    spec  = 1.0 - fpr
    valid = np.where(spec >= target_spec)[0]
    idx   = valid[np.argmax(tpr[valid])] if len(valid) else np.argmax(spec)
    thr   = float(thresholds[idx]) if idx < len(thresholds) else 1.0
    return float(tpr[idx]), thr


def tune_threshold_max_f1(p_pos: np.ndarray, refs: np.ndarray) -> float:
    prec_arr, rec_arr, thresholds = precision_recall_curve(refs, p_pos)
    f1_arr = np.zeros_like(prec_arr[:-1])
    den    = prec_arr[:-1] + rec_arr[:-1]
    ok     = den > 0
    f1_arr[ok] = 2 * prec_arr[:-1][ok] * rec_arr[:-1][ok] / den[ok]
    return float(thresholds[np.argmax(f1_arr)])


def print_metrics_table(metrics: dict, title: str = "") -> None:
    if title:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    for k, v in metrics.items():
        print(f"  {k:<25s}: {v:.4f}" if isinstance(v, float)
              else f"  {k:<25s}: {v}")


def format_best_checkpoint_name(metric_name: str, metric_value: float) -> str:
    return f"best_model_{metric_name}_{metric_value:.4f}.pth"


# ---------------------------------------------------------------------------
# TensorBoard helper
# ---------------------------------------------------------------------------

SCALAR_KEYS = [
    "accuracy", "balanced_accuracy", "sensitivity", "specificity",
    "precision", "f1", "mcc", "roc_auc", "pr_auc", "brier", "fp", "fn", "tp", "tn",
]


def log_metrics(writer: SummaryWriter, metrics: dict,
                prefix: str, step: int) -> None:
    for k in SCALAR_KEYS:
        v = metrics.get(k, float("nan"))
        if not math.isnan(v):
            writer.add_scalar(f"{prefix}/{k}", v, step)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def score_dataset(model: nn.Module, loader: DataLoader,
                  device: torch.device,
                  autocast_ctx) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_p, all_r = [], []
    for imgs, labels in tqdm(loader, desc="Scoring", leave=False):
        with autocast_ctx:
            probs = torch.softmax(model(imgs.to(device)).float(), dim=-1)
        all_p.extend(probs[:, 1].cpu().numpy())
        all_r.extend(labels.numpy())
    return np.array(all_p), np.array(all_r)


@torch.inference_mode()
def compute_dataset_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    autocast_ctx,
) -> float:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    for imgs, labels in tqdm(loader, desc="Val loss", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast_ctx:
            loss = criterion(model(imgs), labels)
        total_loss += loss.item() * imgs.size(0)
        n_samples += imgs.size(0)
    return total_loss / max(n_samples, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler | None,
    autocast_ctx,
    grad_accum_steps: int,
) -> float:
    model.train()
    total_loss   = 0.0
    n_samples    = 0
    n_batches    = len(loader)
    optimizer.zero_grad()

    for batch_idx, (imgs, labels) in enumerate(
            tqdm(loader, desc=f"Train epoch {epoch}", leave=False)):

        imgs, labels = imgs.to(device), labels.to(device)
        is_last_batch = (batch_idx == n_batches - 1)

        # ── Forward under autocast ────────────────────────────────────────
        with autocast_ctx:
            loss = criterion(model(imgs), labels)

        # Scale loss by accumulation steps so gradients are averaged
        # across the full effective batch, not just the micro-batch.
        loss_scaled = loss / grad_accum_steps

        # ── Backward ──────────────────────────────────────────────────────
        if scaler is not None:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        total_loss += loss.item() * imgs.size(0)
        n_samples  += imgs.size(0)

        # ── Optimiser step every grad_accum_steps (or at epoch end) ──────
        should_step = (
            (batch_idx + 1) % grad_accum_steps == 0
            or is_last_batch
        )
        if should_step:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

    return total_loss / n_samples


# ---------------------------------------------------------------------------
# Shared evaluation pipeline (used by both training loop and --eval_only)
# ---------------------------------------------------------------------------

def run_evaluation(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    autocast_ctx,
    writer: SummaryWriter,
    final_epoch: int,
    args: argparse.Namespace,
) -> None:
    """
    Run the full threshold-tuning + test evaluation pipeline and print
    the summary table. Mirrors the post-training evaluation in main().
    """

    # ── Threshold tuning on validation set ─────────────────────────────────
    print("\n─── Threshold tuning on validation set ───")
    val_p, val_r   = score_dataset(model, val_loader, device, autocast_ctx)
    thresh_f1      = tune_threshold_max_f1(val_p, val_r)
    _, thresh_s99  = sensitivity_at_fixed_specificity(val_p, val_r, 0.99)
    _, thresh_s995 = sensitivity_at_fixed_specificity(val_p, val_r, 0.995)

    print(f"  max-F1 threshold    : {thresh_f1:.4f}")
    print(f"  ≥99.0% spec thresh  : {thresh_s99:.4f}")
    print(f"  ≥99.5% spec thresh  : {thresh_s995:.4f}")

    for name, t in [("Val @ max-F1",        thresh_f1),
                    ("Val @ 99.0% spec",     thresh_s99),
                    ("Val @ 99.5% spec",     thresh_s995),
                    ("Val @ 0.5 (default)",  0.5)]:
        print_metrics_table(compute_full_metrics(val_p, val_r, t), name)

    # ── Test set evaluation ─────────────────────────────────────────────────
    print("\n─── Test set evaluation ───")
    test_p, test_r = score_dataset(model, test_loader, device, autocast_ctx)
    n_pos = int(test_r.sum()); n_neg = len(test_r) - n_pos
    print(f"Test: {len(test_r)} samples | "
          f"pos={n_pos} ({100*n_pos/len(test_r):.1f}%) | neg={n_neg}")

    test_configs = [
        ("0.5 (default)",                    0.5),
        ("max-F1 (tuned on val)",             thresh_f1),
        ("≥99.0% specificity (tuned on val)", thresh_s99),
        ("≥99.5% specificity (tuned on val)", thresh_s995),
    ]
    for label, t in test_configs:
        m = compute_full_metrics(test_p, test_r, t)
        print_metrics_table(m, f"Test — {label}")
        log_metrics(writer, m, prefix=f"test/{label[:24]}", step=final_epoch)

    # Sensitivity @ fixed specificity
    print(f"\n{'=' * 60}")
    print("Sensitivity @ fixed specificity (test set)")
    print("=" * 60)
    for target in [0.95, 0.99, 0.995]:
        sens, thr = sensitivity_at_fixed_specificity(test_p, test_r, target)
        print(f"  Spec ≥ {target:.1%}:  "
              f"sensitivity={sens:.4f}   threshold={thr:.4f}")

    # ── Summary table ───────────────────────────────────────────────────────
    summary_rows = [
        ("Test — 0.5 (default)",
         compute_full_metrics(test_p, test_r, 0.5)),
        (f"Test — max-F1 @ {thresh_f1:.3f}",
         compute_full_metrics(test_p, test_r, thresh_f1)),
        (f"Test — spec≥99% @ {thresh_s99:.3f}",
         compute_full_metrics(test_p, test_r, thresh_s99)),
    ]
    cols   = ["sensitivity", "specificity", "precision",
              "f1", "pr_auc", "roc_auc", "balanced_accuracy"]
    cw, nw = 12, 44
    sep    = "=" * (nw + len(cols) * cw)
    print(f"\n{sep}")
    print(f"{'Configuration':<{nw}}" + "".join(f"{c:>{cw}}" for c in cols))
    print("-" * (nw + len(cols) * cw))
    for label, m in summary_rows:
        vals = "".join(f"{m.get(k, float('nan')):>{cw}.4f}" for k in cols)
        print(f"{label:<{nw}}{vals}")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Validation: --eval_only requires --resume ──────────────────────────
    if args.eval_only and args.resume is None:
        raise ValueError("--eval_only requires --resume <checkpoint_path>")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device           : {device}")

    # ── Load checkpoint early so --resume_args can override CLI flags ──────
    ckpt = None
    start_epoch = 1
    if args.resume is not None:
        print(f"\nLoading checkpoint: {args.resume}")
        ckpt = load_checkpoint(args.resume, device)
        if args.resume_args:
            apply_checkpoint_args(ckpt, args)
        resumed_epoch = ckpt.get("epoch", 0)
        if not args.eval_only:
            start_epoch = resumed_epoch + 1
            print(f"[resume] Resuming from epoch {resumed_epoch} "
                  f"→ will train epochs {start_epoch}–{args.epochs}")
            if start_epoch > args.epochs:
                raise ValueError(
                    f"Checkpoint epoch ({resumed_epoch}) >= --epochs "
                    f"({args.epochs}). Increase --epochs or pick an earlier "
                    f"checkpoint."
                )

    print(f"Checkpoint metric: {args.best_metric}")
    print(f"Precision        : {args.precision}")
    if not args.eval_only:
        print(f"Grad accum steps : {args.grad_accum_steps}  "
              f"(effective batch = {args.batch_size * args.grad_accum_steps})")

    # ── Mixed-precision setup ──────────────────────────────────────────────
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    if args.precision == "fp16":
        autocast_ctx = torch.amp.autocast(device_type=amp_device,
                                          dtype=torch.float16)
        scaler       = torch.amp.GradScaler(device=amp_device)
        print("AMP: float16 + GradScaler enabled")
    elif args.precision == "bf16":
        autocast_ctx = torch.amp.autocast(device_type=amp_device,
                                          dtype=torch.bfloat16)
        scaler       = None
        print("AMP: bfloat16 enabled (no GradScaler needed)")
    else:
        autocast_ctx = torch.amp.autocast(device_type=amp_device,
                                          enabled=False)
        scaler       = None
        print("AMP: disabled (fp32)")

    # ── Paths ──────────────────────────────────────────────────────────────
    data_root   = Path(args.data_root)
    nih_dir     = data_root / "nih_filtered_images"
    lidc_16_dir = data_root / "lidc_png_16_bit"
    lidc_u8_dir = data_root / "lidc_png_uint8"
    cls_csv     = data_root / "classification_labels.csv"
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── TensorBoard ────────────────────────────────────────────────────────
    tb_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard logs : {tb_dir}")
    print(f"  → tensorboard --logdir {tb_dir}")

    # ── LIDC preprocessing ─────────────────────────────────────────────────
    if lidc_16_dir.exists():
        df_raw   = pd.read_csv(cls_csv)
        lidc_fns = [fn for fn in df_raw["file_name"].dropna().astype(str)
                    if (lidc_16_dir / fn).exists()]
        if lidc_fns:
            print(f"\nPre-processing {len(lidc_fns)} LIDC 16-bit images…")
            preprocess_lidc_16bit_to_uint8(lidc_16_dir, lidc_u8_dir, lidc_fns)

    # ── CSV ────────────────────────────────────────────────────────────────
    df = pd.read_csv(cls_csv)
    df = df[df["label"].isin(LABEL_MAP)].copy()
    df["label_id"]  = df["label"].map(LABEL_MAP).astype(int)
    df["file_name"] = df["file_name"].astype(str)

    # ── Build splits ───────────────────────────────────────────────────────
    use_split_files = all(x is not None for x in
                          [args.train_split, args.val_split, args.test_split])

    if use_split_files:
        print("\nLoading splits from text files:")
        train_fns = load_split_file(args.train_split)
        val_fns   = load_split_file(args.val_split)
        test_fns  = load_split_file(args.test_split)
        print(f"  train={len(train_fns)}  val={len(val_fns)}  "
              f"test={len(test_fns)}")
        train_records = build_records(df, train_fns, nih_dir, lidc_u8_dir, lidc_16_dir)
        val_records   = build_records(df, val_fns,   nih_dir, lidc_u8_dir, lidc_16_dir)
        test_records  = build_records(df, test_fns,  nih_dir, lidc_u8_dir, lidc_16_dir)

    else:
        print("\nNo split files provided — using automatic stratified splits.")
        records_all, labels_all = [], []
        for _, row in df.iterrows():
            try:
                path = resolve_image_path(
                    row["file_name"], nih_dir, lidc_u8_dir, lidc_16_dir)
                records_all.append({"path": path, "label": int(row["label_id"])})
                labels_all.append(int(row["label_id"]))
            except FileNotFoundError:
                pass

        indices  = list(range(len(records_all)))
        idx_tv, idx_test = train_test_split(
            indices, test_size=args.test_size,
            stratify=labels_all, random_state=args.seed)
        labels_tv = [labels_all[i] for i in idx_tv]
        val_frac  = args.val_size / (args.train_size + args.val_size)
        idx_train, idx_val = train_test_split(
            idx_tv, test_size=val_frac,
            stratify=labels_tv, random_state=args.seed)

        train_records = [records_all[i] for i in idx_train]
        val_records   = [records_all[i] for i in idx_val]
        test_records  = [records_all[i] for i in idx_test]

    def split_summary(name: str, records: list[dict]) -> None:
        n     = len(records)
        n_pos = sum(r["label"] for r in records)
        print(f"  {name:6s}: {n:6d} samples  "
              f"pos={n_pos} ({100*n_pos/max(n,1):.1f}%)  "
              f"neg={n - n_pos}")

    print("\nSplit summary:")
    split_summary("train", train_records)
    split_summary("val",   val_records)
    split_summary("test",  test_records)

    # ── Model + image size ──────────────────────────────────────────────────
    print(f"\nLoading model: {args.model_name}")
    _tmp     = timm.create_model(args.model_name, pretrained=False)

    if args.img_size:
        img_size = args.img_size
        print(f"  Using user-specified image size: {img_size}px")
    else:
        img_size = timm.data.resolve_model_data_config(_tmp)["input_size"][-1]
    del _tmp
    print(f"  Input image size: {img_size}px")

    # When eval_only or resuming we skip downloading pretrained weights
    # because the checkpoint already contains them.
    load_pretrained = args.pretrained and (ckpt is None)
    model = DINOv2Classifier(
        args.model_name,
        pretrained=load_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    # ── Restore model weights from checkpoint ───────────────────────────────
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[resume] Model weights restored from checkpoint.")

    # ── Transforms ──────────────────────────────────────────────────────────
    train_tf = get_train_transforms(img_size)
    eval_tf  = get_val_test_transforms(img_size)

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = NoduleDataset(train_records, train_tf)
    val_ds   = NoduleDataset(val_records,   eval_tf)
    test_ds  = NoduleDataset(test_records,  eval_tf)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    loader_kw = dict(num_workers=args.num_workers, pin_memory=True)

    if args.use_balanced_sampler:
        train_labels = [r["label"] for r in train_records]
        bal_sampler  = BalancedEpochSampler(train_labels, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=bal_sampler, **loader_kw)
        print("Sampler : BalancedEpochSampler (~50/50 per epoch)")
    else:
        bal_sampler  = None
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, **loader_kw)
        print("Sampler : default shuffle")

    val_loader  = DataLoader(val_ds,  batch_size=args.batch_size,
                             shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, **loader_kw)

    # ── Loss ─────────────────────────────────────────────────────────────────
    class_weight = torch.tensor([1.0, args.pos_weight])
    if args.use_focal_loss:
        criterion = FocalCrossEntropyLoss(gamma=args.focal_gamma,
                                          weight=class_weight)
        print(f"Loss    : FocalCrossEntropy  gamma={args.focal_gamma}  "
              f"pos_weight={args.pos_weight}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight.to(device))
        print(f"Loss    : CrossEntropy  pos_weight={args.pos_weight}")

    # ── Eval-only path ────────────────────────────────────────────────────────
    if args.eval_only:
        print(f"\n{'─'*70}")
        print("Mode: EVAL ONLY (no training)")
        print(f"{'─'*70}")
        run_evaluation(
            model, val_loader, test_loader, criterion,
            device, autocast_ctx, writer,
            final_epoch=ckpt.get("epoch", 0),
            args=args,
        )
        writer.flush()
        writer.close()
        print("\nDone.")
        return

    # ── Optimiser / scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Restore optimiser + scheduler state when resuming ──────────────────
    if ckpt is not None:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[resume] Optimizer state restored.")
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print("[resume] Scheduler state restored.")

    # ── Training loop ────────────────────────────────────────────────────────
    best_val  = ckpt.get("best_metric_value", -1.0) if ckpt else -1.0
    best_ckpt = None
    epoch_ckpt_dir = output_dir / "epochs"
    if args.save_epoch_checkpoints:
        epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Epoch checkpoints: enabled ({epoch_ckpt_dir})")

    print(f"\n{'─'*70}")
    print(f"Training epochs {start_epoch}–{args.epochs}  |  "
          f"checkpoint on: val/{args.best_metric}  (higher = better)")
    if ckpt:
        print(f"Resuming with best val/{args.best_metric} = {best_val:.4f} "
              f"(from checkpoint epoch {ckpt.get('epoch', '?')})")
    print(f"{'─'*70}")

    for epoch in range(start_epoch, args.epochs + 1):
        if bal_sampler is not None:
            bal_sampler.set_epoch(epoch - 1)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            scaler=scaler,
            autocast_ctx=autocast_ctx,
            grad_accum_steps=args.grad_accum_steps,
        )
        scheduler.step()

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch % args.eval_every == 0:
            val_p, val_r = score_dataset(model, val_loader, device, autocast_ctx)
            val_loss = compute_dataset_loss(
                model, val_loader, criterion, device, autocast_ctx)
            val_m        = compute_full_metrics(val_p, val_r, threshold=0.5)

            log_metrics(writer, val_m, prefix="val", step=epoch)
            writer.add_scalar("val/loss", val_loss, epoch)

            cur = val_m.get(args.best_metric, float("nan"))
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"roc_auc={val_m['roc_auc']:.4f} | "
                f"pr_auc={val_m['pr_auc']:.4f} | "
                f"bal_acc={val_m['balanced_accuracy']:.4f} | "
                f"f1={val_m['f1']:.4f} | "
                f"sens={val_m['sensitivity']:.4f} | "
                f"spec={val_m['specificity']:.4f}"
                + (f"  ← best_metric={cur:.4f}" if not math.isnan(cur) else "")
            )

            if not math.isnan(cur) and cur > best_val:
                prev_best_ckpt = best_ckpt
                best_val = cur
                best_ckpt = output_dir / format_best_checkpoint_name(
                    args.best_metric, cur)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_metric_name": args.best_metric,
                    "best_metric_value": best_val,
                    "args": vars(args),
                }, best_ckpt)
                if prev_best_ckpt is not None and prev_best_ckpt != best_ckpt:
                    prev_best_ckpt.unlink(missing_ok=True)
                print(f"  ✓ New best val/{args.best_metric}={cur:.4f} "
                      f"— checkpoint saved to {best_ckpt.name}.")
        else:
            print(f"Epoch {epoch:03d}/{args.epochs:03d} | "
                  f"loss={train_loss:.4f}")

        if args.save_epoch_checkpoints:
            epoch_ckpt_path = epoch_ckpt_dir / f"epoch_{epoch:03d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_metric_name": args.best_metric,
                    "best_metric_value": best_val,
                    "args": vars(args),
                },
                epoch_ckpt_path,
            )

    print(f"\nTraining complete. "
          f"Best val/{args.best_metric} = {best_val:.4f}")

    if best_ckpt is None:
        best_ckpt = output_dir / format_best_checkpoint_name(
            args.best_metric, best_val)
        torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_metric_name": args.best_metric,
                    "best_metric_value": best_val,
                    "args": vars(args),
                }, best_ckpt)
        print(f"No validation checkpoint was selected during training; "
              f"saved final model to {best_ckpt.name}.")

    # ── Load best checkpoint ─────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    print(f"Loaded checkpoint: {best_ckpt}")

    # ── Post-training evaluation ─────────────────────────────────────────────
    run_evaluation(
        model, val_loader, test_loader, criterion,
        device, autocast_ctx, writer,
        final_epoch=args.epochs,
        args=args,
    )

    writer.flush()
    writer.close()
    print("\nDone.")


if __name__ == "__main__":
    main()