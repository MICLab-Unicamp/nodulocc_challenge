# %% [markdown]
# # Fine-tune MedGemma on NoduLOCC (Chest X-ray) — Classification only
# 
# This notebook adapts the MedGemma Hugging Face SFT + QLoRA workflow to the
# `./nodulocc/` dataset.
# 
# **Focus**: binary classification (Healthy vs Nodule).
# **Deferred**: localization (keypoints) for later.
# 
# Dataset layout (expected):
# ```
# ./nodulocc/
#   classification_labels.csv
#   localization_labels.csv
#   nih_filtered_images/        (png)
#   lidc_png_16_bit/            (16-bit png)
# ```
# 
# Important preprocessing:
# - LIDC images are 16-bit PNGs but do not use the full range.
# - We convert them to uint8 PNG by clipping per-image values to the
#   [0.5%, 95.5%] percentiles, then rescaling to [0, 255].

# %% [markdown]
# ## Setup (HF auth + installs)

# %%
# ! pip install --upgrade --quiet bitsandbytes datasets scikit-learn pandas peft pillow tensorboard tensorboardX tqdm transformers trl dotenv

# %%
from dotenv import load_dotenv
load_dotenv()  # reads .env and sets os.environ automatically
# %% [markdown]
# ## Paths

# %%
from pathlib import Path

DATA_ROOT = Path("./data/nodulocc")

CLASSIFICATION_CSV = DATA_ROOT / "classification_labels.csv"
LOCALIZATION_CSV = DATA_ROOT / "localization_labels.csv"

NIH_DIR = DATA_ROOT / "nih_filtered_images"
LIDC_16_DIR = DATA_ROOT / "lidc_png_16_bit"

# We will write converted uint8 PNGs here:
LIDC_8_DIR = DATA_ROOT / "lidc_png_uint8"

assert CLASSIFICATION_CSV.exists(), f"Missing: {CLASSIFICATION_CSV}"
assert LOCALIZATION_CSV.exists(), f"Missing: {LOCALIZATION_CSV}"
assert NIH_DIR.exists(), f"Missing: {NIH_DIR}"
assert LIDC_16_DIR.exists(), f"Missing: {LIDC_16_DIR}"

# %% [markdown]
# ## Preprocess LIDC 16-bit PNGs -> uint8 PNGs
# 
# We convert only the LIDC files referenced by the CSVs (faster than converting
# the entire folder). You can switch to preprocessing all LIDC images if needed.

# %%
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm


def to_uint8_percentile_clip(
    arr: np.ndarray,
    low_p: float = 0.5,
    high_p: float = 95.5,
) -> np.ndarray:
    """
    Per-image clip to [low_p, high_p] percentiles, then rescale to [0, 255] uint8.
    """
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale array, got shape={arr.shape}")

    lo, hi = np.percentile(arr, [low_p, high_p])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)

    clipped = np.clip(arr, lo, hi).astype(np.float32)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    return np.round(scaled).astype(np.uint8)


def collect_lidc_filenames_from_csvs() -> list[str]:
    """
    Returns file_names that exist in lidc_png_16_bit and are referenced by the CSVs.
    """
    df_cls = pd.read_csv(CLASSIFICATION_CSV)
    df_loc = pd.read_csv(LOCALIZATION_CSV)

    names = set()
    for col in ["file_name"]:
        if col in df_cls.columns:
            names.update(df_cls[col].dropna().astype(str).tolist())
        if col in df_loc.columns:
            names.update(df_loc[col].dropna().astype(str).tolist())

    lidc_names = [n for n in names if (LIDC_16_DIR / n).exists()]
    return sorted(lidc_names)


def preprocess_lidc_16bit_to_uint8(
    input_dir: Path,
    output_dir: Path,
    file_names: list[str] | None = None,
    low_p: float = 0.5,
    high_p: float = 95.5,
    overwrite: bool = False,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_names is None:
        paths = sorted(input_dir.glob("*.png"))
    else:
        paths = [input_dir / fn for fn in file_names]

    for in_path in tqdm(paths, desc="Converting LIDC 16-bit -> uint8"):
        out_path = output_dir / in_path.name
        if out_path.exists() and not overwrite:
            continue

        img = Image.open(in_path)
        arr = np.array(img)

        if arr.dtype == np.uint8:
            Image.fromarray(arr, mode="L").save(out_path)
            continue

        arr_u8 = to_uint8_percentile_clip(arr, low_p=low_p, high_p=high_p)
        Image.fromarray(arr_u8, mode="L").save(out_path)


LIDC_FILE_NAMES = collect_lidc_filenames_from_csvs()
print(f"LIDC files referenced by CSVs: {len(LIDC_FILE_NAMES)}")

preprocess_lidc_16bit_to_uint8(
    input_dir=LIDC_16_DIR,
    output_dir=LIDC_8_DIR,
    file_names=LIDC_FILE_NAMES,
    low_p=0.5,
    high_p=95.5,
    overwrite=False,
)

# %% [markdown]
# ## Load classification dataset (Healthy vs Nodule)
# 
# We will build a Hugging Face `Dataset` with columns:
# - `image`: image file path (loaded as PIL by `datasets.Image`)
# - `label_id`: 0 (healthy) or 1 (nodule)
# - `file_name`: original filename

# %%
from datasets import ClassLabel, Dataset, Image as HFImage

df_cls = pd.read_csv(CLASSIFICATION_CSV)
df_loc = pd.read_csv(LOCALIZATION_CSV)  # loaded for later use; not used now

LABEL_MAP = {
    "No Finding": 0,  # healthy
    "Nodule": 1,  # nodule present
}


def resolve_image_path(file_name: str) -> str:
    """
    Resolve filename to NIH path or LIDC uint8 path.
    """
    file_name = str(file_name)

    p_nih = NIH_DIR / file_name
    if p_nih.exists():
        return str(p_nih)

    p_lidc_u8 = LIDC_8_DIR / file_name
    if p_lidc_u8.exists():
        return str(p_lidc_u8)

    # Fallback to original 16-bit if needed (not recommended for training)
    p_lidc_16 = LIDC_16_DIR / file_name
    if p_lidc_16.exists():
        return str(p_lidc_16)

    raise FileNotFoundError(
        f"Could not resolve image for file_name={file_name}. "
        f"Checked: {p_nih}, {p_lidc_u8}, {p_lidc_16}"
    )


df_cls = df_cls[df_cls["label"].isin(LABEL_MAP)].copy()
df_cls["label_id"] = df_cls["label"].map(LABEL_MAP).astype(int)
df_cls["image"] = df_cls["file_name"].apply(resolve_image_path)

# filter out all LIDC image rows
lidc_mask = df_cls["image"].str.contains("lidc", case=False)
print(f"Total samples: {len(df_cls)}")
print(f"LIDC samples: {lidc_mask.sum()}")

df_cls = df_cls[~lidc_mask].reset_index(drop=True)  # keep only non-LIDC samples for now

df_cls = df_cls[["image", "label_id", "file_name"]].reset_index(drop=True)

ds = Dataset.from_pandas(df_cls, preserve_index=False)
ds = ds.cast_column("image", HFImage())
ds = ds.cast_column("label_id", ClassLabel(names=["healthy", "nodule"]))

# %%
# from hugging face dataset ds, get class distribution
from collections import Counter

def print_class_distribution(split_name: str, split_ds: Dataset):
    labels = split_ds["label_id"]
    class_counts = Counter(labels)

    print(f"\n{split_name} set class distribution:")
    for class_id, count in class_counts.items():
        class_name = split_ds.features["label_id"].int2str(class_id)
        percentage = (count / len(split_ds)) * 100
        print(f"Class '{class_name}' (ID: {class_id}): {count} samples ({percentage:.2f}%)")
    
print_class_distribution("Full", ds)

# %% [markdown]
# ## Convert to MedGemma chat format (multimodal SFT)

# %%
from typing import Any

CLASS_OPTIONS = ["A", "B"]

PROMPT = (
  "You are a radiology assistant. Determine whether this frontal chest X-ray "
  "shows a lung nodule.\n"
  "Answer with exactly one letter:\n"
  "A = Healthy (no lung nodule)\n"
  "B = Nodule present"
)


def format_example(example: dict[str, Any]) -> dict[str, Any]:
    label_text = CLASS_OPTIONS[int(example["label_id"])]

    example["messages"] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": label_text}],
        },
    ]
    return example


ds = ds.map(format_example)

# %% [markdown]
# ## Train/validation split (stratified)

# %%
from datasets import Dataset

SPLIT_SEED = 42

TRAIN_SIZE = 0.80
VAL_SIZE = 0.05
TEST_SIZE = 0.15

assert abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) < 1e-6

ds_temp = ds.train_test_split(
    test_size=TEST_SIZE,
    seed=SPLIT_SEED,
    stratify_by_column="label_id",
)

train_val_ds = ds_temp["train"]
test_ds = ds_temp["test"]

val_ratio_relative = VAL_SIZE / (TRAIN_SIZE + VAL_SIZE)

ds_final = train_val_ds.train_test_split(
    test_size=val_ratio_relative,
    seed=SPLIT_SEED,
    stratify_by_column="label_id",
)

train_ds = ds_final["train"]
val_ds = ds_final["test"]

print(f"Train size: {len(train_ds)}")
print(f"Validation size: {len(val_ds)}")
print(f"Test size: {len(test_ds)}")

# %% [markdown]
# ## Load MedGemma + QLoRA config

# %%
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

model_id = "google/medgemma-1.5-4b-it"

if torch.cuda.get_device_capability()[0] < 8:
    raise ValueError(
        "GPU does not support bfloat16 (need Ampere/Hopper). Use A100/H100, etc."
    )

model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
)

model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
processor = AutoProcessor.from_pretrained(model_id)

# Right padding for training
processor.tokenizer.padding_side = "right"

# %% [markdown]
# ## LoRA (PEFT) config

# %%
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    # target_modules=['q_proj', 'v_proj'],
	# target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
    # modules_to_save=[
    #     "lm_head",
    #     "embed_tokens",
    # ],
)

# %% [markdown]
# ## Data collator (image + text) with robust masking

# %%
import torch
from typing import Any

POS_LOSS_WEIGHT = 20.0
NEG_LOSS_WEIGHT = 1.0

_ASSISTANT_TURN_PREFIX = "<start_of_turn>model\n"
_ASSISTANT_TOKENS = processor.tokenizer.encode(
    _ASSISTANT_TURN_PREFIX, add_special_tokens=False
)

def _find_assistant_start(input_ids: torch.Tensor, marker: list[int]) -> int:
    seq = input_ids.tolist()
    m = len(marker)
    for i in range(len(seq) - m, -1, -1):
        if seq[i : i + m] == marker:
            return i + m
    return 0

def collate_fn(examples: list[dict[str, Any]]):
    images = []
    full_texts = []
    class_labels = []

    for ex in examples:
        images.append([ex["image"].convert("RGB")])

        full_text = processor.apply_chat_template(
            ex["messages"],
            add_generation_prompt=False,
            tokenize=False,
        )
        full_texts.append(full_text)

        class_labels.append(int(ex["label_id"]))  # 0=healthy(A), 1=nodule(B)

    batch = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    labels = batch["input_ids"].clone()

    # Mask padding
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask image tokens (keep your existing safeguards)
    boi = processor.tokenizer.special_tokens_map.get("boi_token", None)
    if boi is not None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids(boi)
        labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    # Mask everything BEFORE the assistant answer
    for i in range(labels.size(0)):
        assistant_start = _find_assistant_start(
            batch["input_ids"][i], _ASSISTANT_TOKENS
        )
        labels[i, :assistant_start] = -100

    batch["labels"] = labels
    batch["class_labels"] = torch.tensor(class_labels, dtype=torch.long)
    return batch
# %% [markdown]
# ## Training config (TRL SFTConfig)

# %%
from trl import SFTConfig

num_train_epochs = 22
learning_rate = 2e-4

args = SFTConfig(
    output_dir="checkpoints/medgemma-1.5-4b-it-sft-lora-nodulocc-cls",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=5,
    load_best_model_at_end=True,
    greater_is_better=False,
    metric_for_best_model="eval_loss",
    eval_strategy="steps",
    eval_steps=200,
    learning_rate=learning_rate,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    push_to_hub=False,
    report_to="tensorboard",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    label_names=["labels"],
)

# %% [markdown]
# ## Train (SFTTrainer)

# %%

import torch
import torch.nn.functional as F
from trl import SFTTrainer

# Must already be defined (you already do this earlier)
# A_TOKEN_ID, B_TOKEN_ID

def _first_label_pos(labels: torch.Tensor) -> torch.Tensor:
    mask = labels != -100
    has_any = mask.any(dim=1)
    pos = mask.float().argmax(dim=1)
    pos = torch.where(has_any, pos, torch.full_like(pos, -1))
    return pos

def focal_loss_from_logits(
    logits_2c: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    logits_2c: (B, 2)
    targets:   (B,)
    """
    ce = F.cross_entropy(
        logits_2c,
        targets,
        weight=class_weight,
        reduction="none",
    )
    probs = torch.softmax(logits_2c, dim=-1)
    p_t = probs.gather(1, targets.view(-1, 1)).squeeze(1).clamp(1e-6, 1.0)
    loss = ((1.0 - p_t) ** gamma) * ce
    return loss.mean()

class CustomSFTTrainer(SFTTrainer):
    def __init__(
        self,
        *args,
        class_weight: torch.Tensor | None = None,
        focal_gamma: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._class_weight = class_weight
        self._focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_seq = inputs.get("labels", None)          # (B, T) for position
        class_labels = inputs.get("class_labels", None)  # (B,)

        if labels_seq is None or class_labels is None:
            raise ValueError(
                "Batch must contain `labels` (seq) and `class_labels` (0/1)."
            )

        # Don't pass non-model keys to forward
        model_inputs = {
            k: v for k, v in inputs.items() if k not in {"class_labels"}
        }

        outputs = model(**model_inputs)
        logits = outputs.logits  # (B, T, V)

        # Find the FIRST supervised label token position (should be 'A' or 'B')
        pos = _first_label_pos(labels_seq)  # (B,)
        valid = pos != -1
        if valid.sum().item() == 0:
            loss = logits.new_tensor(0.0)
            return (loss, outputs) if return_outputs else loss

        # For causal LM: logits[t-1] predicts labels[t]
        time_idx = (pos - 1).clamp(min=0)  # (B,)

        rows = torch.arange(logits.size(0), device=logits.device)
        ab_logits = torch.stack(
            [
                logits[rows, time_idx, A_TOKEN_ID],
                logits[rows, time_idx, B_TOKEN_ID],
            ],
            dim=-1,
        )  # (B, 2)

        # Filter invalid rows (if any)
        ab_logits = ab_logits[valid].float()
        targets = class_labels.to(logits.device)[valid]

        class_weight = self._class_weight
        if class_weight is not None:
            class_weight = class_weight.to(logits.device).float()

        if self._focal_gamma is not None and self._focal_gamma > 0:
            loss = focal_loss_from_logits(
                ab_logits,
                targets,
                gamma=float(self._focal_gamma),
                class_weight=class_weight,
            )
        else:
            loss = F.cross_entropy(
                ab_logits,
                targets,
                weight=class_weight,
            )

        return (loss, outputs) if return_outputs else loss
    
# %%

import math
import numpy as np
import torch


class RotatingBalancedDataset(torch.utils.data.Dataset):
    """
    Each epoch returns:
      - all positives
      - exactly len(positives) negatives, taken as a rotating chunk
        from the full negative pool (covers all negatives over multiple epochs).
    """

    def __init__(
        self,
        base_ds,
        label_col: str = "label_id",
        pos_label: int = 1,
        neg_label: int = 0,
        seed: int = 0,
    ):
        self.base_ds = base_ds
        self.label_col = label_col
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.seed = int(seed)

        labels = np.array(base_ds[label_col])
        self.pos_idx = np.where(labels == pos_label)[0].tolist()
        self.neg_idx = np.where(labels == neg_label)[0].tolist()

        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            raise ValueError("Need at least 1 positive and 1 negative sample.")

        self.n_pos = len(self.pos_idx)
        self.n_neg = len(self.neg_idx)

        self._epoch_indices: list[int] = []
        self.set_epoch(0)

    def negatives_for_epoch(self, epoch: int) -> list[int]:
        """
        Deterministic rotating-chunk selection.

        Think of negatives as being partitioned into cycles, where each cycle is
        a full permutation of all negatives. Each epoch consumes n_pos negatives.
        """
        need = self.n_pos
        out: list[int] = []

        global_start = epoch * self.n_pos
        cycle = global_start // self.n_neg
        offset = global_start % self.n_neg

        while need > 0:
            rng = np.random.default_rng(self.seed + int(cycle))
            perm = rng.permutation(self.neg_idx).tolist()

            take = min(need, self.n_neg - offset)
            out.extend(perm[offset : offset + take])

            need -= take
            cycle += 1
            offset = 0

        return out

    def set_epoch(self, epoch: int) -> None:
        neg_chunk = self.negatives_for_epoch(int(epoch))

        # Balance: all positives + chunk of negatives
        indices = list(self.pos_idx) + list(neg_chunk)

        # Shuffle within the epoch for mixing
        rng = np.random.default_rng(self.seed + 10_000 + int(epoch))
        rng.shuffle(indices)

        self._epoch_indices = indices

    def __len__(self) -> int:
        return len(self._epoch_indices)

    def __getitem__(self, i: int):
        return self.base_ds[int(self._epoch_indices[i])]

from transformers import TrainerCallback

class SetEpochOnDatasetCallback(TrainerCallback):
    def __init__(self, rotating_ds: RotatingBalancedDataset):
        self.rotating_ds = rotating_ds

    def on_epoch_begin(self, args, state, control, **kwargs):
        # state.epoch can be float; convert safely
        epoch = int(state.epoch) if state.epoch is not None else 0
        self.rotating_ds.set_epoch(epoch)
        return control

train_ds = RotatingBalancedDataset(
    base_ds=train_ds,
    label_col="label_id",
    pos_label=1,
    neg_label=0,
    seed=SPLIT_SEED
)

# %%

import numpy as np
import torch

# --- Resolve token ids for "A" and "B" exactly as used in labels ---
A_ids = processor.tokenizer.encode("A", add_special_tokens=False)
B_ids = processor.tokenizer.encode("B", add_special_tokens=False)

if len(A_ids) != 1 or len(B_ids) != 1:
    raise ValueError(
        f"'A'/'B' are not single tokens with this tokenizer: A={A_ids}, B={B_ids}. "
        "If this happens, we can switch to a sequence logprob method."
    )

A_TOKEN_ID = A_ids[0]
B_TOKEN_ID = B_ids[0]

def _first_label_pos(labels: torch.Tensor) -> torch.Tensor:
    """
    labels: (B, T) with -100 masking
    returns: (B,) position of first non--100 label token, or -1 if none
    """
    mask = labels != -100
    has_any = mask.any(dim=1)
    pos = mask.float().argmax(dim=1)  # 0 if all False
    pos = torch.where(has_any, pos, torch.full_like(pos, -1))
    return pos


def preprocess_logits_for_metrics(logits, labels):
    """
    Called by Trainer on each eval batch.
    Returns a SMALL tensor (B, 2): logits for [A, B] at the answer position.
    """
    if isinstance(logits, (tuple, list)):
        logits = logits[0]  # (B, T, V)

    # labels: (B, T)
    pos = _first_label_pos(labels)  # first label token position (assistant answer)
    # For causal LM, logits[t-1] predicts labels[t]
    time_idx = (pos - 1).clamp(min=0)

    bsz = labels.size(0)
    rows = torch.arange(bsz, device=logits.device)

    # Gather logits for A/B at that time step
    a_logit = logits[rows, time_idx, A_TOKEN_ID]
    b_logit = logits[rows, time_idx, B_TOKEN_ID]
    ab = torch.stack([a_logit, b_logit], dim=-1)  # (B, 2)

    # If a sample had no labels (pos == -1), mark as NaN so we can ignore later
    invalid = pos == -1
    if invalid.any():
        ab[invalid] = torch.nan

    return ab


def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def compute_metrics(eval_pred):
    """
    eval_pred.predictions: (N, 2) logits for [A, B]
    eval_pred.label_ids: (N, T) label token ids with -100 masking
    """
    ab = eval_pred.predictions  # numpy
    label_ids = eval_pred.label_ids  # numpy

    # Filter invalid rows (NaNs) if any
    valid = np.isfinite(ab).all(axis=1)
    ab = ab[valid]
    label_ids = label_ids[valid]

    if len(ab) == 0:
        return {}

    # probs for class B (nodule)
    # softmax over [A, B]
    ab_max = np.max(ab, axis=1, keepdims=True)
    ab_exp = np.exp(ab - ab_max)
    ab_prob = ab_exp / np.sum(ab_exp, axis=1, keepdims=True)
    p_b = ab_prob[:, 1]

    pred = np.argmax(ab, axis=1).astype(int)  # 0 -> A(healthy), 1 -> B(nodule)

    # True label: first non--100 token should be A or B
    # label_ids: (N, T)
    mask = label_ids != -100
    pos = mask.argmax(axis=1)
    true_tok = label_ids[np.arange(len(label_ids)), pos]

    # map token -> class
    # (If something odd appears, drop it)
    true = np.full(len(true_tok), -1, dtype=int)
    true[true_tok == A_TOKEN_ID] = 0
    true[true_tok == B_TOKEN_ID] = 1

    keep = true != -1
    pred = pred[keep]
    true = true[keep]
    p_b = p_b[keep]

    if len(true) == 0:
        return {}

    # Confusion matrix components (positive class = 1 / nodule)
    tp = int(np.sum((pred == 1) & (true == 1)))
    tn = int(np.sum((pred == 0) & (true == 0)))
    fp = int(np.sum((pred == 1) & (true == 0)))
    fn = int(np.sum((pred == 0) & (true == 1)))

    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)  # sensitivity
    spec = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0
    bal_acc = 0.5 * (rec + spec)

    # MCC
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn - fp * fn) / np.sqrt(denom)) if denom > 0 else 0.0

    # Brier score for prob of positive class
    brier = float(np.mean((p_b - true) ** 2))

    # ROC-AUC / PR-AUC require both classes present
    roc_auc = float("nan")
    pr_auc = float("nan")
    if len(np.unique(true)) == 2:
        # ROC-AUC via rank method (simple implementation)
        order = np.argsort(p_b)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(p_b)) + 1
        n_pos = np.sum(true == 1)
        n_neg = np.sum(true == 0)
        sum_ranks_pos = np.sum(ranks[true == 1])
        roc_auc = float(
            (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        )

        # PR-AUC (Average Precision) simple implementation
        # Sort by descending score
        desc = np.argsort(-p_b)
        y = true[desc]
        tp_cum = np.cumsum(y == 1)
        fp_cum = np.cumsum(y == 0)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        recall = tp_cum / np.maximum(n_pos, 1)
        # AP = sum over each positive of precision at that hit / n_pos
        pr_auc = float(np.sum(precision[y == 1]) / max(n_pos, 1))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "coverage": 1.0,  # no generation parsing failures in this method
    }
# %%
trainer = CustomSFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=peft_config,
    processing_class=processor,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    class_weight=torch.tensor([NEG_LOSS_WEIGHT, POS_LOSS_WEIGHT]),
    focal_gamma=None,
)
trainer.add_callback(SetEpochOnDatasetCallback(rotating_ds=train_ds))

# %%
trainer.train(
    resume_from_checkpoint=False
)

# %% [markdown]
# ## Evaluation — Logit-based scoring
 
# Instead of `generate()` + regex parsing, we do a single forward pass and
# extract the logits for tokens **A** (healthy) and **B** (nodule) at the
# position where the model would start generating.  This gives us a
# calibrated probability $p(B)$ that we can threshold however we like.

# %%
import re
import torch
import numpy as np
from tqdm.auto import tqdm
from peft import PeftModel
from transformers import AutoModelForImageTextToText
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    roc_curve,
)

# Clear any leftover objects from training
for _name in ("trainer", "model", "base_model", "ft_model", "pt_model"):
    if _name in globals():
        del globals()[_name]
torch.cuda.empty_cache()

model_id = "google/medgemma-1.5-4b-it"

# ── Resolve single-token IDs for A and B ──
A_ids = processor.tokenizer.encode("A", add_special_tokens=False)
B_ids = processor.tokenizer.encode("B", add_special_tokens=False)
assert len(A_ids) == 1 and len(B_ids) == 1, (
    f"A/B must be single tokens: A={A_ids}, B={B_ids}"
)
A_TOKEN_ID = A_ids[0]
B_TOKEN_ID = B_ids[0]

# ── Prompt (same as training) ──
PROMPT = (
    "You are a radiology assistant. Determine whether this frontal chest "
    "X-ray shows a lung nodule.\n"
    "Answer with exactly one letter:\n"
    "A = Healthy (no lung nodule)\n"
    "B = Nodule present"
)


def _build_user_messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

# %%
def predict_logits_batch(
    model_ref,
    examples: list[dict],
) -> np.ndarray:
    """
    Forward-pass only (no generation).

    Returns
    -------
    p_b : np.ndarray, shape (batch,)
        Probability of class B (nodule) from softmax over [logit_A, logit_B].
    """
    images = [[ex["image"].convert("RGB")] for ex in examples]
    messages = _build_user_messages()

    prompt_texts = [
        processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        for _ in examples
    ]

    batch = processor(
        text=prompt_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(model_ref.device)

    with torch.inference_mode():
        outputs = model_ref(**batch)

    logits = outputs.logits  # (B, T, V)

    # The last real (non-pad) token position is where the model predicts the
    # first generated token.  With left-padding the last column is always
    # the last real token for every row.
    # More robust: find last non-pad position per row.
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    input_ids = batch["input_ids"]  # (B, T)
    # mask: True where NOT pad
    not_pad = input_ids != pad_id
    # last non-pad index per row
    last_pos = not_pad.long().cumsum(dim=1).argmax(dim=1)  # (B,)

    bsz = logits.size(0)
    rows = torch.arange(bsz, device=logits.device)

    a_logit = logits[rows, last_pos, A_TOKEN_ID].float().cpu().numpy()
    b_logit = logits[rows, last_pos, B_TOKEN_ID].float().cpu().numpy()

    ab = np.stack([a_logit, b_logit], axis=1)  # (B, 2)
    ab_max = ab.max(axis=1, keepdims=True)
    ab_exp = np.exp(ab - ab_max)
    ab_prob = ab_exp / ab_exp.sum(axis=1, keepdims=True)

    return ab_prob[:, 1]  # p(B)


def run_logit_evaluation(
    model_ref,
    dataset,
    batch_size: int = 1,
    desc: str = "Logit scoring",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    p_b_all : np.ndarray (N,)   — predicted P(nodule)
    refs    : np.ndarray (N,)   — ground-truth labels (0 or 1)
    """
    p_b_list: list[float] = []
    refs: list[int] = []

    for start in tqdm(range(0, len(dataset), batch_size), desc=desc):
        end = min(start + batch_size, len(dataset))
        batch_examples = [dataset[i] for i in range(start, end)]

        p_b = predict_logits_batch(model_ref, batch_examples)
        p_b_list.extend(p_b.tolist())
        refs.extend(int(ex["label_id"]) for ex in batch_examples)

        if (start // batch_size + 1) % 10 == 0:
            torch.cuda.empty_cache()

    return np.array(p_b_list), np.array(refs)

# %%
# ── Comprehensive metric computation from p(B) + ground truth ──

def compute_full_metrics(
    p_b: np.ndarray,
    refs: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all requested metrics at a given decision threshold."""
    preds = (p_b >= threshold).astype(int)

    tp = int(((preds == 1) & (refs == 1)).sum())
    tn = int(((preds == 0) & (refs == 0)).sum())
    fp = int(((preds == 1) & (refs == 0)).sum())
    fn = int(((preds == 0) & (refs == 1)).sum())

    def _safe(a, b):
        return float(a) / float(b) if b else 0.0

    sensitivity = _safe(tp, tp + fn)          # recall
    specificity = _safe(tn, tn + fp)
    precision = _safe(tp, tp + fp)
    f1 = _safe(2 * precision * sensitivity, precision + sensitivity)
    accuracy = _safe(tp + tn, tp + tn + fp + fn)
    balanced_acc = 0.5 * (sensitivity + specificity)

    # MCC
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt(denom) if denom > 0 else 0.0

    # PR-AUC and ROC-AUC (need both classes)
    pr_auc = roc_auc = float("nan")
    if len(np.unique(refs)) == 2:
        pr_auc = average_precision_score(refs, p_b)
        roc_auc = roc_auc_score(refs, p_b)

    # Brier score
    brier = float(np.mean((p_b - refs) ** 2))

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "mcc": float(mcc),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def sensitivity_at_fixed_specificity(
    p_b: np.ndarray,
    refs: np.ndarray,
    target_spec: float = 0.99,
) -> tuple[float, float]:
    """
    Find the threshold that gives ≥ target_spec specificity,
    then report the sensitivity at that point.

    Returns (sensitivity, threshold).
    """
    fpr, tpr, thresholds = roc_curve(refs, p_b)
    spec = 1.0 - fpr  # specificity = 1 - FPR

    # Find indices where specificity >= target
    valid = np.where(spec >= target_spec)[0]
    if len(valid) == 0:
        # Can't reach that specificity — return the highest specificity point
        idx = np.argmax(spec)
    else:
        # Among valid points, pick the one with highest sensitivity (TPR)
        idx = valid[np.argmax(tpr[valid])]

    return float(tpr[idx]), float(thresholds[idx]) if idx < len(thresholds) else 1.0


def find_threshold_for_target_specificity(
    p_b: np.ndarray,
    refs: np.ndarray,
    target_spec: float = 0.99,
) -> float:
    """Return the threshold that achieves ≥ target_spec specificity."""
    _, thresh = sensitivity_at_fixed_specificity(p_b, refs, target_spec)
    return thresh


def tune_threshold_max_f1(
    p_b: np.ndarray,
    refs: np.ndarray,
) -> float:
    """Find the threshold that maximises F1 on the given data."""
    prec_arr, rec_arr, thresholds = precision_recall_curve(refs, p_b)
    # precision_recall_curve returns len(thresholds) = len(prec) - 1
    f1_arr = np.zeros_like(prec_arr[:-1])
    den = prec_arr[:-1] + rec_arr[:-1]
    valid = den > 0
    f1_arr[valid] = 2 * prec_arr[:-1][valid] * rec_arr[:-1][valid] / den[valid]

    best_idx = np.argmax(f1_arr)
    return float(thresholds[best_idx])


def print_metrics_table(metrics: dict, title: str = ""):
    if title:
        print(f"\n{'=' * 60}")
        print(title)
        print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<25s}: {v:.4f}")
        else:
            print(f"  {k:<25s}: {v}")

# %% [markdown]
# ### Step 1 — Threshold tuning on the validation set

# %%
# in args.output_dir, get latest checkpoint (highest step number) using Pathlib
from pathlib import Path

def get_latest_checkpoint(output_dir: str) -> str | None:
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"No output directory found at {output_dir}")
        return None

    checkpoint_dirs = list(output_path.glob("checkpoint-*"))
    if not checkpoint_dirs:
        print(f"No checkpoint directories found in {output_dir}")
        return None

    # Extract step numbers and find the max
    def extract_step(dir_name: str) -> int:
        match = re.search(r"checkpoint-(\d+)", dir_name)
        return int(match.group(1)) if match else -1

    latest_checkpoint = max(checkpoint_dirs, key=lambda d: extract_step(d.name))
    return str(latest_checkpoint)

checkpoint_path = get_latest_checkpoint(args.output_dir)
if checkpoint_path is None:
    raise ValueError(f"No checkpoint found in {args.output_dir}")
print(f"Loading model from checkpoint: {checkpoint_path}")

# %%
BATCH_SIZE = 1

# ── Load fine-tuned model ──
base_model = AutoModelForImageTextToText.from_pretrained(
    model_id, **model_kwargs
)
ft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
ft_model.eval()

processor.tokenizer.padding_side = "left"

# ── Score the validation set ──
print(f"Scoring validation set ({len(val_ds)} samples) …")
val_p_b, val_refs = run_logit_evaluation(
    ft_model, val_ds, batch_size=BATCH_SIZE, desc="Val logit scoring"
)

# ── Tune thresholds ──
thresh_f1 = tune_threshold_max_f1(val_p_b, val_refs)
thresh_spec99 = find_threshold_for_target_specificity(
    val_p_b, val_refs, target_spec=0.99
)
thresh_spec995 = find_threshold_for_target_specificity(
    val_p_b, val_refs, target_spec=0.995
)

print(f"\nThreshold for max F1 on val:           {thresh_f1:.4f}")
print(f"Threshold for ≥99.0% specificity:      {thresh_spec99:.4f}")
print(f"Threshold for ≥99.5% specificity:      {thresh_spec995:.4f}")

# Show val metrics at each threshold
for name, t in [
    ("Val @ max-F1 threshold", thresh_f1),
    ("Val @ 99% specificity", thresh_spec99),
    ("Val @ 99.5% specificity", thresh_spec995),
    ("Val @ 0.5 (default argmax)", 0.5),
]:
    m = compute_full_metrics(val_p_b, val_refs, threshold=t)
    print_metrics_table(m, title=name)

# %% [markdown]
### Step 2 — Evaluate on the FULL imbalanced test set

# %%
print(f"\nScoring FULL test set ({len(test_ds)} samples) …")
print_class_distribution("Full Test", test_ds)

test_p_b, test_refs = run_logit_evaluation(
    ft_model, test_ds, batch_size=BATCH_SIZE, desc="Test logit scoring (FT)"
)

# ── Report at every operating point ──
for name, t in [
    ("Test — default threshold (0.5)", 0.5),
    ("Test — max-F1 threshold (tuned on val)", thresh_f1),
    ("Test — ≥99.0% specificity (tuned on val)", thresh_spec99),
    ("Test — ≥99.5% specificity (tuned on val)", thresh_spec995),
]:
    m = compute_full_metrics(test_p_b, test_refs, threshold=t)
    print_metrics_table(m, title=name)

# ── Sensitivity @ fixed specificity (computed directly on test) ──
print("\n" + "=" * 60)
print("Sensitivity @ fixed specificity (test set)")
print("=" * 60)
for target_spec in [0.95, 0.99, 0.995]:
    sens, thr = sensitivity_at_fixed_specificity(
        test_p_b, test_refs, target_spec=target_spec
    )
    print(
        f"  Specificity ≥ {target_spec:.1%}:  "
        f"sensitivity = {sens:.4f}  (threshold = {thr:.4f})"
    )

# %% [markdown]
### Step 3 — Zero-shot baseline (logit-based, full test set)

# %%
# Cleanup fine-tuned model before loading pretrained
del ft_model, base_model
torch.cuda.empty_cache()

pt_model = AutoModelForImageTextToText.from_pretrained(
    model_id, **model_kwargs
)
pt_model.eval()

print(f"\nScoring FULL test set with PRETRAINED model ({len(test_ds)} samples) …")
pt_test_p_b, pt_test_refs = run_logit_evaluation(
    pt_model, test_ds, batch_size=BATCH_SIZE, desc="Test logit scoring (PT)"
)

pt_metrics_05 = compute_full_metrics(pt_test_p_b, pt_test_refs, threshold=0.5)
print_metrics_table(pt_metrics_05, title="Zero-shot — threshold 0.5 (full test)")

# Best-case F1 for zero-shot
pt_thresh_f1 = tune_threshold_max_f1(pt_test_p_b, pt_test_refs)
pt_metrics_f1 = compute_full_metrics(
    pt_test_p_b, pt_test_refs, threshold=pt_thresh_f1
)
print_metrics_table(
    pt_metrics_f1,
    title=f"Zero-shot — oracle max-F1 threshold {pt_thresh_f1:.4f} (full test)",
)

del pt_model
torch.cuda.empty_cache()

# %% [markdown]
### Step 4 — Side-by-side comparison table

# %%
ft_metrics_f1_test = compute_full_metrics(
    test_p_b, test_refs, threshold=thresh_f1
)
ft_metrics_spec99_test = compute_full_metrics(
    test_p_b, test_refs, threshold=thresh_spec99
)

rows = [
    ("Zero-shot (0.5)", pt_metrics_05),
    (f"Zero-shot (oracle F1 @ {pt_thresh_f1:.3f})", pt_metrics_f1),
    ("Fine-tuned (0.5)", compute_full_metrics(test_p_b, test_refs, 0.5)),
    (f"Fine-tuned (max-F1 @ {thresh_f1:.3f})", ft_metrics_f1_test),
    (f"Fine-tuned (spec≥99% @ {thresh_spec99:.3f})", ft_metrics_spec99_test),
]

header_metrics = [
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "pr_auc",
    "roc_auc",
    "balanced_accuracy",
]

col_w = 12
name_w = 42
print("\n" + "=" * (name_w + len(header_metrics) * (col_w + 1)))
print(
    f"{'Configuration':<{name_w}}"
    + "".join(f"{m:>{col_w}}" for m in header_metrics)
)
print("-" * (name_w + len(header_metrics) * (col_w + 1)))

for label, m in rows:
    vals = "".join(
        f"{m.get(k, float('nan')):>{col_w}.4f}" for k in header_metrics
    )
    print(f"{label:<{name_w}}{vals}")

print("=" * (name_w + len(header_metrics) * (col_w + 1)))