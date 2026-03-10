#!/usr/bin/env python
"""
NoduLoCC classification inference script using a merged MedGemma model from
Hugging Face.

Output:
  <output_dir>/classification_test_results.csv

Required CSV columns:
  - file_name
  - label
  - confidence

Example:
  python inference.py \
    --input_dir ./data/test_images \
    --output_dir ./outputs \
    --model_id k298976/medgemma-1.5-4b-it-nodulocc-cls \
    --threshold 0.5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

load_dotenv()

DEFAULT_MODEL_ID = "k298976/medgemma-1.5-4b-it-nodulocc-cls"

PROMPT = (
    "You are a radiology assistant. Determine whether this frontal chest X-ray "
    "shows a lung nodule.\n"
    "Answer with exactly one letter:\n"
    "A = Healthy (no lung nodule)\n"
    "B = Nodule present"
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run chest X-ray nodule classification inference with a merged "
            "MedGemma model from Hugging Face."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where classification_test_results.csv will be saved.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model repo ID for the merged model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Inference batch size. Use 1 if GPU memory is limited.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Decision threshold for predicting 'Nodule' from p(Nodule). "
            "Set this to your validation-tuned threshold if desired."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories for images.",
    )
    parser.add_argument(
        "--file_name_mode",
        choices=["name", "relative"],
        default="name",
        help=(
            "How to write file_name into the output CSV. "
            "'name' uses basename only. "
            "'relative' uses the path relative to input_dir."
        ),
    )
    parser.add_argument(
        "--confidence_mode",
        choices=["nodule", "predicted"],
        default="nodule",
        help=(
            "How to populate the confidence column. "
            "'nodule' writes p(Nodule). "
            "'predicted' writes the confidence of the predicted class."
        ),
    )
    parser.add_argument(
        "--low_percentile",
        type=float,
        default=0.5,
        help="Lower percentile for grayscale clipping (applied to all bit depths).",
    )
    parser.add_argument(
        "--high_percentile",
        type=float,
        default=95.5,
        help="Upper percentile for grayscale clipping (applied to all bit depths).",
    )
    return parser.parse_args()


def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def percentile_clip_and_normalise(
    arr: np.ndarray,
    low_p: float = 0.5,
    high_p: float = 95.5,
) -> np.ndarray:
    """
    Clip a single-channel 2-D array to [low_p, high_p] percentiles and
    rescale to uint8 [0, 255].

    Applying this to 8-bit images as well removes tail outliers (e.g. burnt-in
    annotations, detector edge artefacts) and re-stretches contrast to the
    full 0-255 range, matching the normalisation seen during training.

    Parameters
    ----------
    arr:
        2-D float or integer array. Must be 2-D (single channel).
    low_p, high_p:
        Percentile bounds for clipping. Defaults match training preprocessing.

    Returns
    -------
    uint8 ndarray of the same shape.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale array, got shape {arr.shape}")

    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [low_p, high_p])

    # Guard against flat images (e.g. all-zero padding frames)
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return np.zeros(arr.shape, dtype=np.uint8)

    clipped = np.clip(arr, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    return np.round(scaled).astype(np.uint8)


def load_image_for_model(
    image_path: Path,
    low_p: float = 0.5,
    high_p: float = 95.5,
) -> Image.Image:
    """
    Load any supported image and return a normalised RGB PIL image.

    Preprocessing pipeline
    ----------------------
    1. Read raw pixel data via PIL (preserving original bit depth).
    2. For multi-channel images, convert to luminance (single channel) so that
       the same clipping logic applies regardless of how the DICOM-to-PNG
       export colourised the image.
    3. Apply percentile clip + uint8 rescale via `percentile_clip_and_normalise`.
       This step is intentionally applied to *all* images (not just 16-bit)
       so that:
         - 16-bit scanner images are correctly windowed into uint8.
         - 8-bit images with tail outliers (burnt-in text, edge artefacts)
           receive the same contrast stretching used during training.
    4. Convert the resulting uint8 grayscale image to RGB (3-channel), which
       is required by the MedGemma processor.
    """
    with Image.open(image_path) as img:
        arr = np.array(img)

    # --- Reduce to a single 2-D channel -----------------------------------
    if arr.ndim == 3:
        # Use luminance weights if the image was stored as RGB/RGBA.
        # This is more faithful than just taking channel 0 for colour images,
        # and is a no-op for images that are RGB but actually grayscale.
        pil_gray = Image.fromarray(arr).convert("L")
        arr = np.array(pil_gray)          # now shape (H, W), dtype uint8

    if arr.ndim != 2:
        raise ValueError(
            f"Unsupported image shape after channel reduction: {arr.shape} "
            f"for {image_path}"
        )

    # --- Percentile clip + rescale to uint8 (all bit depths) --------------
    arr_u8 = percentile_clip_and_normalise(arr, low_p=low_p, high_p=high_p)

    # --- Return as RGB PIL image ------------------------------------------
    return Image.fromarray(arr_u8, mode="L").convert("RGB")


def collect_image_paths(input_dir: Path, recursive: bool) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if recursive:
        paths = [
            p
            for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        paths = [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

    paths = sorted(paths)
    if not paths:
        raise ValueError(f"No image files found under: {input_dir}")

    return paths


def load_model_and_processor(model_id: str):
    hf_token = os.getenv("HF_TOKEN") or None
    torch_dtype = get_torch_dtype()

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    processor = AutoProcessor.from_pretrained(model_id, token=hf_token)

    processor.tokenizer.padding_side = "left"
    model.eval()

    device = next(model.parameters()).device
    return model, processor, device


def resolve_ab_token_ids(processor) -> tuple[int, int]:
    a_ids = processor.tokenizer.encode("A", add_special_tokens=False)
    b_ids = processor.tokenizer.encode("B", add_special_tokens=False)

    if len(a_ids) != 1 or len(b_ids) != 1:
        raise ValueError(
            f"'A'/'B' are not single tokens for this tokenizer: "
            f"A={a_ids}, B={b_ids}"
        )

    return a_ids[0], b_ids[0]


def build_prompt_text(processor) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def move_batch_to_device(batch, device):
    return {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
    }


def predict_p_nodule_batch(
    model,
    processor,
    device: torch.device,
    prompt_text: str,
    image_paths: list[Path],
    a_token_id: int,
    b_token_id: int,
    low_p: float,
    high_p: float,
) -> np.ndarray:
    images = [
        [load_image_for_model(p, low_p=low_p, high_p=high_p)] for p in image_paths
    ]
    texts = [prompt_text] * len(image_paths)

    batch = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    batch = move_batch_to_device(batch, device)

    with torch.inference_mode():
        outputs = model(**batch)

    logits = outputs.logits
    input_ids = batch["input_ids"]

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    not_pad = input_ids != pad_id
    positions = torch.arange(
        input_ids.size(1), device=input_ids.device
    ).unsqueeze(0)
    last_pos = (positions * not_pad.long()).max(dim=1).values

    rows = torch.arange(logits.size(0), device=logits.device)
    a_logit = logits[rows, last_pos, a_token_id].float()
    b_logit = logits[rows, last_pos, b_token_id].float()

    ab_logits = torch.stack([a_logit, b_logit], dim=-1)
    ab_probs = torch.softmax(ab_logits, dim=-1)

    return ab_probs[:, 1].detach().cpu().numpy()


def format_file_name(
    image_path: Path,
    input_dir: Path,
    file_name_mode: str,
) -> str:
    if file_name_mode == "relative":
        return image_path.relative_to(input_dir).as_posix()
    return image_path.name


def confidence_from_prob(
    p_nodule: float,
    pred_label: str,
    confidence_mode: str,
) -> float:
    if confidence_mode == "predicted":
        if pred_label == "Nodule":
            return float(p_nodule)
        return float(1.0 - p_nodule)

    return float(p_nodule)


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "classification_test_results.csv"

    print(f"Loading merged model from Hugging Face: {args.model_id}")
    model, processor, device = load_model_and_processor(args.model_id)

    a_token_id, b_token_id = resolve_ab_token_ids(processor)
    prompt_text = build_prompt_text(processor)

    image_paths = collect_image_paths(args.input_dir, recursive=args.recursive)
    print(f"Found {len(image_paths)} image(s) in {args.input_dir}")

    rows: list[dict[str, object]] = []

    for start in tqdm(
        range(0, len(image_paths), args.batch_size),
        desc="Running inference",
    ):
        end = min(start + args.batch_size, len(image_paths))
        batch_paths = image_paths[start:end]

        p_nodule_batch = predict_p_nodule_batch(
            model=model,
            processor=processor,
            device=device,
            prompt_text=prompt_text,
            image_paths=batch_paths,
            a_token_id=a_token_id,
            b_token_id=b_token_id,
            low_p=args.low_percentile,
            high_p=args.high_percentile,
        )

        for image_path, p_nodule in zip(batch_paths, p_nodule_batch):
            p_nodule = float(p_nodule)
            label = "Nodule" if p_nodule >= args.threshold else "No Finding"
            confidence = confidence_from_prob(
                p_nodule=p_nodule,
                pred_label=label,
                confidence_mode=args.confidence_mode,
            )

            confidence = f"{confidence:.5f}"

            rows.append(
                {
                    "file_name": format_file_name(
                        image_path=image_path,
                        input_dir=args.input_dir,
                        file_name_mode=args.file_name_mode,
                    ),
                    "label": label,
                    "confidence": confidence,
                }
            )

    df = pd.DataFrame(rows, columns=["file_name", "label", "confidence"])
    df.to_csv(output_csv, index=False)

    print(f"Saved predictions to: {output_csv}")
    print(df.head())


if __name__ == "__main__":
    main()