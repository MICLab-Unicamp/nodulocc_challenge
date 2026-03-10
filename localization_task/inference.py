"""
NoduLoCC2026 — Zero-shot CheXagent-2 Localization Inference Script
===================================================================
Runs CheXagent-2 (StanfordAIMI/CheXagent-2-3b) on a directory of chest
X-ray images and writes a CSV conforming to the NoduLoCC2026 localization
submission format:

    localization_test_results.csv
        file_name, x, y, confidence

Usage
-----
    python infer_localization.py \\
        --input_dir  /path/to/images \\
        --output_dir /path/to/results

Notes
-----
* The confidence column is set to 1.0 for every predicted point because the
  model returns hard bounding-box predictions without a probability score.
* Images are preprocessed with percentile clipping before inference to remove
  tail outliers (burnt-in annotations, detector edge artefacts) and re-stretch
  contrast. This is applied to all bit depths, not just 16-bit images.
  The preprocessed image is written to a temp PNG so CheXagent-2's tokenizer
  can load it via its expected file-path interface.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "StanfordAIMI/CheXagent-2-3b"
SYSTEM_PROMPT = "You are a helpful assistant."
LOC_PROMPT = (
    "Locate areas in the chest X-ray where lung nodules are present, "
    "using bounding box coordinates"
)
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

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
    full 0-255 range. For 16-bit images it performs the necessary windowing
    into uint8.

    Parameters
    ----------
    arr:
        2-D float or integer array.
    low_p, high_p:
        Percentile bounds for clipping.

    Returns
    -------
    uint8 ndarray of the same spatial shape.
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


def preprocess_image(
    image_path: Path,
    low_p: float = 0.5,
    high_p: float = 95.5,
) -> Image.Image:
    """
    Load any supported image and return a normalised RGB PIL image.

    Preprocessing pipeline
    ----------------------
    1. Read raw pixel data via PIL (preserving original bit depth).
    2. Reduce to a single 2-D channel via luminance conversion so that the
       same clipping logic applies regardless of how the DICOM-to-PNG export
       colourised the image.
    3. Apply percentile clip + uint8 rescale unconditionally (all bit depths).
    4. Convert to RGB, which is required by the model.
    """
    with Image.open(image_path) as img:
        arr = np.array(img)

    # Reduce to 2-D using perceptual luminance weights
    if arr.ndim == 3:
        arr = np.array(Image.fromarray(arr).convert("L"))

    if arr.ndim != 2:
        raise ValueError(
            f"Unsupported image shape after channel reduction: {arr.shape} "
            f"for {image_path}"
        )

    arr_u8 = percentile_clip_and_normalise(arr, low_p=low_p, high_p=high_p)
    return Image.fromarray(arr_u8, mode="L").convert("RGB")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str = MODEL_ID):
    """Load CheXagent-2 tokenizer and model onto the available device."""
    print(f"Loading tokenizer from: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model from: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    print(f"Model loaded on: {device}")
    return tokenizer, model

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def generate_localization_response(
    tokenizer,
    model,
    image_path: str,
    prompt: str = LOC_PROMPT,
    system_prompt: str = SYSTEM_PROMPT,
    max_new_tokens: int = 512,
) -> str:
    """Run one forward pass and return the raw text response."""
    query = tokenizer.from_list_format(
        [{"image": image_path}, {"text": prompt}]
    )
    conv = [
        {"from": "system", "value": system_prompt},
        {"from": "human", "value": query},
    ]
    model_inputs = tokenizer.apply_chat_template(
        conv, add_generation_prompt=True, return_tensors="pt"
    )

    if isinstance(model_inputs, torch.Tensor):
        input_ids = model_inputs
        attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs.get(
            "attention_mask", torch.ones_like(input_ids)
        )

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )[0]

    prompt_len = input_ids.shape[1]
    response_ids = output[prompt_len:]
    if (
        tokenizer.eos_token_id is not None
        and response_ids.numel() > 0
        and response_ids[-1].item() == tokenizer.eos_token_id
    ):
        response_ids = response_ids[:-1]

    return tokenizer.decode(response_ids).strip()


def _parse_box_string(box_str: str) -> list[float] | None:
    nums = re.findall(r"[-+]?\d*\.?\d+", str(box_str))
    if len(nums) != 4:
        return None
    x1, y1, x2, y2 = [float(n) for n in nums]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [
        float(np.clip(x1, 0.0, 100.0)),
        float(np.clip(y1, 0.0, 100.0)),
        float(np.clip(x2, 0.0, 100.0)),
        float(np.clip(y2, 0.0, 100.0)),
    ]


def parse_boxes_100(response: str, tokenizer) -> list[list[float]]:
    """Extract normalised [0-100] bounding boxes from the model response."""
    boxes: list[list[float]] = []

    # Primary: official tokenizer helper
    try:
        parsed = tokenizer.to_list_format(response)
        for item in parsed:
            if isinstance(item, dict) and "box" in item:
                box = _parse_box_string(item["box"])
                if box is not None:
                    boxes.append(box)
    except Exception:
        pass

    # Fallback: regex
    if not boxes:
        pattern = re.compile(
            r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,"
            r"\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)"
        )
        for match in pattern.finditer(response):
            box = _parse_box_string(",".join(match.groups()))
            if box is not None:
                boxes.append(box)

    # Deduplicate
    seen: set[tuple] = set()
    deduped: list[list[float]] = []
    for box in boxes:
        key = tuple(round(v, 4) for v in box)
        if key not in seen:
            seen.add(key)
            deduped.append(box)

    return deduped


def boxes100_to_px(
    boxes_100: list[list[float]], width: int, height: int
) -> list[list[float]]:
    return [
        [
            x1 / 100.0 * width,
            y1 / 100.0 * height,
            x2 / 100.0 * width,
            y2 / 100.0 * height,
        ]
        for x1, y1, x2, y2 in boxes_100
    ]


def boxes_to_centers(boxes_px: list[list[float]]) -> list[list[float]]:
    return [
        [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]
        for x1, y1, x2, y2 in boxes_px
    ]

# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def collect_images(input_dir: Path) -> list[Path]:
    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(
            f"No supported images found in: {input_dir}\n"
            f"Supported extensions: {SUPPORTED_EXTENSIONS}"
        )
    return images

# ---------------------------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------------------------

def run_inference(
    input_dir: Path,
    output_dir: Path,
    max_new_tokens: int = 512,
    cache_file: Path | None = None,
    low_percentile: float = 0.5,
    high_percentile: float = 95.5,
) -> Path:
    """
    Run localisation inference on all images in *input_dir* and write
    ``localization_test_results.csv`` to *output_dir*.

    Returns the path of the written CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_images(input_dir)
    print(f"Found {len(image_paths)} image(s) in: {input_dir}")

    tokenizer, model = load_model()

    # Load cache if available
    results: list[dict] = []
    processed: set[str] = set()
    if cache_file and cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        processed = {r["file_name"] for r in results}
        print(f"Resuming from cache: {len(processed)} already done.")

    # Single shared temp file reused across all images to avoid FS churn.
    # CheXagent-2's tokenizer requires a file path, not a PIL object, so we
    # write the preprocessed image to disk before each forward pass.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        for img_path in tqdm(image_paths, desc="Localization inference"):
            file_name = img_path.name
            if file_name in processed:
                continue

            # Capture original dimensions before any preprocessing
            with Image.open(img_path) as img:
                width, height = img.size

            # Preprocess: percentile clip + normalise → uint8 RGB PNG
            preprocessed = preprocess_image(
                img_path,
                low_p=low_percentile,
                high_p=high_percentile,
            )
            preprocessed.save(tmp_path, format="PNG")

            response = generate_localization_response(
                tokenizer, model,
                image_path=str(tmp_path),
                prompt=LOC_PROMPT,
                system_prompt=SYSTEM_PROMPT,
                max_new_tokens=max_new_tokens,
            )

            boxes_100 = parse_boxes_100(response, tokenizer)
            boxes_px  = boxes100_to_px(boxes_100, width, height)
            points    = boxes_to_centers(boxes_px)

            results.append({
                "file_name":      file_name,
                "image_path":     str(img_path),
                "width":          width,
                "height":         height,
                "raw_response":   response,
                "pred_boxes_100": boxes_100,
                "pred_boxes_px":  boxes_px,
                "pred_points":    points,
                "n_pred":         len(points),
            })
            processed.add(file_name)

            # Save cache periodically
            if cache_file:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                torch.cuda.empty_cache()

    finally:
        # Always clean up the shared temp file
        tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Build submission CSV
    # ------------------------------------------------------------------
    rows: list[dict] = []
    for rec in results:
        if rec["n_pred"] == 0:
            # No nodule found — emit nothing (localization task: only
            # positive detections are required).
            continue
        for x, y in rec["pred_points"]:
            rows.append({
                "file_name":  rec["file_name"],
                "x":          round(x, 4),
                "y":          round(y, 4),
                "confidence": 1.0,
            })

    csv_path = output_dir / "localization_test_results.csv"
    df = pd.DataFrame(rows, columns=["file_name", "x", "y", "confidence"])
    df.to_csv(csv_path, index=False)

    print(f"\nDone. Results written to: {csv_path}")
    print(f"  Total rows      : {len(df)}")
    print(f"  Images with pred: {df['file_name'].nunique()} / {len(results)}")
    return csv_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NoduLoCC2026 – zero-shot CheXagent-2 localization inference"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Directory containing input chest X-ray images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Directory where localization_test_results.csv will be saved.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens for generation (default: 512).",
    )
    parser.add_argument(
        "--cache_file",
        type=Path,
        default=None,
        help="Path to an intermediate JSON cache for resumable runs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file. CLI flags override config values.",
    )
    parser.add_argument(
        "--low_percentile",
        type=float,
        default=0.5,
        help="Lower percentile for grayscale clipping (applied to all bit depths, default: 0.5).",
    )
    parser.add_argument(
        "--high_percentile",
        type=float,
        default=95.5,
        help="Upper percentile for grayscale clipping (applied to all bit depths, default: 95.5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        print(
            f"ERROR: input_dir does not exist or is not a directory: {args.input_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    run_inference(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        cache_file=args.cache_file,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
    )


if __name__ == "__main__":
    main()