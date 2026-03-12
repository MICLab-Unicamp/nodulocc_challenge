#!/usr/bin/env python3
"""
Evaluate NoduLoCC2026 keypoint predictions against ground-truth keypoints.

This version uses threshold-aware bipartite matching per threshold:
- only GT/pred pairs within the threshold are eligible
- matching maximizes the number of valid matches (TP)
- among maximum-cardinality matchings, it prefers smaller distances

Supported prediction inputs
---------------------------
1. Inference cache JSON (list of per-image records), e.g.
   [
     {
       "file_name": "0001.png",
       "image_path": "data/nodulocc/lidc_png_16_bit/0001.png",
       "width": 2022,
       "height": 2022,
       "pred_points": [[1213.2, 525.72]],
       "n_pred": 1
     },
     ...
   ]

2. Submission CSV
   file_name,x,y,confidence

Ground truth
------------
Expected at:
    data/nodulocc/localization_labels.csv

Metrics
-------
For each threshold:
- TP / FP / FN
- Precision / Recall / F1
- PCK
- MRE ± std (pixels)
- Median Radial Error (pixels)
- Normalized error stats (fraction of image diagonal)

Thresholds
----------
By default:
- tight:    0.02 × image diagonal
- tolerant: 0.05 × image diagonal

Outputs
-------
- eval_summary.json
- per_image_metrics.csv

Usage
-----
python eval_localization.py \
    --predictions results_cache.json \
    --ground_truth data/nodulocc/localization_labels.csv \
    --output_dir eval_results

python eval_localization.py \
    --predictions results/localization_test_results.csv \
    --ground_truth data/nodulocc/localization_labels.csv \
    --image_roots data/nodulocc/nih_filtered_images data/nodulocc/lidc_png_16_bit \
    --output_dir eval_results
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate keypoint localization predictions with "
        "threshold-aware matching."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to prediction JSON cache or submission CSV.",
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        default=Path("data/nodulocc/localization_labels.csv"),
        help="Path to localization_labels.csv.",
    )
    parser.add_argument(
        "--image_roots",
        type=Path,
        nargs="*",
        default=[
            Path("data/nodulocc/nih_filtered_images"),
            Path("data/nodulocc/lidc_png_16_bit"),
        ],
        help=(
            "Directories used to resolve image sizes when width/height are not "
            "present in predictions."
        ),
    )
    parser.add_argument(
        "--tight_frac",
        type=float,
        default=0.02,
        help="Tight threshold as fraction of image diagonal (default: 0.02).",
    )
    parser.add_argument(
        "--tolerant_frac",
        type=float,
        default=0.05,
        help="Tolerant threshold as fraction of image diagonal (default: 0.05).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("eval_results"),
        help="Directory to save eval_summary.json and per_image_metrics.csv.",
    )
    return parser.parse_args()


def _points_array(points: list[list[float]] | list[tuple[float, float]]) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    arr = np.asarray(points, dtype=np.float64)
    return arr.reshape(-1, 2)


def load_ground_truth(
    gt_csv: Path,
) -> dict[str, list[list[float]]]:
    df = pd.read_csv(gt_csv)

    required = {"file_name", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Ground-truth CSV is missing required columns: {sorted(missing)}"
        )

    gt_by_file: dict[str, list[list[float]]] = {}
    for row in df.itertuples(index=False):
        gt_by_file.setdefault(row.file_name, []).append([float(row.x), float(row.y)])
    return gt_by_file


def load_predictions_json(
    pred_json: Path,
) -> tuple[dict[str, list[list[float]]], dict[str, tuple[int, int]]]:
    with open(pred_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Prediction JSON must be a list of per-image records.")

    pred_by_file: dict[str, list[list[float]]] = {}
    size_by_file: dict[str, tuple[int, int]] = {}

    for rec in data:
        if not isinstance(rec, dict):
            continue

        file_name = rec.get("file_name")
        if not file_name:
            continue

        pred_points = rec.get("pred_points", [])
        if pred_points is None:
            pred_points = []

        pred_by_file.setdefault(file_name, [])
        for pt in pred_points:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                pred_by_file[file_name].append([float(pt[0]), float(pt[1])])

        width = rec.get("width")
        height = rec.get("height")
        if width is not None and height is not None:
            size_by_file[file_name] = (int(width), int(height))

    return pred_by_file, size_by_file


def load_predictions_csv(
    pred_csv: Path,
) -> tuple[dict[str, list[list[float]]], dict[str, tuple[int, int]]]:
    df = pd.read_csv(pred_csv)

    required = {"file_name", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Prediction CSV is missing required columns: {sorted(missing)}"
        )

    pred_by_file: dict[str, list[list[float]]] = {}
    for row in df.itertuples(index=False):
        pred_by_file.setdefault(row.file_name, []).append([float(row.x), float(row.y)])

    return pred_by_file, {}


def load_predictions(
    pred_path: Path,
) -> tuple[dict[str, list[list[float]]], dict[str, tuple[int, int]]]:
    suffix = pred_path.suffix.lower()
    if suffix == ".json":
        return load_predictions_json(pred_path)
    if suffix == ".csv":
        return load_predictions_csv(pred_path)
    raise ValueError(
        f"Unsupported prediction format: {pred_path}. Expected .json or .csv."
    )


def build_image_index(image_roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in image_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                index.setdefault(path.name, path)
    return index


def resolve_image_size(
    file_name: str,
    size_by_file: dict[str, tuple[int, int]],
    image_index: dict[str, Path],
) -> tuple[int, int]:
    if file_name in size_by_file:
        return size_by_file[file_name]

    img_path = image_index.get(file_name)
    if img_path is None:
        raise FileNotFoundError(
            f"Could not resolve image size for {file_name}. "
            f"Provide prediction JSON with width/height or supply valid "
            f"--image_roots."
        )

    with Image.open(img_path) as img:
        width, height = img.size
    return int(width), int(height)


def pairwise_distances(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if len(gt) == 0 or len(pred) == 0:
        return np.empty((len(gt), len(pred)), dtype=np.float64)
    diff = gt[:, None, :] - pred[None, :, :]
    return np.linalg.norm(diff, axis=2)


def threshold_aware_matching(
    gt: np.ndarray,
    pred: np.ndarray,
    threshold_px: float,
) -> list[dict[str, Any]]:
    """
    Maximum-cardinality matching under a distance threshold.

    Only edges with distance <= threshold_px are allowed.
    Among all maximum-cardinality matchings, smaller distances are preferred.
    """
    if len(gt) == 0 or len(pred) == 0:
        return []

    dist = pairwise_distances(gt, pred)

    G = nx.Graph()
    gt_nodes = [f"g{i}" for i in range(len(gt))]
    pred_nodes = [f"p{j}" for j in range(len(pred))]

    G.add_nodes_from(gt_nodes, bipartite=0)
    G.add_nodes_from(pred_nodes, bipartite=1)

    # Maximize cardinality first; then prefer smaller distances.
    # Since max_weight_matching maximizes total weight, assign weight = C - d.
    # With all valid edges having positive weight, maxcardinality=True gives the
    # largest number of matches, and within that prefers smaller distances.
    C = float(threshold_px) + 1.0

    for i in range(len(gt)):
        for j in range(len(pred)):
            d = float(dist[i, j])
            if d <= threshold_px:
                G.add_edge(
                    f"g{i}",
                    f"p{j}",
                    weight=C - d,
                    distance_px=d,
                    gt_index=i,
                    pred_index=j,
                )

    matching = nx.algorithms.matching.max_weight_matching(
        G,
        maxcardinality=True,
        weight="weight",
    )

    matches: list[dict[str, Any]] = []
    for u, v in matching:
        if u.startswith("g"):
            g_node, p_node = u, v
        else:
            g_node, p_node = v, u

        if not (g_node.startswith("g") and p_node.startswith("p")):
            continue

        edge = G[g_node][p_node]
        matches.append(
            {
                "gt_index": int(edge["gt_index"]),
                "pred_index": int(edge["pred_index"]),
                "distance_px": float(edge["distance_px"]),
            }
        )

    matches.sort(key=lambda m: (m["gt_index"], m["pred_index"]))
    return matches


def safe_precision(tp: int, fp: int, gt_count: int) -> float:
    denom = tp + fp
    if denom == 0:
        return 1.0 if gt_count == 0 else 0.0
    return tp / denom


def safe_recall(tp: int, fn: int, pred_count: int) -> float:
    denom = tp + fn
    if denom == 0:
        return 1.0 if pred_count == 0 else 0.0
    return tp / denom


def safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def std_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.std(values))


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


def evaluate(
    gt_by_file: dict[str, list[list[float]]],
    pred_by_file: dict[str, list[list[float]]],
    size_by_file: dict[str, tuple[int, int]],
    image_index: dict[str, Path],
    tight_frac: float,
    tolerant_frac: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    all_files = sorted(set(gt_by_file) | set(pred_by_file))

    per_image_rows: list[dict[str, Any]] = []

    total_gt = 0
    total_pred = 0

    total_tp_tight = 0
    total_fp_tight = 0
    total_fn_tight = 0

    total_tp_tolerant = 0
    total_fp_tolerant = 0
    total_fn_tolerant = 0

    matched_errors_px_tight: list[float] = []
    matched_errors_norm_tight: list[float] = []

    matched_errors_px_tolerant: list[float] = []
    matched_errors_norm_tolerant: list[float] = []

    macro_precision_tight: list[float] = []
    macro_recall_tight: list[float] = []
    macro_f1_tight: list[float] = []

    macro_precision_tolerant: list[float] = []
    macro_recall_tolerant: list[float] = []
    macro_f1_tolerant: list[float] = []

    pck_tight_hits = 0
    pck_tolerant_hits = 0

    for file_name in all_files:
        gt = _points_array(gt_by_file.get(file_name, []))
        pred = _points_array(pred_by_file.get(file_name, []))

        width, height = resolve_image_size(file_name, size_by_file, image_index)
        diagonal = float(math.hypot(width, height))
        tight_px = tight_frac * diagonal
        tolerant_px = tolerant_frac * diagonal

        matches_tight = threshold_aware_matching(gt, pred, tight_px)
        matches_tolerant = threshold_aware_matching(gt, pred, tolerant_px)

        tp_tight = len(matches_tight)
        fn_tight = len(gt) - tp_tight
        fp_tight = len(pred) - tp_tight

        tp_tolerant = len(matches_tolerant)
        fn_tolerant = len(gt) - tp_tolerant
        fp_tolerant = len(pred) - tp_tolerant

        errors_px_tight = [m["distance_px"] for m in matches_tight]
        errors_px_tolerant = [m["distance_px"] for m in matches_tolerant]

        errors_norm_tight = [d / diagonal for d in errors_px_tight]
        errors_norm_tolerant = [d / diagonal for d in errors_px_tolerant]

        precision_tight = safe_precision(tp_tight, fp_tight, len(gt))
        recall_tight = safe_recall(tp_tight, fn_tight, len(pred))
        f1_tight = safe_f1(precision_tight, recall_tight)

        precision_tolerant = safe_precision(tp_tolerant, fp_tolerant, len(gt))
        recall_tolerant = safe_recall(tp_tolerant, fn_tolerant, len(pred))
        f1_tolerant = safe_f1(precision_tolerant, recall_tolerant)

        total_gt += len(gt)
        total_pred += len(pred)

        total_tp_tight += tp_tight
        total_fp_tight += fp_tight
        total_fn_tight += fn_tight

        total_tp_tolerant += tp_tolerant
        total_fp_tolerant += fp_tolerant
        total_fn_tolerant += fn_tolerant

        matched_errors_px_tight.extend(errors_px_tight)
        matched_errors_norm_tight.extend(errors_norm_tight)

        matched_errors_px_tolerant.extend(errors_px_tolerant)
        matched_errors_norm_tolerant.extend(errors_norm_tolerant)

        macro_precision_tight.append(precision_tight)
        macro_recall_tight.append(recall_tight)
        macro_f1_tight.append(f1_tight)

        macro_precision_tolerant.append(precision_tolerant)
        macro_recall_tolerant.append(recall_tolerant)
        macro_f1_tolerant.append(f1_tolerant)

        pck_tight_hits += tp_tight
        pck_tolerant_hits += tp_tolerant

        per_image_rows.append(
            {
                "file_name": file_name,
                "width": width,
                "height": height,
                "diagonal": diagonal,
                "n_gt": int(len(gt)),
                "n_pred": int(len(pred)),
                "tight_threshold_px": tight_px,
                "tolerant_threshold_px": tolerant_px,
                "tp_tight": tp_tight,
                "fp_tight": fp_tight,
                "fn_tight": fn_tight,
                "precision_tight": precision_tight,
                "recall_tight": recall_tight,
                "f1_tight": f1_tight,
                "tp_tolerant": tp_tolerant,
                "fp_tolerant": fp_tolerant,
                "fn_tolerant": fn_tolerant,
                "precision_tolerant": precision_tolerant,
                "recall_tolerant": recall_tolerant,
                "f1_tolerant": f1_tolerant,
                "mean_error_px_tight": mean_or_none(errors_px_tight),
                "median_error_px_tight": median_or_none(errors_px_tight),
                "mean_error_px_tolerant": mean_or_none(errors_px_tolerant),
                "median_error_px_tolerant": median_or_none(errors_px_tolerant),
            }
        )

    micro_precision_tight = safe_precision(
        total_tp_tight, total_fp_tight, total_gt
    )
    micro_recall_tight = safe_recall(
        total_tp_tight, total_fn_tight, total_pred
    )
    micro_f1_tight = safe_f1(micro_precision_tight, micro_recall_tight)

    micro_precision_tolerant = safe_precision(
        total_tp_tolerant, total_fp_tolerant, total_gt
    )
    micro_recall_tolerant = safe_recall(
        total_tp_tolerant, total_fn_tolerant, total_pred
    )
    micro_f1_tolerant = safe_f1(
        micro_precision_tolerant, micro_recall_tolerant
    )

    summary = {
        "n_images_evaluated": len(all_files),
        "n_gt_points": int(total_gt),
        "n_pred_points": int(total_pred),
        "thresholds": {
            "tight_fraction_of_diagonal": tight_frac,
            "tolerant_fraction_of_diagonal": tolerant_frac,
        },
        "tight_threshold_metrics": {
            "tp": int(total_tp_tight),
            "fp": int(total_fp_tight),
            "fn": int(total_fn_tight),
            "precision_micro": micro_precision_tight,
            "recall_micro": micro_recall_tight,
            "f1_micro": micro_f1_tight,
            "precision_macro": mean_or_none(macro_precision_tight),
            "recall_macro": mean_or_none(macro_recall_tight),
            "f1_macro": mean_or_none(macro_f1_tight),
            "pck": (pck_tight_hits / total_gt) if total_gt > 0 else None,
            "mre_px_mean": mean_or_none(matched_errors_px_tight),
            "mre_px_std": std_or_none(matched_errors_px_tight),
            "median_radial_error_px": median_or_none(matched_errors_px_tight),
            "mre_diag_mean": mean_or_none(matched_errors_norm_tight),
            "mre_diag_std": std_or_none(matched_errors_norm_tight),
            "median_radial_error_diag": median_or_none(
                matched_errors_norm_tight
            ),
            "n_matched_pairs": len(matched_errors_px_tight),
        },
        "tolerant_threshold_metrics": {
            "tp": int(total_tp_tolerant),
            "fp": int(total_fp_tolerant),
            "fn": int(total_fn_tolerant),
            "precision_micro": micro_precision_tolerant,
            "recall_micro": micro_recall_tolerant,
            "f1_micro": micro_f1_tolerant,
            "precision_macro": mean_or_none(macro_precision_tolerant),
            "recall_macro": mean_or_none(macro_recall_tolerant),
            "f1_macro": mean_or_none(macro_f1_tolerant),
            "pck": (pck_tolerant_hits / total_gt) if total_gt > 0 else None,
            "mre_px_mean": mean_or_none(matched_errors_px_tolerant),
            "mre_px_std": std_or_none(matched_errors_px_tolerant),
            "median_radial_error_px": median_or_none(
                matched_errors_px_tolerant
            ),
            "mre_diag_mean": mean_or_none(matched_errors_norm_tolerant),
            "mre_diag_std": std_or_none(matched_errors_norm_tolerant),
            "median_radial_error_diag": median_or_none(
                matched_errors_norm_tolerant
            ),
            "n_matched_pairs": len(matched_errors_px_tolerant),
        },
        "notes": {
            "matching": (
                "Threshold-aware matching is computed per image and per "
                "threshold. Only GT/pred pairs within threshold are eligible."
            ),
            "matching_objective": (
                "The matching maximizes the number of valid pairs first, then "
                "prefers smaller distances among those solutions."
            ),
            "pck_definition": (
                "PCK is matched keypoints within threshold divided by total GT "
                "keypoints."
            ),
            "normalization": (
                "Thresholds are fractions of each image diagonal."
            ),
        },
    }

    per_image_df = pd.DataFrame(per_image_rows)
    return summary, per_image_df


def print_summary(summary: dict[str, Any]) -> None:
    tight = summary["tight_threshold_metrics"]
    tolerant = summary["tolerant_threshold_metrics"]

    print("\nEvaluation summary")
    print("------------------")
    print(f"Images evaluated : {summary['n_images_evaluated']}")
    print(f"GT keypoints      : {summary['n_gt_points']}")
    print(f"Pred keypoints    : {summary['n_pred_points']}")

    print("\nTight threshold")
    print(
        f"  frac(diag)      : "
        f"{summary['thresholds']['tight_fraction_of_diagonal']}"
    )
    print(f"  TP / FP / FN    : {tight['tp']} / {tight['fp']} / {tight['fn']}")
    print(f"  Precision       : {tight['precision_micro']:.6f} (micro)")
    print(f"  Recall          : {tight['recall_micro']:.6f} (micro)")
    print(f"  F1              : {tight['f1_micro']:.6f} (micro)")
    print(
        f"  PCK             : {tight['pck']:.6f}"
        if tight["pck"] is not None
        else "  PCK             : None"
    )
    print(
        f"  MRE ± std (px)  : "
        f"{tight['mre_px_mean']:.6f} ± {tight['mre_px_std']:.6f}"
        if tight["mre_px_mean"] is not None and tight["mre_px_std"] is not None
        else "  MRE ± std (px)  : None"
    )
    print(
        f"  Median RE (px)  : {tight['median_radial_error_px']:.6f}"
        if tight["median_radial_error_px"] is not None
        else "  Median RE (px)  : None"
    )

    print("\nTolerant threshold")
    print(
        f"  frac(diag)      : "
        f"{summary['thresholds']['tolerant_fraction_of_diagonal']}"
    )
    print(
        f"  TP / FP / FN    : "
        f"{tolerant['tp']} / {tolerant['fp']} / {tolerant['fn']}"
    )
    print(f"  Precision       : {tolerant['precision_micro']:.6f} (micro)")
    print(f"  Recall          : {tolerant['recall_micro']:.6f} (micro)")
    print(f"  F1              : {tolerant['f1_micro']:.6f} (micro)")
    print(
        f"  PCK             : {tolerant['pck']:.6f}"
        if tolerant["pck"] is not None
        else "  PCK             : None"
    )
    print(
        f"  MRE ± std (px)  : "
        f"{tolerant['mre_px_mean']:.6f} ± {tolerant['mre_px_std']:.6f}"
        if tolerant["mre_px_mean"] is not None
        and tolerant["mre_px_std"] is not None
        else "  MRE ± std (px)  : None"
    )
    print(
        f"  Median RE (px)  : {tolerant['median_radial_error_px']:.6f}"
        if tolerant["median_radial_error_px"] is not None
        else "  Median RE (px)  : None"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_by_file = load_ground_truth(args.ground_truth)
    pred_by_file, size_by_file = load_predictions(args.predictions)
    image_index = build_image_index(args.image_roots)

    summary, per_image_df = evaluate(
        gt_by_file=gt_by_file,
        pred_by_file=pred_by_file,
        size_by_file=size_by_file,
        image_index=image_index,
        tight_frac=args.tight_frac,
        tolerant_frac=args.tolerant_frac,
    )

    summary_path = args.output_dir / "eval_summary.json"
    per_image_path = args.output_dir / "per_image_metrics.csv"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    per_image_df.to_csv(per_image_path, index=False)

    print_summary(summary)
    print(f"\nSaved summary JSON : {summary_path}")
    print(f"Saved per-image CSV: {per_image_path}")


if __name__ == "__main__":
    main()