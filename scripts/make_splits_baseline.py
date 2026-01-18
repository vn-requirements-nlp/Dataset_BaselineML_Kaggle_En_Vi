# scripts/make_splits_baseline.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if "text" not in obj:
                raise ValueError(f"Line {ln}: missing key 'text'")

            # normalize legacy key
            if "labels" not in obj and "label" in obj:
                obj["labels"] = obj["label"]

            if "labels" not in obj and "label_ids" not in obj:
                raise ValueError(f"Line {ln}: missing 'labels'/'label_ids' (multi-label)")

            rows.append(obj)
    return rows


def load_labelmap(path: Path) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Expect labelmap_multilabel.json format:
    {
      "label_names": [...],
      "label2id": {...},
      "id2label": {"0":"Functional (F)", ...}
    }
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    label_names = obj.get("label_names")
    label2id = obj.get("label2id")
    id2label_raw = obj.get("id2label")

    if not isinstance(label_names, list) or not isinstance(label2id, dict) or not isinstance(id2label_raw, dict):
        raise ValueError("Invalid labelmap format. Expected keys: label_names, label2id, id2label.")

    # id2label keys might be strings
    id2label: Dict[int, str] = {}
    for k, v in id2label_raw.items():
        id2label[int(k)] = v

    # ensure consistent
    if len(label_names) != len(label2id) or len(label_names) != len(id2label):
        # not fatal, but warn via exception for safety
        raise ValueError("Labelmap size mismatch among label_names/label2id/id2label.")

    return label_names, {k: int(v) for k, v in label2id.items()}, id2label


def to_multihot(row: Dict[str, Any], label2id: Dict[str, int], num_labels: int) -> np.ndarray:
    """
    Converts one JSONL row to a multi-hot vector.
    Prefer 'label_ids' if present; fallback to 'labels' (list[str]).
    """
    y = np.zeros((num_labels,), dtype=np.int64)

    if "label_ids" in row and row["label_ids"] is not None:
        for lid in row["label_ids"]:
            y[int(lid)] = 1
        return y

    labels = row.get("labels", [])
    if labels is None:
        labels = []
    if not isinstance(labels, list):
        raise ValueError("Expected 'labels' to be a list[str]")

    for lab in labels:
        if lab not in label2id:
            raise ValueError(f"Unknown label name in data: {lab}")
        y[label2id[lab]] = 1
    return y


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_split_indices(split_path: Path, split_name: str) -> List[int]:
    obj = json.loads(split_path.read_text(encoding="utf-8"))
    key = {"train": "train_idx", "val": "val_idx", "test": "test_idx"}[split_name]
    idxs = obj.get(key)
    if not isinstance(idxs, list):
        raise ValueError(f"Split file missing key '{key}' or invalid format.")
    return idxs


def best_threshold_for_label(
    y_true_1d: np.ndarray,
    scores_1d: np.ndarray,
    grid: np.ndarray
) -> Dict[str, float]:
    """
    Find threshold that maximizes F1 for one label.
    Returns dict: {"thr":..., "f1":..., "p":..., "r":...}
    """
    y_true_1d = y_true_1d.astype(np.int64)

    best = {"thr": 0.5, "f1": 0.0, "p": 0.0, "r": 0.0}

    # if label never appears in val => keep default threshold
    if y_true_1d.sum() == 0:
        return best

    for thr in grid:
        y_pred = (scores_1d >= thr).astype(np.int64)

        tp = int(((y_pred == 1) & (y_true_1d == 1)).sum())
        fp = int(((y_pred == 1) & (y_true_1d == 0)).sum())
        fn = int(((y_pred == 0) & (y_true_1d == 1)).sum())

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        # tie-breaker: higher f1, then higher recall, then higher precision
        if (f1 > best["f1"]) or (
            abs(f1 - best["f1"]) < 1e-12 and (r > best["r"] or (abs(r - best["r"]) < 1e-12 and p > best["p"]))
        ):
            best = {"thr": float(thr), "f1": float(f1), "p": float(p), "r": float(r)}

    return best
