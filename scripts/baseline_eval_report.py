# scripts/baseline_eval_report.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-grade evaluation for baseline ML:
- Evaluate on fixed split (train/val/test) OR evaluate ALL rows (for Gold test set)
- Apply per-label thresholds tuned from VAL
- Print classification_report and summary metrics
- IMPORTANT: keep preprocessing consistent with training (pyvi ViTokenizer if used)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from scripts.baseline_make_splits import (
    read_jsonl, load_labelmap, to_multihot, load_split_indices, sigmoid
)
from scripts.baseline_feature_utils import (
    load_seed_words_from_model_dir, build_seed_features,
    append_seed_features, seed_words_enabled
)


def maybe_tokenize_vi(text: str, use_vitokenizer: bool) -> str:
    if not use_vitokenizer:
        return text
    try:
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(text)
    except Exception:
        # fallback: no tokenizer installed
        return text


def load_use_vitokenizer(model_dir: Path) -> bool:
    cfg_path = model_dir / "train_config.json"
    if not cfg_path.exists():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return bool(cfg.get("tfidf", {}).get("use_vitokenizer", False))
    except Exception:
        return False


def load_thresholds(thresholds_json: str | None) -> Dict[str, float]:
    if thresholds_json is None:
        return {}
    obj = json.loads(Path(thresholds_json).read_text(encoding="utf-8"))
    out = {}
    for label, info in obj.items():
        if isinstance(info, dict) and "thr" in info:
            out[label] = float(info["thr"])
        elif isinstance(info, (int, float)):
            out[label] = float(info)
    return out


def get_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        return sigmoid(scores)
    return np.asarray(model.predict(X)).astype(np.float32)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--labelmap_path", required=True)

    # split_path: OPTIONAL (if not provided => eval ALL rows, suitable for Gold test set)
    ap.add_argument("--split_path", default=None)
    ap.add_argument("--split_name", default="test", choices=["train", "val", "test"])
    ap.add_argument("--precomputed_dir", default=None, help="Use cached features (from baseline_precompute_fold_features.py)")

    ap.add_argument("--thresholds_json", default=None)
    ap.add_argument("--threshold", type=float, default=0.5, help="fallback threshold if label missing")
    ap.add_argument("--run_name", default=None, help="Title to show on console/report header (e.g., 'VAL-SILVER | seed44 | Dataset_Full_VI')")
    ap.add_argument("--out_report", default=None)
    ap.add_argument("--out_metrics", default=None)
    return ap.parse_args()


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    model = joblib.load(model_dir / "model.pkl")
    vectorizer = joblib.load(model_dir / "vectorizer.pkl")

    label_names, label2id, _ = load_labelmap(Path(args.labelmap_path))
    num_labels = len(label_names)

    if args.precomputed_dir:
        if not args.split_path:
            raise SystemExit("precomputed_dir requires --split_path and --split_name")
        pre_dir = Path(args.precomputed_dir)
        X = load_npz(pre_dir / f"X_{args.split_name}.npz")
        y_true = np.load(pre_dir / f"Y_{args.split_name}.npy")
    else:
        rows = read_jsonl(Path(args.data_path))

        if args.split_path:
            idxs = load_split_indices(Path(args.split_path), args.split_name)
            subset = [rows[i] for i in idxs]
        else:
            subset = rows  # GOLD: evaluate all

        use_vitokenizer = load_use_vitokenizer(model_dir)
        texts = [maybe_tokenize_vi(r["text"], use_vitokenizer) for r in subset]
        X = vectorizer.transform(texts)
        seed_words_map = load_seed_words_from_model_dir(model_dir)
        if seed_words_enabled(seed_words_map):
            seed_feats = build_seed_features(texts, seed_words_map, label_names)
            X = append_seed_features(X, seed_feats)

        y_true = np.stack([to_multihot(r, label2id, num_labels) for r in subset], axis=0)
    probs = get_scores(model, X)

    thr_map = load_thresholds(args.thresholds_json)
    thr_vec = np.array([thr_map.get(name, args.threshold) for name in label_names], dtype=np.float32)
    y_pred = (probs >= thr_vec[None, :]).astype(np.int64)

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    data_tag = Path(args.data_path).stem  # Dataset_Full_VI (nếu file là Dataset_Full_VI.jsonl)
    mode = "ALL" if not args.split_path else args.split_name.upper()

    # Nếu user truyền run_name thì dùng, không thì tự tạo title mặc định
    title = args.run_name or f"{mode} | {data_tag}"

    print("\n" + "="*100)
    print(f"📊 Classification report (per label) | {title}")
    print("="*100)
    print(report)

    metrics: Dict[str, Any] = {}
    metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["precision_micro"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["recall_micro"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["exact_match"] = float(np.mean(np.all(y_true == y_pred, axis=1)))

    print("\n📌Summary metrics:")
    for k in ["f1_micro", "f1_macro", "f1_weighted", "precision_micro", "recall_micro", "exact_match"]:
        print(f"- {k}: {metrics[k]:.6f}")

    if args.out_report:
        Path(args.out_report).write_text(report, encoding="utf-8")
        print("Saved report:", args.out_report)

    if args.out_metrics:
        Path(args.out_metrics).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved metrics:", args.out_metrics)

    print("\n")

if __name__ == "__main__":
    main()
