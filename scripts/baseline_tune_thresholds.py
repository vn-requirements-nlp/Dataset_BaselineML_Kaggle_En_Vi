# scripts/baseline_tune_thresholds.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from scipy.sparse import load_npz

from scripts.baseline_make_splits import (
    read_jsonl, load_labelmap, to_multihot, load_split_indices,
    sigmoid, best_threshold_for_label
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


def get_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    if hasattr(model, "decision_function"):
        return sigmoid(np.asarray(model.decision_function(X)))
    pred = model.predict(X)
    return np.asarray(pred).astype(np.float32)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Directory containing model.pkl + vectorizer.pkl")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--labelmap_path", required=True)
    ap.add_argument("--split_path", required=True)
    ap.add_argument("--split_name", default="val", choices=["train", "val", "test"])
    ap.add_argument("--precomputed_dir", default=None, help="Use cached features (from baseline_precompute_fold_features.py)")

    ap.add_argument("--thr_min", type=float, default=0.05)
    ap.add_argument("--thr_max", type=float, default=0.95)
    ap.add_argument("--thr_step", type=float, default=0.01)
    return ap.parse_args()


def main():
    args = parse_args()

    model_dir = Path(args.model_dir)
    model = joblib.load(model_dir / "model.pkl")

    label_names, label2id, _ = load_labelmap(Path(args.labelmap_path))
    num_labels = len(label_names)

    if args.precomputed_dir:
        pre_dir = Path(args.precomputed_dir)
        X = load_npz(pre_dir / f"X_{args.split_name}.npz")
        y_true = np.load(pre_dir / f"Y_{args.split_name}.npy")
    else:
        vectorizer = joblib.load(model_dir / "vectorizer.pkl")
        rows = read_jsonl(Path(args.data_path))

        idxs = load_split_indices(Path(args.split_path), args.split_name)
        subset = [rows[i] for i in idxs]

        use_vitokenizer = load_use_vitokenizer(model_dir)
        texts = [maybe_tokenize_vi(r["text"], use_vitokenizer) for r in subset]
        X = vectorizer.transform(texts)
        seed_words_map = load_seed_words_from_model_dir(model_dir)
        if seed_words_enabled(seed_words_map):
            seed_feats = build_seed_features(texts, seed_words_map, label_names)
            X = append_seed_features(X, seed_feats)

        y_true = np.stack([to_multihot(r, label2id, num_labels) for r in subset], axis=0)
    probs = get_scores(model, X)

    if probs.shape != y_true.shape:
        raise RuntimeError(f"Score shape mismatch: probs={probs.shape} vs y_true={y_true.shape}")

    grid = np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step)

    out: Dict[str, Any] = {}
    f1s = []
    for j, name in enumerate(label_names):
        best = best_threshold_for_label(y_true[:, j], probs[:, j], grid)
        out[name] = best
        f1s.append(best["f1"])

    out_path = model_dir / "thresholds.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", out_path)
    print("Mean per-label F1:", float(np.mean(f1s)))


if __name__ == "__main__":
    main()
