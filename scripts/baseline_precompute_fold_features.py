# scripts/baseline_precompute_fold_features.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.baseline_make_splits import (
    read_jsonl, load_labelmap, to_multihot, safe_mkdir, load_split_indices
)
from scripts.baseline_feature_utils import (
    load_seed_words, normalize_seed_words, build_seed_features,
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--labelmap_path", required=True)
    ap.add_argument("--split_path", required=True)
    ap.add_argument("--output_dir", required=True, help="e.g. features/seed42/fold0")

    # TF-IDF params
    ap.add_argument("--ngram_min", type=int, default=1)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--min_df", type=int, default=1)
    ap.add_argument("--use_vitokenizer", action="store_true")
    ap.add_argument("--seed_words_json", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    safe_mkdir(out_dir)

    rows = read_jsonl(Path(args.data_path))
    label_names, label2id, _ = load_labelmap(Path(args.labelmap_path))
    num_labels = len(label_names)

    train_idx = load_split_indices(Path(args.split_path), "train")
    val_idx = load_split_indices(Path(args.split_path), "val")
    test_idx = load_split_indices(Path(args.split_path), "test")

    def build_subset(idxs: List[int]) -> List[Dict[str, Any]]:
        return [rows[i] for i in idxs]

    train_rows = build_subset(train_idx)
    val_rows = build_subset(val_idx)
    test_rows = build_subset(test_idx)

    X_train_texts = [maybe_tokenize_vi(r["text"], args.use_vitokenizer) for r in train_rows]
    X_val_texts = [maybe_tokenize_vi(r["text"], args.use_vitokenizer) for r in val_rows]
    X_test_texts = [maybe_tokenize_vi(r["text"], args.use_vitokenizer) for r in test_rows]

    Y_train = np.stack([to_multihot(r, label2id, num_labels) for r in train_rows], axis=0)
    Y_val = np.stack([to_multihot(r, label2id, num_labels) for r in val_rows], axis=0)
    Y_test = np.stack([to_multihot(r, label2id, num_labels) for r in test_rows], axis=0)

    vectorizer = TfidfVectorizer(
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
        lowercase=True,
    )
    X_train = vectorizer.fit_transform(X_train_texts)
    X_val = vectorizer.transform(X_val_texts)
    X_test = vectorizer.transform(X_test_texts)

    seed_words_map = {}
    if args.seed_words_json:
        seed_words_map, unknown = load_seed_words(Path(args.seed_words_json), label_names)
        if unknown:
            print("[WARN] seed_words_json has unknown label keys:", ", ".join(unknown))
        if args.use_vitokenizer:
            seed_words_map = normalize_seed_words(
                seed_words_map, tokenizer=lambda t: maybe_tokenize_vi(t, True)
            )
        else:
            seed_words_map = normalize_seed_words(seed_words_map, tokenizer=None)

        (out_dir / "seed_words.json").write_text(
            json.dumps(seed_words_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if seed_words_enabled(seed_words_map):
        X_train = append_seed_features(
            X_train, build_seed_features(X_train_texts, seed_words_map, label_names)
        )
        X_val = append_seed_features(
            X_val, build_seed_features(X_val_texts, seed_words_map, label_names)
        )
        X_test = append_seed_features(
            X_test, build_seed_features(X_test_texts, seed_words_map, label_names)
        )

    joblib.dump(vectorizer, out_dir / "vectorizer.pkl")
    save_npz(out_dir / "X_train.npz", X_train)
    save_npz(out_dir / "X_val.npz", X_val)
    save_npz(out_dir / "X_test.npz", X_test)
    np.save(out_dir / "Y_train.npy", Y_train)
    np.save(out_dir / "Y_val.npy", Y_val)
    np.save(out_dir / "Y_test.npy", Y_test)

    meta = {
        "data_path": str(args.data_path),
        "labelmap_path": str(args.labelmap_path),
        "split_path": str(args.split_path),
        "label_names": label_names,
        "num_labels": num_labels,
        "n_train": int(Y_train.shape[0]),
        "n_val": int(Y_val.shape[0]),
        "n_test": int(Y_test.shape[0]),
        "tfidf": {
            "ngram_min": args.ngram_min,
            "ngram_max": args.ngram_max,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "use_vitokenizer": bool(args.use_vitokenizer),
        },
        "seed_words": {
            "enabled": seed_words_enabled(seed_words_map),
            "source_path": args.seed_words_json,
            "label_to_words": seed_words_map,
            "feature_type": "presence",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved:", out_dir)
    print("- vectorizer.pkl")
    print("- X_train.npz / X_val.npz / X_test.npz")
    print("- Y_train.npy / Y_val.npy / Y_test.npy")
    print("- meta.json")


if __name__ == "__main__":
    main()
