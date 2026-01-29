# scripts/baseline_train_ml.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train traditional ML baselines for multi-label classification (paper-grade):
- Uses fixed stratified split indices (train/val/test) from split_seedXX.json
- Fit TF-IDF ONLY on TRAIN to avoid leakage
- Train multi-label classifier (OneVsRest)
- Save: vectorizer.pkl, model.pkl, labelmap.json snapshot, train_config.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier

from scripts.baseline_make_splits import (
    read_jsonl, load_labelmap, to_multihot, safe_mkdir, load_split_indices
)
from scripts.baseline_feature_utils import (
    load_seed_words, normalize_seed_words, build_seed_features,
    append_seed_features, seed_words_enabled
)


def build_model(algo: str, args) -> Any:
    algo = algo.lower()
    ovr_kwargs = {"n_jobs": args.ovr_n_jobs}

    # Naive Bayes
    if algo == "nb":
        base = MultinomialNB(alpha=args.nb_alpha)
        return OneVsRestClassifier(base, **ovr_kwargs)

    # Logistic Regression
    if algo == "logreg":
        base = LogisticRegression(
            max_iter=args.logreg_max_iter,
            C=args.logreg_C,
            solver="liblinear" if args.logreg_solver == "liblinear" else "lbfgs",
        )
        return OneVsRestClassifier(base, **ovr_kwargs)

    # SVM (Linear)
    if algo == "linearsvm":
        base = LinearSVC(C=args.svm_C)
        return OneVsRestClassifier(base, **ovr_kwargs)

    # SVM (RBF)
    if algo == "svmrbf":
        base = SVC(kernel="rbf", C=args.svm_C, gamma=args.svm_gamma, probability=False)
        return OneVsRestClassifier(base, n_jobs=1)

    # Random Forest
    if algo == "rf":
        base = RandomForestClassifier(
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth if args.rf_max_depth > 0 else None,
            random_state=args.seed,
            n_jobs=args.rf_n_jobs,
        )
        return OneVsRestClassifier(base, **ovr_kwargs)
    
    # Decision Tree
    if algo == "dt":
        base = DecisionTreeClassifier(random_state=args.seed)
        return OneVsRestClassifier(base, **ovr_kwargs)

    # KNN
    # KNN chạy khá chậm với TF-IDF dimension lớn, nên cẩn thận
    if algo == "knn3":
        base = KNeighborsClassifier(n_neighbors=3, n_jobs=args.rf_n_jobs)
        return OneVsRestClassifier(base, **ovr_kwargs)
    
    if algo == "knn5":
        base = KNeighborsClassifier(n_neighbors=5, n_jobs=args.rf_n_jobs)
        return OneVsRestClassifier(base, **ovr_kwargs)

    if algo == "knn7":
        base = KNeighborsClassifier(n_neighbors=7, n_jobs=args.rf_n_jobs)
        return OneVsRestClassifier(base, **ovr_kwargs)

    # Gradient Boosting (sklearn)
    if algo == "gb":
        base = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=args.seed
        )
        return OneVsRestClassifier(base, **ovr_kwargs)

    # AdaBoost
    if algo == "adaboost":
        base = AdaBoostClassifier(
            n_estimators=50, random_state=args.seed
        )
        return OneVsRestClassifier(base, **ovr_kwargs)

    # CatBoost
    if algo == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("Please run: pip install catboost")
        base = CatBoostClassifier(
            iterations=100, learning_rate=0.1, depth=6, 
            verbose=0, allow_writing_files=False, random_state=args.seed
        )
        return OneVsRestClassifier(base, **ovr_kwargs)

    raise ValueError(f"Unknown algo: {algo}")


def maybe_tokenize_vi(text: str, use_vitokenizer: bool) -> str:
    if not use_vitokenizer:
        return text
    try:
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize(text)
    except Exception:
        # fallback: no tokenizer
        return text


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True, help="e.g. data/Dataset_Full_VI.jsonl")
    ap.add_argument("--labelmap_path", required=True, help="e.g. data/labelmap_multilabel.json")
    ap.add_argument("--split_path", required=True, help="e.g. data/splits/split_seed42.json")
    ap.add_argument("--output_dir", required=True, help="e.g. models/baseline_vi/logreg/seed42")
    ap.add_argument("--precomputed_dir", default=None, help="Use cached features (from baseline_precompute_fold_features.py)")

    ap.add_argument(
        "--algo",
        default="logreg",
        choices=["nb", "logreg", "linearsvm", "svmrbf", "rf", "dt", "knn3", "knn5", "knn7", "gb", "adaboost", "catboost"],
    )

    # TF-IDF params
    ap.add_argument("--ngram_min", type=int, default=1)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--min_df", type=int, default=1)
    ap.add_argument("--use_vitokenizer", action="store_true", help="Tokenize Vietnamese text with pyvi")
    ap.add_argument("--seed_words_json", default=None, help="JSON file: {label: [seed_word, ...]}")

    # Seeds
    ap.add_argument("--seed", type=int, default=42)

    # Algo params
    ap.add_argument("--nb_alpha", type=float, default=1.0)

    ap.add_argument("--logreg_max_iter", type=int, default=2000)
    ap.add_argument("--logreg_C", type=float, default=1.0)
    ap.add_argument("--logreg_solver", type=str, default="lbfgs", choices=["lbfgs", "liblinear"])

    ap.add_argument("--svm_C", type=float, default=1.0)
    ap.add_argument("--svm_gamma", type=str, default="scale", help="Only for svmrbf")

    ap.add_argument("--rf_estimators", type=int, default=300)
    ap.add_argument("--rf_max_depth", type=int, default=0, help="0 means None")
    ap.add_argument("--rf_n_jobs", type=int, default=-1)
    ap.add_argument("--ovr_n_jobs", type=int, default=-1, help="Parallelize OneVsRest across labels")

    return ap.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    safe_mkdir(out_dir)

    data_path = Path(args.data_path)
    labelmap_path = Path(args.labelmap_path)
    split_path = Path(args.split_path)

    label_names, label2id, id2label = load_labelmap(labelmap_path)
    num_labels = len(label_names)

    seed_words_map = {}
    meta: Dict[str, Any] = {}
    use_precomputed = bool(args.precomputed_dir)

    if use_precomputed:
        pre_dir = Path(args.precomputed_dir)
        meta_path = pre_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        if isinstance(meta.get("label_names"), list) and meta.get("label_names") != label_names:
            print("[WARN] label_names mismatch between labelmap and precomputed meta.")

        if args.seed_words_json:
            print("[WARN] --seed_words_json ignored because --precomputed_dir is used.")

        seed_words_path = pre_dir / "seed_words.json"
        if seed_words_path.exists():
            try:
                seed_words_map = json.loads(seed_words_path.read_text(encoding="utf-8"))
            except Exception:
                seed_words_map = {}
            (out_dir / "seed_words.json").write_text(
                json.dumps(seed_words_map, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        elif isinstance(meta.get("seed_words", {}).get("label_to_words"), dict):
            seed_words_map = meta["seed_words"]["label_to_words"]

        vectorizer = joblib.load(pre_dir / "vectorizer.pkl")
        X_train = load_npz(pre_dir / "X_train.npz")
        Y_train = np.load(pre_dir / "Y_train.npy")
        n_train = int(meta.get("n_train", Y_train.shape[0]))
        n_val = int(meta.get("n_val", -1))
        n_test = int(meta.get("n_test", -1))

    else:
        rows = read_jsonl(data_path)

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

            # Save a normalized copy for inference
            (out_dir / "seed_words.json").write_text(
                json.dumps(seed_words_map, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        train_idx = load_split_indices(split_path, "train")
        val_idx = load_split_indices(split_path, "val")
        test_idx = load_split_indices(split_path, "test")

        # TRAIN subset
        train_rows = [rows[i] for i in train_idx]
        X_train_texts = [maybe_tokenize_vi(r["text"], args.use_vitokenizer) for r in train_rows]
        Y_train = np.stack([to_multihot(r, label2id, num_labels) for r in train_rows], axis=0)

    print(">>> Baseline ML training")
    print("Data:", data_path)
    print("Algo:", args.algo)
    if use_precomputed:
        print("Train/Val/Test:", n_train, n_val, n_test)
    else:
        print("Train/Val/Test:", len(train_idx), len(val_idx), len(test_idx))
    print("Labels:", num_labels)

    if not use_precomputed:
        # Fit vectorizer ONLY on TRAIN
        vectorizer = TfidfVectorizer(
            ngram_range=(args.ngram_min, args.ngram_max),
            max_features=args.max_features,
            min_df=args.min_df,
            lowercase=True,
        )
        X_train = vectorizer.fit_transform(X_train_texts)

        if seed_words_enabled(seed_words_map):
            seed_feats = build_seed_features(X_train_texts, seed_words_map, label_names)
            X_train = append_seed_features(X_train, seed_feats)

    model = build_model(args.algo, args)
    model.fit(X_train, Y_train)

    # Save artifacts
    joblib.dump(vectorizer, out_dir / "vectorizer.pkl")
    joblib.dump(model, out_dir / "model.pkl")

    # Save labelmap snapshot
    (out_dir / "labelmap.json").write_text(
        json.dumps(
            {"label_names": label_names, "label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Save config
    tfidf_cfg = {
        "ngram_min": args.ngram_min,
        "ngram_max": args.ngram_max,
        "max_features": args.max_features,
        "min_df": args.min_df,
        "use_vitokenizer": bool(args.use_vitokenizer),
    }
    if use_precomputed and isinstance(meta.get("tfidf"), dict):
        tfidf_cfg = meta["tfidf"]

    data_path_cfg = meta.get("data_path", str(data_path)) if use_precomputed else str(data_path)
    split_path_cfg = meta.get("split_path", str(split_path)) if use_precomputed else str(split_path)

    cfg = {
        "algo": args.algo,
        "seed": args.seed,
        "data_path": data_path_cfg,
        "labelmap_path": str(labelmap_path),
        "split_path": split_path_cfg,
        "precomputed_dir": args.precomputed_dir,
        "tfidf": tfidf_cfg,
        "algo_params": {
            "nb_alpha": args.nb_alpha,
            "logreg_max_iter": args.logreg_max_iter,
            "logreg_C": args.logreg_C,
            "logreg_solver": args.logreg_solver,
            "svm_C": args.svm_C,
            "svm_gamma": args.svm_gamma,
            "rf_estimators": args.rf_estimators,
            "rf_max_depth": args.rf_max_depth,
            "rf_n_jobs": args.rf_n_jobs,
            "ovr_n_jobs": args.ovr_n_jobs,
        },
        "seed_words": {
            "enabled": seed_words_enabled(seed_words_map),
            "source_path": args.seed_words_json,
            "label_to_words": seed_words_map,
            "feature_type": "presence",
        },
    }
    (out_dir / "train_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("✅ Saved:", out_dir)
    print("- vectorizer.pkl")
    print("- model.pkl")
    print("- thresholds.json (tune next)")
    print("🎉 Done!")


if __name__ == "__main__":
    main()
