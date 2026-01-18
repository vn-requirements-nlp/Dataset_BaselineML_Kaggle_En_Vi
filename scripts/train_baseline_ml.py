# scripts/train_baseline_ml.py
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

from scripts.make_splits_baseline import (
    read_jsonl, load_labelmap, to_multihot, safe_mkdir, load_split_indices
)


def build_model(algo: str, args) -> Any:
    algo = algo.lower()

    # Naive Bayes
    if algo == "nb":
        base = MultinomialNB(alpha=args.nb_alpha)
        return OneVsRestClassifier(base)

    # Logistic Regression
    if algo == "logreg":
        base = LogisticRegression(
            max_iter=args.logreg_max_iter,
            C=args.logreg_C,
            solver="liblinear" if args.logreg_solver == "liblinear" else "lbfgs",
        )
        return OneVsRestClassifier(base)

    # SVM (Linear)
    if algo == "linearsvm":
        base = LinearSVC(C=args.svm_C)
        return OneVsRestClassifier(base)

    # SVM (RBF)
    if algo == "svmrbf":
        base = SVC(kernel="rbf", C=args.svm_C, gamma=args.svm_gamma, probability=True)
        return OneVsRestClassifier(base)

    # Random Forest
    if algo == "rf":
        base = RandomForestClassifier(
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth if args.rf_max_depth > 0 else None,
            random_state=args.seed,
            n_jobs=args.rf_n_jobs,
        )
        return OneVsRestClassifier(base)
    
    # Decision Tree
    if algo == "dt":
        base = DecisionTreeClassifier(random_state=args.seed)
        return OneVsRestClassifier(base)

    # KNN
    if algo == "knn":
        # KNN chạy khá chậm với TF-IDF dimension lớn, nên cẩn thận
        base = KNeighborsClassifier(n_neighbors=5, n_jobs=args.rf_n_jobs)
        return OneVsRestClassifier(base)

    # Gradient Boosting (sklearn)
    if algo == "gb":
        base = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=args.seed
        )
        return OneVsRestClassifier(base)

    # AdaBoost
    if algo == "adaboost":
        base = AdaBoostClassifier(
            n_estimators=50, random_state=args.seed
        )
        return OneVsRestClassifier(base)

    # CatBoost
    if algo == "catboost":
        if CatBoostClassifier is None:
            raise ImportError("Please run: pip install catboost")
        base = CatBoostClassifier(
            iterations=100, learning_rate=0.1, depth=6, 
            verbose=0, allow_writing_files=False, random_state=args.seed
        )
        return OneVsRestClassifier(base)

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

    ap.add_argument("--algo", default="logreg", choices=["nb", "logreg", "linearsvm", "svmrbf", "rf", "dt", "knn", "gb", "adaboost", "catboost"])

    # TF-IDF params
    ap.add_argument("--ngram_min", type=int, default=1)
    ap.add_argument("--ngram_max", type=int, default=2)
    ap.add_argument("--max_features", type=int, default=50000)
    ap.add_argument("--min_df", type=int, default=1)
    ap.add_argument("--use_vitokenizer", action="store_true", help="Tokenize Vietnamese text with pyvi")

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

    return ap.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    safe_mkdir(out_dir)

    data_path = Path(args.data_path)
    labelmap_path = Path(args.labelmap_path)
    split_path = Path(args.split_path)

    rows = read_jsonl(data_path)
    label_names, label2id, id2label = load_labelmap(labelmap_path)
    num_labels = len(label_names)

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
    print("Train/Val/Test:", len(train_idx), len(val_idx), len(test_idx))
    print("Labels:", num_labels)

    # Fit vectorizer ONLY on TRAIN
    vectorizer = TfidfVectorizer(
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        min_df=args.min_df,
        lowercase=True,
    )
    X_train = vectorizer.fit_transform(X_train_texts)

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
    cfg = {
        "algo": args.algo,
        "seed": args.seed,
        "data_path": str(data_path),
        "labelmap_path": str(labelmap_path),
        "split_path": str(split_path),
        "tfidf": {
            "ngram_min": args.ngram_min,
            "ngram_max": args.ngram_max,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "use_vitokenizer": bool(args.use_vitokenizer),
        },
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
