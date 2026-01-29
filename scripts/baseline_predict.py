# scripts/baseline_predict.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
import numpy as np

from scripts.baseline_make_splits import load_labelmap, sigmoid
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


def load_thresholds(thresholds_json: Optional[str]) -> Dict[str, float]:
    if not thresholds_json:
        return {}
    obj = json.loads(Path(thresholds_json).read_text(encoding="utf-8"))
    out: Dict[str, float] = {}
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
        return sigmoid(np.asarray(model.decision_function(X)))
    return np.asarray(model.predict(X)).astype(np.float32)


def read_txt_lines(path: str) -> List[str]:
    # Accept UTF-8 (with/without BOM). Also handles Windows newlines.
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def write_csv_utf8_sig(path: str, fieldnames: List[str], rows: List[Dict[str, object]]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Directory with model.pkl + vectorizer.pkl + labelmap.json")
    ap.add_argument("--labelmap_path", default=None, help="If not provided, will use <model_dir>/labelmap.json")
    ap.add_argument("--thresholds_json", default=None, help="If not provided, will use <model_dir>/thresholds.json if exists")
    ap.add_argument("--threshold", type=float, default=0.5, help="fallback threshold")

    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="Single input text")
    g.add_argument("--input_jsonl", type=str, help="JSONL with {'text': .} per line")
    g.add_argument("--input_txt", type=str, help="TXT (one requirement per line)")

    # old output (keep)
    ap.add_argument("--output_jsonl", type=str, default=None, help="Write predictions to JSONL")

    # new output like transformer_predict.py
    ap.add_argument("--output_csv", type=str, default=None, help="Write predictions to CSV UTF-8 with BOM (Excel-friendly)")
    ap.add_argument("--include_probs", action="store_true", help="Add per-label probability columns (suffix: __prob)")
    ap.add_argument("--include_active_labels", action="store_true", help="Add ActiveLabels column (labels predicted = 1)")

    ap.add_argument("--print_all", action="store_true", help="Print all label probs (single --text mode)")
    return ap.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    model = joblib.load(model_dir / "model.pkl")
    vectorizer = joblib.load(model_dir / "vectorizer.pkl")

    use_vitokenizer = load_use_vitokenizer(model_dir)

    labelmap_path = Path(args.labelmap_path) if args.labelmap_path else (model_dir / "labelmap.json")
    label_names, label2id, _ = load_labelmap(labelmap_path)
    num_labels = len(label_names)
    seed_words_map = load_seed_words_from_model_dir(model_dir)
    use_seed_words = seed_words_enabled(seed_words_map)

    thr_path = args.thresholds_json
    if thr_path is None:
        cand = model_dir / "thresholds.json"
        if cand.exists():
            thr_path = str(cand)

    thr_map = load_thresholds(thr_path)
    thr_vec = np.array([thr_map.get(name, args.threshold) for name in label_names], dtype=np.float32)

    def predict_one(t_raw: str) -> Dict[str, Any]:
        t = maybe_tokenize_vi(t_raw, use_vitokenizer)
        X = vectorizer.transform([t])
        if use_seed_words:
            seed_feats = build_seed_features([t], seed_words_map, label_names)
            X = append_seed_features(X, seed_feats)
        probs = get_scores(model, X)[0]  # (num_labels,)
        pred01 = (probs >= thr_vec).astype(int)

        active = []
        active_ids = []
        for j in range(num_labels):
            if pred01[j] == 1:
                active.append(label_names[j])
                active_ids.append(j)

        return {
            "text": t_raw,
            "pred01": pred01.tolist(),          # NEW: for CSV 0/1 columns
            "pred_labels": active,
            "pred_label_ids": active_ids,
            "probs": probs.tolist(),
        }

    outputs: List[Dict[str, Any]] = []

    if args.text is not None:
        out = predict_one(args.text)
        outputs.append(out)

        print("\nActive labels:")
        if not out["pred_labels"]:
            print("(none)")
        else:
            for lab in out["pred_labels"]:
                j = label2id[lab]
                print(f"- {lab}: {out['probs'][j]:.4f} (thr={thr_vec[j]:.2f})")

        if args.print_all:
            print("\nAll labels probs:")
            for j, lab in enumerate(label_names):
                print(f"- {lab}: {out['probs'][j]:.4f} (thr={thr_vec[j]:.2f})")

    elif args.input_jsonl is not None:
        in_path = Path(args.input_jsonl)
        with in_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                out = predict_one(obj["text"])
                outputs.append(out)
        print(f"Predicted {len(outputs)} lines from {in_path}")

    else:
        # args.input_txt
        texts = read_txt_lines(args.input_txt)
        if not texts:
            raise SystemExit("input_txt không có dòng nào (hoặc toàn dòng trống).")
        for t in texts:
            outputs.append(predict_one(t))
        print(f"Predicted {len(outputs)} lines from {args.input_txt}")

    # old JSONL output (keep)
    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as w:
            for o in outputs:
                w.write(json.dumps(o, ensure_ascii=False) + "\n")
        print("Saved:", out_path)

    # new CSV output
    if args.output_csv:
        rows: List[Dict[str, object]] = []
        for o in outputs:
            row: Dict[str, object] = {"RequirementText": o["text"]}
            for lab, v01 in zip(label_names, o["pred01"]):
                row[lab] = int(v01)

            if args.include_probs:
                for lab, pv in zip(label_names, o["probs"]):
                    row[f"{lab}__prob"] = float(pv)

            if args.include_active_labels:
                row["ActiveLabels"] = "; ".join(o["pred_labels"])

            rows.append(row)

        fieldnames = ["RequirementText"] + label_names
        if args.include_probs:
            fieldnames += [f"{lab}__prob" for lab in label_names]
        if args.include_active_labels:
            fieldnames += ["ActiveLabels"]

        write_csv_utf8_sig(args.output_csv, fieldnames, rows)
        print("Saved:", args.output_csv)

    print(f"use_vitokenizer(auto from train_config): {use_vitokenizer} | labels={num_labels}")


if __name__ == "__main__":
    main()
