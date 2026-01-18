# Software Requirements Classification (EN/VI) - Baseline ML

Phân loại nhãn đa nhãn (multi-label) cho yêu cầu phần mềm, hỗ trợ cả tiếng Việt và tiếng Anh. Dự án này tập trung vào các mô hình ML truyền thống với TF-IDF.

## Dataset & định dạng
Dữ liệu dạng JSONL (mỗi dòng 1 JSON):
- Trường bắt buộc: `text`
- Nhãn (multi-label): `labels` (list[str]) hoặc `label_ids` (list[int])

Tập tin hiện có:
- `data/Dataset_Full_VI.jsonl`
- `data/Dataset_Full_EN.jsonl`
- `data/PROMISE-relabeled-NICE_VI.jsonl`
- `data/PROMISE-relabeled-NICE_EN.jsonl`
- `data/labelmap_multilabel.json`
- `data/splits/split_seed42.json` (chia tập VI, tỷ lệ 80/10/10)

## Các baseline hiện có
- Naive Bayes (`nb`)
- Logistic Regression (`logreg`)
- Linear SVM (`linearsvm`)
- SVM RBF (`svmrbf`)
- Random Forest (`rf`)
- Decision Tree (`dt`)
- KNN (`knn`)
- Gradient Boosting (`gb`)
- AdaBoost (`adaboost`)
- CatBoost (`catboost`)

## Cài đặt
1. Tạo môi trường ảo
   `python -m venv .venv`
2. Kích hoạt
   - PowerShell: `.\.venv\Scripts\Activate.ps1`
   - cmd: `.\.venv\Scripts\activate`
3. Cài thư viện
   `pip install -r requirements.txt`

## Hướng dẫn train
Ví dụ train cho tập VI với Logistic Regression:
```bash
python scripts/train_baseline_ml.py ^
  --data_path data/Dataset_Full_VI.jsonl ^
  --labelmap_path data/labelmap_multilabel.json ^
  --split_path data/splits/split_seed42.json ^
  --output_dir models/baseline_vi/logreg/seed42 ^
  --algo logreg ^
  --use_vitokenizer
```

Artifact tạo ra trong `--output_dir`:
- `vectorizer.pkl`
- `model.pkl`
- `labelmap.json`
- `train_config.json`

## Tune thresholds (từ tập val)
```bash
python scripts/tune_thresholds_baseline.py ^
  --model_dir models/baseline_vi/logreg/seed42 ^
  --data_path data/Dataset_Full_VI.jsonl ^
  --labelmap_path data/labelmap_multilabel.json ^
  --split_path data/splits/split_seed42.json
```
Output: `thresholds.json` trong `--model_dir`.

## Đánh giá
```bash
python scripts/eval_baseline_report.py ^
  --model_dir models/baseline_vi/logreg/seed42 ^
  --data_path data/Dataset_Full_VI.jsonl ^
  --labelmap_path data/labelmap_multilabel.json ^
  --split_path data/splits/split_seed42.json ^
  --split_name test ^
  --thresholds_json models/baseline_vi/logreg/seed42/thresholds.json
```
Nếu muốn đánh giá toàn bộ tập (gold), bỏ qua `--split_path`.

## Dự đoán
Dự đoán 1 câu:
```bash
python scripts/predict_baseline.py ^
  --model_dir models/baseline_vi/logreg/seed42 ^
  --text "Hệ thống phải khôi phục trong 5 giây"
```

Dự đoán từ TXT và xuất CSV (Excel-friendly):
```bash
python scripts/predict_baseline.py ^
  --model_dir models/baseline_vi/logreg/seed42 ^
  --input_txt path/to/requirements.txt ^
  --output_csv outputs/pred.csv ^
  --include_probs ^
  --include_active_labels
```

## Ghi chú
- File split hiện tại tham chiếu đến tập VI. Nếu dùng tập EN, hãy tạo split tương ứng và cập nhật `--split_path`.
- `pyvi` được dùng để tokenize tiếng Việt; nếu không có, script sẽ tự động fallback.
