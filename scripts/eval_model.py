# load config + load artifacts + run predictions

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
import joblib
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from newsclf.preprocessing.spacy_preprocess import preprocess_many


def load_cfg(path: str) -> dict:
    """
    Load YAML config into a dict.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    # Paths from config
    test_path = Path(cfg["dataset"]["test_path"])
    text_col = cfg["dataset"]["text_col"]
    label_col = cfg["dataset"]["label_col"]
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])

    # check required files exist
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    vec_path = artifacts_dir / "vectorizer.joblib"
    model_path = artifacts_dir / "model.joblib"
    labels_path = artifacts_dir / "labels.json"

    for p in [vec_path, model_path, labels_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing artifacts: {p}\n"
                "Run training first: python scripts/train_model.py --config configs/base.yaml"
            )

    # load data
    df = pd.read_csv(test_path)
    X_raw = df[text_col].astype(str).tolist()
    y_true = df[label_col].astype(int).tolist()

    # preprocess + vectorize
    X_clean = preprocess_many(X_raw, cfg)
    vec = joblib.load(vec_path)
    clf = joblib.load(model_path)

    # eval uses transform(), not fit_transform()
    X_vec = vec.transform(X_clean)

    # predict
    y_pred = clf.predict(X_vec)

    # metrics
    acc = float(accuracy_score(y_true, y_pred))

    # classification_report gives per-class precision/recall/F1 + macro avg
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # confusion matrix: rows=true labels, cols=pred labels
    cm = confusion_matrix(y_true, y_pred)

    # load label list
    with open(labels_path, "r", encoding="utf-8") as f:
        label_info = json.load(f)
    labels = label_info["labels"]

    # save outputs
    results_dir = Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    stamp = date.today().isoformat()
    eval_json_path = results_dir / f"eval_{stamp}.json"
    cm_csv_path = results_dir / f"confusion_matrix_{stamp}.csv"

    payload = {
        "date": stamp,
        "test_path": str(test_path),
        "n_test": len(df),
        "accuracy": acc,
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "per_class": report_dict,  # includes each class + macro/weighted averages
        "labels": labels,
    }

    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # save confusion matrix as CSV with headers
    cm_df = pd.DataFrame(
        cm, index=[f"true_{i}" for i in labels], columns=[f"pred_{i}" for i in labels]
    )
    cm_df.to_csv(cm_csv_path, index=True)

    print("accuracy:", acc)
    print("macro f1:", payload["macro_f1"])
    print("saved:", eval_json_path)
    print("saved:", cm_csv_path)


if __name__ == "__main__":
    main()
