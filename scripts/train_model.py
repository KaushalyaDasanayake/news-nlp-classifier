# Entry point scripts for training/eval

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from newsclf.preprocessing.spacy_preprocess import preprocess_many


def loaf_cfg(path: str) -> dict:
    """
    Load YAML config into a python dict
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# create unique fingerprint
def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
        return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = loaf_cfg(args.config)

    train_path = cfg["dataset"]["train_path"]
    text_col = cfg["dataset"]["text_col"]
    label_col = cfg["dataset"]["label_col"]

    train_file = Path(train_path)
    if not train_file.exists():
        raise FileNotFoundError(
            f"Train file not found: {train_file}\n"
            "Run: python scripts/download_data.py and python scripts/make_splits.py first."
        )

    df = pd.read_csv(train_file)
    print("loaded:", train_file)
    print("rows:", len(df))
    print("columns:", list(df.columns))

    X_raw = df[text_col].astype(str).tolist()
    y = df[label_col].astype(int).tolist()

    X = preprocess_many(X_raw, cfg)
    print("Preprocessing done")

    # TF-IDF + LogReg
    tfidf_cfg = cfg["model"]["tfidf"]
    vec = TfidfVectorizer(
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        min_df=int(tfidf_cfg["min_df"]),
        max_df=float(tfidf_cfg["max_df"]),
        max_features=int(tfidf_cfg["max_features"]),
    )

    X_vec = vec.fit_transform(X)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_vec, y)

    print("Training done")

    art_dir = Path(cfg["paths"]["artifacts_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vec, art_dir / "vectorizer.joblib")
    joblib.dump(clf, art_dir / "model.joblib")

    labels = sorted(set(y))
    with open(art_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"labels": labels}, f, indent=2)

    meta = {
        "timestamp": int(time.time()),
        "config_path": args.config,
        "config_sha256": sha256_of_file(args.config),
        "train_path": train_path,
        "train_rows": len(df),
        "model_type": "tfidf_logreg",
    }
    with open(art_dir / "train_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved artifacts to:", art_dir)


if __name__ == "__main__":
    main()
