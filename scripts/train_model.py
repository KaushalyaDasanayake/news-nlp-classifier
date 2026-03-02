# Entry point scripts for training/eval

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml

from newsclf.preprocessing.spacy_preprocess import preprocess_many

def loaf_cfg(path: str) -> dict:
    """
    Load YAML config into a python dict
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
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

    # verify preprocessing works ent-to-end
    sample_texts = df[text_col].astype(str).head(5).tolist()
    cleaned = preprocess_many(sample_texts, cfg)

    for i, (raw, clean) in enumerate(zip(sample_texts, cleaned), start=1):
        print(f"{i} Raw: {raw[:80]!r}")
        print(f"   Clean: {clean[:80]!r}")

    # labels sanity check
    y = df[label_col].head(5).tolist()
    print("\nSample labels:", y)

if __name__ == "__main__":
    main()

