# Download dataset

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset

def load_config(path: str = "configs/base.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(Path(cfg["paths"]["raw_dir"]))

    raw_dir = Path(cfg["paths"]["raw_dir"])
    ensure_dir(raw_dir)

    ds = load_dataset("ag_news")

    train_df = pd.DataFrame(ds["train"])[["text", "label"]]
    test_df = pd.DataFrame(ds["test"])[["text", "label"]]

    train_df.to_csv(raw_dir / "ag_news_train_raw.csv", index=False)
    test_df.to_csv(raw_dir / "ag_news_test_raw.csv", index=False)
    
    print("Saved raw AG News dataset to data/raw/")



if __name__ == "__main__":
    main()