from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(path: str = "configs/base.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])

    ensure_dir(processed_dir)
    ensure_dir(artifacts_dir)

    label_col = cfg["dataset"]["label_col"]
    seed = int(cfg["project"]["seed"])

    train_raw = pd.read_csv(raw_dir / "ag_news_train_raw.csv")
    test_df = pd.read_csv(raw_dir / "ag_news_test_raw.csv")

    train_df, val_df = train_test_split(
        train_raw,
        test_size=0.1,
        random_state=seed,
        stratify=train_raw[label_col],
    )

    train_df.to_csv(cfg["dataset"]["train_path"], index=False)
    val_df.to_csv(cfg["dataset"]["val_path"], index=False)
    test_df.to_csv(cfg["dataset"]["test_path"], index=False)

    label_map = {int(i): str(int(i)) for i in sorted(train_raw[label_col].unique())}
    with open(Path(cfg["paths"]["artifacts_dir"]) / "labels.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("Saved train/val/test splits and labels.json")


if __name__ == "__main__":
    main()
