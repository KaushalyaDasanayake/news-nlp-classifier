from __future__ import annotations

import yaml

from newsclf.preprocessing.spacy_preprocess import (
    PreprocessConfig,
    preprocess_many,
    preprocess_one,
)


def load_cfg() -> dict:
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_url_replacement():
    cfg = load_cfg()
    cfg["preprocessing"]["replace_urls"] = True
    cfg["preprocessing"]["replace_emails"] = False
    cfg["preprocessing"]["replace_numbers"] = False
    cfg["preprocessing"]["lemmatize"] = False
    cfg["preprocessing"]["remove_stopwords"] = False

    out = preprocess_many(["Visit https://example.com now"], cfg)[0]
    assert "<URL>" in out


def test_empty_input_returns_empty_string():
    pcfg = PreprocessConfig()
    assert preprocess_one("", pcfg) == ""
    assert preprocess_one(None, pcfg) == ""


def test_deterministic():
    cfg = load_cfg()
    cfg["preprocessing"]["replace_urls"] = True
    cfg["preprocessing"]["replace_emails"] = True

    text = "Email me at testme@example.com and visit https://example.com"
    out1 = preprocess_many([text], cfg)[0]
    out2 = preprocess_many([text], cfg)[0]
    assert out1 == out2
