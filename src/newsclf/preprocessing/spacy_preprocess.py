# spacy preprocess

from __future__ import annotations
import re
import unicodedata
from functools import lru_cache
from dataclasses import dataclass
from typing import Any


# --- Precompiled regex patterns (fast + consistent) ---
URL_RE = re.compile(r"(?i)\b(?:https?://\S+|www\.\S+)\b")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b")
NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")

# keep placeholders like <URL> plus normal words (don't)
TOKEN_RE = re.compile(r"<[A-Z]+>|\w+(?:'\w+)?")

# --- Special token protection (so spaCy won't split <URL> into <, URL, >)
SPECIAL_TOKENS = {
    "<EMAIL>": "SPECIAL_EMAIL",
    "<URL>": "SPECIAL_URL",
    "<NUM>": "SPECIAL_NUM"
}
REVERSE_SPECIAL_TOKENS = {v: k for k, v in SPECIAL_TOKENS.items()}

@dataclass(frozen=True)
class PreprocessConfig:
    lowercase: bool = True
    replace_urls: bool = True
    replace_emails: bool = True
    replace_numbers: bool = False 
    keep_punct: bool = False
    lemmatize: bool = False
    remove_stopwords: bool = False
    use_spacy: bool = False


def preprocess_config(cfg:dict) -> PreprocessConfig:
    """
    Convert config/base.yaml dict -> PreprocessConfig.
    """
    p = cfg.get("preprocessing", {})

    # if lemmatize/stopwords enabled, spaCy is needed
    use_spacy = bool(p.get("lemmatize", False) or p.get("remove_stopwords", False))

    if "use_spacy" in p:
        use_spacy = bool(p["use_spacy"])

    return PreprocessConfig(
        lowercase=bool(p.get("lowercase", True)),
        replace_urls=bool(p.get("replace_urls", True)),
        replace_emails=bool(p.get("replace_emails", True)),
        replace_numbers=bool(p.get("replace_numbers", False)),
        keep_punct=bool(p.get("keep_punct", False)),
        lemmatize=bool(p.get("lemmatize", False)),
        remove_stopwords=bool(p.get("remove_stopwords", False)),
        use_spacy=use_spacy
    )


@lru_cache
def get_nlp(model_name: str, disable: tuple[str, ...]):
    """
    Load spaCy model once per (model_name, disable list)
    Cached so repeated calls are fast.
    """
    import spacy
    return spacy.load(model_name, disable=list(disable))


# deterministic cleanup
def _basic_clean(text: str, cfg: PreprocessConfig) -> str:
    """
    Cheap, deterministic cleanup before tokenization.
    """
    # trim spaces at both ends
    text = text.strip()

    # normalize unicode 
    text = unicodedata.normalize("NFKC", text)

    # lowercase 
    if cfg.lowercase:
        text = text.lower()

    # collapse whitespace 
    text = re.sub(r"\s+", " ", text)

    # replace patterns with special tokens 
    if cfg.replace_emails:
        text = EMAIL_RE.sub(" <EMAIL> ", text)

    if cfg.replace_urls:
        text = URL_RE.sub(" <URL> ", text)

    if cfg.replace_numbers:
        text = NUMBER_RE.sub(" <NUM> ", text)

    # clean extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    cfg = PreprocessConfig(replace_urls=True, replace_emails=True, replace_numbers=True, lowercase=True)
    s = " Email: Test@Example.com   Visit https://example.com \n Price 1,200.50 "
    print(_basic_clean(s, cfg))