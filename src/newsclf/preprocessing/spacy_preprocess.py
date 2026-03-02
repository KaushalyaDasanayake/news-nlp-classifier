# spacy preprocess

from __future__ import annotations
import re
import unicodedata
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Optional, Iterable


# --- Precompiled regex patterns (fast + consistent) ---
URL_RE = re.compile(r"(?i)\b(?:https?://\S+|www\.\S+)\b")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b")
NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")

# keep placeholders like <URL> plus normal words (don't)
TOKEN_RE = re.compile(r"<[A-Za-z]+>|\w+(?:'\w+)?")

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


# hanle None/none-string
def preprocess_one(text: Optional[str], cfg: PreprocessConfig, *, cfg_yaml: dict | None = None) -> str:
    """
    Preprocess one text into a normalized string.
    """
    
    # safe input handling
    if text is None:
        return ""
    
    if not isinstance(text, str):
        text = str(text)

    # basic cleaning
    text = _basic_clean(text, cfg)

    if not text:
        return ""
    
    # spaCy branch
    if cfg.use_spacy:
        # protect placeholders before spaCy
        protected = text
        for k, v in SPECIAL_TOKENS.items():
            protected = protected.replace(k, v)

        # get model settings
        model_name = "en_core_web_sm"
        disable = ("ner", "parser")

        if cfg_yaml is not None:
            sp = cfg_yaml.get("spacy", {})
            model_name = sp.get("model", model_name)
            disable = tuple(sp.get("disable", list(disable)))

        nlp = get_nlp(model_name, disable)
        doc = nlp(protected)

        tokens: list[str] = []

        for tok in doc:
            if tok.is_space:
                continue

            if (not cfg.keep_punct) and tok.is_punct:
                continue

            if cfg.remove_stopwords and tok.is_stop:
                continue

            raw = tok.text.strip()

            # restore placeholders first
            upper_raw = raw.upper()
            restored = REVERSE_SPECIAL_TOKENS.get(upper_raw)
            if restored is not None:
                tokens.append(restored)
                continue

            # normal token processing
            out = tok.lemma_ if cfg.lemmatize else raw
            out = out.strip()

            if out:
                tokens.append(out)

        return " ".join(tokens)
    
    # regex fallback branch
    if cfg.keep_punct:
        tokens = text.split()
    else:
        tokens = TOKEN_RE.findall(text)

    # Normalize special tokens to uppercase form
    normalized_tokens = []
    for tok in tokens:
        if tok.startswith("<") and tok.endswith(">"):
            normalized_tokens.append(tok.upper())
        else:
            normalized_tokens.append(tok)

    return " ".join(normalized_tokens)

# preprocessing for a batch of texts
def preprocess_many(texts: Iterable[Optional[str]], cfg_yaml: dict) -> list[str]:
    """
    Fast preprocessing for a batch of texts.
    Uses spaCy nlp.pipe() when spaCy is enabled.
    """
    cfg = preprocess_config(cfg_yaml)
    sp = cfg_yaml.get("spacy", {})

    model_name = sp.get("model", "en_core_web_sm")
    disable = tuple(sp.get("disable", ["ner", "parser"]))
    batch_size = int(sp.get("batch_size", 256))
    n_process = int(sp.get("n_process", 1))

    # basic clean
    cleaned: list[str] = []
    for t in texts:
        if t is None:
            cleaned.append("")
        elif not isinstance(t, str):
            cleaned.append(_basic_clean(str(t), cfg))
        else:
            cleaned.append(_basic_clean(t, cfg))
    
    # if spaCy not enabled make fallback
    if not cfg.use_spacy:
        return [preprocess_one(t, cfg) for t in cleaned]
    
    # protect placeholders
    protected_texts: list[str] = []
    for t in cleaned:
        pt = t
        for k, v in SPECIAL_TOKENS.items():
            pt = pt.replace(k, v)
        protected_texts.append(pt)

    # fast spaCy pipe
    nlp = get_nlp(model_name, disable)
    outputs: list[str] = []

    for doc in nlp.pipe(protected_texts, batch_size=batch_size, n_process=n_process):
        tokens: list[str] = []

        for tok in doc:
            if tok.is_space:
                continue
            if (not cfg.keep_punct) and tok.is_punct:
                continue
            if cfg.remove_stopwords and tok.is_stop:
                continue

            raw = tok.text.strip()

            # restore placeholders first
            upper_raw = raw.upper()
            restored = REVERSE_SPECIAL_TOKENS.get(upper_raw)
            if restored is not None:
                tokens.append(restored)
                continue

            # normal token processing
            out = tok.lemma_ if cfg.lemmatize else raw
            out = out.strip()

            if out:
                tokens.append(out)

        outputs.append(" ".join(tokens))
    return outputs