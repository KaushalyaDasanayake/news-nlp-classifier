# spacy preprocess

from __future__ import annotations
import re
from functools import lru_cache


URL_RE = re.compile(r"(?i)\b(?:https?://\S+|www\.\S+)\b")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b")
NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")
TOKEN_RE = re.compile(r"<[A-Z]+>|\w+(?:'\w+)?")

# ----- Lazy-load spacy model -----


@lru_cache(maxsize=1)
def get_nlp():
    import spacy

    # disable components we don't need for speed
    return spacy.load("en_core_web_sm", disable=["ner", "parser"])


# ----- Precomplied regex patterns (faster + consistent) ----------


def preprocess(
    text: str | None,
    *,
    replace_numbers: bool = False,  # default False: numbers can be meaningful
    keep_punct: bool = False,
    use_spacy: bool = False,
    lemmatize: bool = False,
    remove_stopwords: bool = False,
) -> str:

    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # remove leading/trailing spaces
    text = text.strip()
    # convert to lowercase
    text = text.lower()
    # collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text)

    # replace emails with special token
    text = EMAIL_RE.sub(" <Email> ", text)
    # replace urls with special token
    text = URL_RE.sub(" <URL> ", text)

    if replace_numbers:
        text = NUMBER_RE.sub(" <NUM> ", text)

    # collapse spaces again (because replacements add spaces)
    text = re.sub(r"\s+", " ", text).strip()

    # tokenize
    if use_spacy:
        nlp = get_nlp()
        doc = nlp(text)

        tokens = []
        for tok in doc:
            if tok.is_space:
                continue
            if (not keep_punct) and tok.is_punct:
                continue
            # remove stopwords (is, to, the)
            if remove_stopwords and tok.is_stop:
                continue
            # tokens.append(tok.text)

            # lemmatize
            out = tok.lemma_ if lemmatize else tok.text
            tokens.append(out)

        return " ".join(tokens)

    if keep_punct:
        # split punctuation stays attached to words
        tokens = text.split()
        print("TOKENS (keep_punct=True):", tokens)
    else:
        # regex tokenizer: keep words + special tokens, drops punctuation
        tokens = TOKEN_RE.findall(text)
        print("TOKENS (keep_punct=False):", tokens)

    return " ".join(tokens)


def main() -> None:
    s = "I'm running to the store and I'm happy."
    print("spacy raw:       ", preprocess(s, use_spacy=True))
    print("spacy lemma:     ", preprocess(s, use_spacy=True, lemmatize=True))
    print("spacy no stops:  ", preprocess(s, use_spacy=True, lemmatize=True, remove_stopwords=True))


if __name__ == "__main__":
    main()
