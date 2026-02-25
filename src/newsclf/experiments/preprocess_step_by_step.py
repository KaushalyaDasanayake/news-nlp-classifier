# spacy preprocess

from __future__ import annotations
import re

URL_RE = re.compile(r"(?i)\b(?:https?://\S+|www\.\S+)\b")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b")
NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")
TOKEN_RE = re.compile(r"<[A-Z]+>|\w+(?:'\w+)?")

# ----- Precomplied regex patterns (faster + consistent) ----------

def preprocess(
        text: str | None,
        *,
        replace_numbers: bool = False,
        keep_punct: bool = False
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
    samples = [
        "Hello, world! This is NLP.",
        "Email me at test@example.com!!!",
        "Price is 1,200.50 USD.",
    ]
    for s in samples:
        # out = preprocess(s)
        # print(f"IN: {repr(s)} -> OUT: {repr(out)}")

        out1 = preprocess(s, replace_numbers=False, keep_punct=False)
        out2 = preprocess(s, replace_numbers=True, keep_punct=True)

        print(f"IN: {repr(s)}")
        print(f"  no_punct:  {repr(out1)}")
        print(f"  keep_punct: {repr(out2)}")

if __name__ == "__main__":
    main()