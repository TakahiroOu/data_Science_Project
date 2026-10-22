"""Shared constants and data loading for analysis.py and tfidf.py."""
import json
import re
from pathlib import Path

import pandas as pd

# The keys used in the json-files. Each key holds a list of documents,
# and each document is a list of lemmatized tokens.
LABELS = ("positive", "neutral", "negative")

# lemmas, also used as search terms for the tweets
KEYWORDS = [
    "hallitus", "presidentti", "venäjä", "puolue",
    "ulkopolitiikka", "talouspolitiikka", "turvallisuuspolitiikka",
    "maahanmuuttopolitiikka", "perussuomalainen", "pääministeri",
    ]

URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_LETTER_RE = re.compile(r"[^a-zäöå\s]")  # keep a-z plus Finnish letters; adjust as needed
SPACE_RE = re.compile(r"\s+")
YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def load_data(data_dir="data", source=None):
    """Read all json-files in the folder (for example data/).

    Each file looks like:
        {"positive": [["hän", "nauttia", ...], [...]], "neutral": [...], "negative": [...]}

    Returns a DataFrame with one row per document: "text" (tokens joined with spaces),
    "label", "year" (from the file name) and "source".
    """
    files = sorted(Path(data_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files found in {Path(data_dir).resolve()}")

    docs, labels, years = [], [], []
    for path in files:
        match = YEAR_RE.search(path.stem)
        if match:
            year = int(match.group(1))
        else:
            year = None
            print(f"Warning: no year found in file name '{path.name}'")

        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for label in LABELS:
            for tokens in data.get(label, []):
                docs.append(" ".join(map(str, tokens)))
                labels.append(label)
                years.append(year)

    df = pd.DataFrame({"text": docs, "label": labels, "year": years})
    df["year"] = df["year"].astype("Int64")
    df["source"] = source or Path(data_dir).name
    print(f"[{df['source'].iloc[0]}] Loaded {len(files)} files, {len(df)} documents")
    print(df["label"].value_counts().to_string())
    return df
