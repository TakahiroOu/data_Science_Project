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

# Yearly result tables and where the graphs are saved
NEWS_RESULTS = Path("results/news_keyword_yearly.csv")
TWITTER_RESULTS = Path("results/twitter_keyword_yearly.csv")
PARTY_RESULTS = Path("results/twitter_party_yearly.csv")
NEWS_PARTY_RESULTS = Path("results/news_party_yearly.csv")
GRAPH_DIR = Path("results/graphs")

# Search terms in the news results that differ from the keyword names
NEWS_KEYWORD_NAMES = {
    "maahanmuutto": "maahanmuuttopolitiikka",
    "perussuomalai": "perussuomalainen",
}
# The same mapping the other way: keyword -> search term used for the news
NEWS_SEARCH_TERMS = {name: term for term, name in NEWS_KEYWORD_NAMES.items()}

# Years covered by both the news and the tweets
YEARS = range(2013, 2023)


def load_data(data_dir="data", source=None, pattern="*.json"):
    """Read all json-files matching pattern in the folder (for example data/).

    Each file looks like:
        {"positive": [["hän", "nauttia", ...], [...]], "neutral": [...], "negative": [...]}

    Returns a DataFrame with one row per document: "text" (tokens joined with spaces),
    "label", "year" (from the file name) and "source".
    """
    files = sorted(Path(data_dir).glob(pattern))
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


def read_results(path, source):
    """One yearly results table with only the columns needed for plotting."""
    columns = ["keyword", "year", "n_docs", "positive", "neutral", "negative"]
    df = pd.read_csv(path, encoding="utf-8-sig")[columns].copy()
    df["source"] = source
    df["year"] = df["year"].astype(int)
    return df


def load_results():
    """Return (news + Twitter per keyword, news + Twitter per party)."""
    news = read_results(NEWS_RESULTS, "News")
    twitter = read_results(TWITTER_RESULTS, "Twitter")
    parties = read_results(PARTY_RESULTS, "Twitter")
    if NEWS_PARTY_RESULTS.exists():
        news_parties = read_results(NEWS_PARTY_RESULTS, "News")
        parties = pd.concat([news_parties, parties], ignore_index=True)

    # The news file also has an "ALL DOCUMENTS" baseline row per year,
    # which has no Twitter counterpart
    news = news[news["keyword"] != "ALL DOCUMENTS"]

    df = pd.concat([news, twitter], ignore_index=True)
    df["keyword"] = df["keyword"].str.lower().str.strip().replace(NEWS_KEYWORD_NAMES)

    # Only the topic keywords; parties are compared separately
    df = df[df["keyword"].isin(KEYWORDS)]
    return df, parties
