import json
import re
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

DATA_DIR = Path("data")

# The keys used in the json-files. Each key holds a list of documents,
# and each document is a list of lemmatized tokens.
LABELS = ("positive", "neutral", "negative")


try:
    FINNISH_STOP_WORDS = set(stopwords.words("finnish"))
except LookupError:
    nltk.download("stopwords")
    FINNISH_STOP_WORDS = set(stopwords.words("finnish"))

URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_LETTER_RE = re.compile(r"[^a-zäöå\s]")  # keep a-z plus Finnish letters; adjust as needed
SPACE_RE = re.compile(r"\s+")
YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")


def load_data(data_dir=DATA_DIR):
    """Read all json-files in the folder (for example data/).

    Each file looks like:
        {"positive": [["hän", "nauttia", ...], [...]], "neutral": [...], "negative": [...]}

    Returns a DataFrame with one row per document: "text" (tokens joined with spaces) and "label".
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
    print(f"Loaded {len(files)} files, {len(df)} documents")
    print(df["label"].value_counts().to_string())
    print(df.head())
    return df


def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = NON_LETTER_RE.sub(" ", text)  # drops digits and punctuation
    tokens = [t for t in SPACE_RE.split(text) if len(t) > 2 and t not in FINNISH_STOP_WORDS]
    return " ".join(tokens)


def keyword_analysis(lemma_docs, labels, K=10, stop_words=None):
    labels = np.array(labels)
    masks = {name: labels == name for name in LABELS}

    # 1) TF-IDF per label (IDF computed over the whole corpus)
    tfidf_vec = TfidfVectorizer(max_df=0.5, min_df=2, stop_words=stop_words)
    X_tfidf = tfidf_vec.fit_transform(lemma_docs)
    tfidf_vocab = tfidf_vec.get_feature_names_out()

    tfidf_results = {}
    for name, mask in masks.items():
        if mask.sum() == 0:  # no documents with this label
            tfidf_results[name] = []
            continue
        mean_scores = np.asarray(X_tfidf[mask].mean(axis=0)).ravel()
        top = mean_scores.argsort()[::-1][:K]
        tfidf_results[name] = [
            (tfidf_vocab[i], float(mean_scores[i]))
            for i in top if mean_scores[i] > 0
        ]

    # 2) Most frequent words overall, with a count per label
    count_vec = CountVectorizer(min_df=2, stop_words=stop_words)
    X_count = count_vec.fit_transform(lemma_docs)
    vocab = count_vec.get_feature_names_out()

    total = np.asarray(X_count.sum(axis=0)).ravel()
    per_label = {
        name: np.asarray(X_count[mask].sum(axis=0)).ravel()
        for name, mask in masks.items()
    }

    freq_results = [
        {
            "word": vocab[i],
            "total": int(total[i]),
            **{name: int(counts[i]) for name, counts in per_label.items()},
        }
        for i in total.argsort()[::-1][:K]
    ]

    # Print
    for name, pairs in tfidf_results.items():
        print(f"\nTF-IDF top {K} ({name}):")
        for word, score in pairs:
            print(f"  {word:20s} {score:.4f}")

    counts_str = ", ".join(f"{mask.sum()} {name}" for name, mask in masks.items())
    print(f"\nLeads: {counts_str}")

    print(f"\nTop {K} most frequent words:")
    print(f"  {'word':20s} {'total':>6s}" + "".join(f" {name[:3]:>6s}" for name in LABELS))
    for r in freq_results:
        print(f"  {r['word']:20s} {r['total']:6d}" + "".join(f" {r[name]:6d}" for name in LABELS))

    return {"tfidf": tfidf_results, "frequent": freq_results}


def top_words_overall(lemma_docs, K=10, stop_words=None):
    """Top K words across all documents, no labels."""
    # By mean TF-IDF score
    tfidf_vec = TfidfVectorizer(max_df=0.5, min_df=2, stop_words=stop_words)
    X_tfidf = tfidf_vec.fit_transform(lemma_docs)
    tfidf_vocab = tfidf_vec.get_feature_names_out()
    mean_scores = np.asarray(X_tfidf.mean(axis=0)).ravel()
    top_tfidf = [
        (tfidf_vocab[i], float(mean_scores[i]))
        for i in mean_scores.argsort()[::-1][:K]
    ]

    # By raw frequency
    count_vec = CountVectorizer(min_df=2, stop_words=stop_words)
    X_count = count_vec.fit_transform(lemma_docs)
    vocab = count_vec.get_feature_names_out()
    total = np.asarray(X_count.sum(axis=0)).ravel()
    top_freq = [(vocab[i], int(total[i])) for i in total.argsort()[::-1][:K]]

    print(f"\nTF-IDF top {K} (all documents):")
    for word, score in top_tfidf:
        print(f"  {word:20s} {score:.4f}")

    print(f"\nTop {K} most frequent words (all documents):")
    for word, count in top_freq:
        print(f"  {word:20s} {count:6d}")

    return {"tfidf": top_tfidf, "frequent": top_freq}





def main():
    df = load_data()

    # Clean the texts and drop documents that became empty (labels stay aligned)
    df["clean"] = df["text"].map(clean_text)
    df = df[df["clean"] != ""].reset_index(drop=True)

    keyword_analysis(df["clean"], df["label"], K=50)
    top_words_overall(df["clean"], K=50)


if __name__ == "__main__":
    main()
