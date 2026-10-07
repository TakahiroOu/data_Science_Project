import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path("data")

"""All of the json-files in the same folder, for example data/ """

def load_json_file(path: Path) -> pd.DataFrame:
    """Read one JSON file (regular JSON or JSON Lines) into a DataFrame."""
    with path.open(encoding="utf-8") as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            # Fall back to JSON Lines: one JSON object per line
            f.seek(0)
            obj = [json.loads(line) for line in f if line.strip()]

    if isinstance(obj, dict):
        obj = [obj]  # a single record
    return pd.json_normalize(obj)  # also flattens nested objects

files = sorted(DATA_DIR.glob("*.json"))
if not files:
    raise FileNotFoundError(f"No .json files found in {DATA_DIR.resolve()}")

df = pd.concat((load_json_file(p) for p in files), ignore_index=True)
print(f"Loaded {len(files)} files -> DataFrame with shape {df.shape}")


first_col = df.iloc[:, 0]
print(f"First column: '{df.columns[0]}'")

leads = first_col.dropna().astype(str).str.strip()
leads = leads[leads != ""].reset_index(drop=True)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_LETTER_RE = re.compile(r"[^a-zäöå\s]")  # keep a-z plus Finnish letters; adjust as needed
SPACE_RE = re.compile(r"\s+")

"""
def clean_text(text: str) -> str:
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = NON_LETTER_RE.sub(" ", text)  # drops digits and punctuation
    tokens = [t for t in SPACE_RE.split(text) if len(t) > 2 and t not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


cleaned = texts.map(clean_text)
cleaned = cleaned[cleaned != ""]  # drop documents that became empty
"""

def keyword_analysis(lemma_docs, labels, K=10, stop_words=None):
    labels = np.array(labels)
    pos_mask = labels == "positive"
    neg_mask = labels == "negative"
    n_pos, n_neg = pos_mask.sum(), neg_mask.sum()


    # TF-IDF per label (IDF computed over the whole corpus)
    tfidf_vec = TfidfVectorizer(max_df=0.5, min_df=2, stop_words=stop_words)
    X_tfidf = tfidf_vec.fit_transform(lemma_docs)
    tfidf_vocab = tfidf_vec.get_feature_names_out()

    tfidf_results = {}
    for name, mask in (("positive", pos_mask), ("negative", neg_mask)):
        mean_scores = np.asarray(X_tfidf[mask].mean(axis=0)).ravel()
        top = mean_scores.argsort()[::-1][:K]
        tfidf_results[name] = [
            (tfidf_vocab[i], float(mean_scores[i]))
            for i in top if mean_scores[i] > 0
        ]


    # 2) Most frequent words overall, with positive/negative counts
    count_vec = CountVectorizer(min_df=2, stop_words=stop_words)
    X_count = count_vec.fit_transform(lemma_docs)
    vocab = count_vec.get_feature_names_out()

    total = np.asarray(X_count.sum(axis=0)).ravel()
    pos = np.asarray(X_count[pos_mask].sum(axis=0)).ravel()
    neg = np.asarray(X_count[neg_mask].sum(axis=0)).ravel()

    freq_results = [
        {
            "word": vocab[i],
            "total": int(total[i]),
            "positive": int(pos[i]),
            "negative": int(neg[i]),
        }
        for i in total.argsort()[::-1][:K]
    ]

    # Print
    for name, pairs in tfidf_results.items():
        print(f"\nTF-IDF top {K} ({name}):")
        for word, score in pairs:
            print(f"  {word:20s} {score:.4f}")

    print(f"\nLeads: {pos_mask.sum()} positive, {neg_mask.sum()} negative")
    print(f"\nTop {K} most frequent words:")
    print(f"  {'word':20s} {'total':>6s} {'pos':>6s} {'neg':>6s}")
    for r in freq_results:
        print(f"  {r['word']:20s} {r['total']:6d} "
              f"{r['positive']:6d} {r['negative']:6d}")

    return {"tfidf": tfidf_results, "frequent": freq_results}



