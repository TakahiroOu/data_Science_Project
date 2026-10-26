"""Classify tweets per keyword and per party with FinBERT and build the yearly tables.

Input:  data/twitter/<searchterm><number>.csv, e.g. hallitus1.csv, SDP1.csv, kepu2.csv
        (first column = tweet text, second column = date starting with the year)
Output: data/twitter/sentiment/<name>_sentiment.json and <name>_yearly_statistics.json
        results/twitter_keyword_yearly.csv and results/twitter_party_yearly.csv

Usage:
    python twitter_sentiment.py                # classify all tweets, then write the tables
    python twitter_sentiment.py --tables-only  # only rebuild the tables from the saved json
"""
import csv
import html
import json
import re
import sys
from pathlib import Path

from common import KEYWORDS, LABELS

INPUT_DIR = Path("data/twitter")
OUTPUT_DIR = INPUT_DIR / "sentiment"
RESULTS_DIR = Path("results")
BATCH_SIZE = 8

YEARS = range(2013, 2023)

# Name -> the single-word search terms you used. The name is used in the
# output files and tables. Missing input files are simply skipped.
KEYWORD_TERMS = {kw: [kw] for kw in KEYWORDS}
KEYWORD_TERMS["maahanmuuttopolitiikka"] = ["maahanmuutto"]  # broader search term

PARTIES = {
    "SDP": ["SDP"],
    "Vihreät": ["vihreät"],
    "Vasemmistoliitto": ["vasemmistoliitto"],
    "Keskusta": ["kepu"],          # plain "keskusta" mostly means "city centre"
    "RKP": ["RKP"],
    "Kokoomus": ["kokoomus"],
    "Kristillisdemokraatit": ["kristillisdemokraatit"],
    "Perussuomalaiset": ["perussuomalaiset"],
    "Liike Nyt": ["liikenyt"],
}

GROUPS = {
    "twitter_keyword_yearly.csv": KEYWORD_TERMS,
    "twitter_party_yearly.csv": PARTIES,
}


def find_input_files(folder, term):
    pattern = re.compile(rf"^{re.escape(term)}(\d+)\.csv$", re.IGNORECASE)
    files = []
    for path in folder.iterdir():
        match = pattern.match(path.name)
        if match:
            files.append((int(match.group(1)), path))
    files.sort(key=lambda item: item[0])
    return [path for _, path in files]


def read_tweets(csv_path):
    tweets = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            text = row[0].strip()
            date = row[1].strip()
            if text.lower() in ("tweet text (full)", "text"):
                continue
            if not text or len(date) < 4:
                continue
            text = html.unescape(text)
            year = date[:4]
            if year.isdigit() and YEARS.start <= int(year) < YEARS.stop:
                tweets.append({"text": text, "date": date, "year": year})
    return tweets


def collect_tweets(search_terms, folder):
    """Read every file for every search term, dropping duplicates."""
    seen = set()
    tweets = []
    used_files = []

    for term in search_terms:
        for in_path in find_input_files(folder, term):
            used_files.append(in_path.name)
            file_tweets = read_tweets(in_path)
            added = 0
            for tweet in file_tweets:
                key = (tweet["text"], tweet["date"])
                if key in seen:
                    continue
                seen.add(key)
                tweet["search_term"] = term
                tweets.append(tweet)
                added += 1
            print(f"{in_path.name}: {len(file_tweets)} tweets "
                  f"({added} new after removing duplicates)")

    return tweets, used_files


def process(name, search_terms, tokenizer, model):
    from sentiment_model import classify

    print(f"\n{'=' * 50}")
    print(f"Processing: {name}  (search terms: {', '.join(search_terms)})")

    tweets, used_files = collect_tweets(search_terms, INPUT_DIR)

    if not used_files:
        print(f"No CSV files found for {name}. Skipping.")
        return

    results = {"positive": [], "negative": [], "neutral": []}
    yearly_results = {
        str(year): {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
        for year in YEARS
    }

    if tweets:
        texts = [tweet["text"] for tweet in tweets]
        labels, scores = classify(texts, tokenizer, model, batch_size=BATCH_SIZE)

        for tweet, label, score in zip(tweets, labels, scores):
            results[label].append({
                "text": tweet["text"],
                "date": tweet["date"],
                "year": tweet["year"],
                "search_term": tweet["search_term"],
                "score": score,
            })
            yearly_results[tweet["year"]][label] += 1
            yearly_results[tweet["year"]]["total"] += 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sentiment_path = OUTPUT_DIR / f"{name}_sentiment.json"
    with open(sentiment_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    statistics_path = OUTPUT_DIR / f"{name}_yearly_statistics.json"
    with open(statistics_path, "w", encoding="utf-8") as f:
        json.dump(yearly_results, f, ensure_ascii=False, indent=2)

    print(f"\nYearly statistics for {name}:")
    for year in YEARS:
        c = yearly_results[str(year)]
        print(f"{year}: Positive={c['positive']}, Negative={c['negative']}, "
              f"Neutral={c['neutral']}, Total={c['total']}")

    print(f"Unique tweets: {len(tweets)}")
    print(f"Sentiment JSON saved: {sentiment_path.name}")
    print(f"Yearly statistics saved: {statistics_path.name}")


def write_yearly_table(names, out_path):
    """Combine the saved *_yearly_statistics.json files into one table with
    the same columns as results/news_keyword_yearly.csv."""
    rows = []
    for name in names:
        statistics_path = OUTPUT_DIR / f"{name}_yearly_statistics.json"
        if not statistics_path.exists():
            print(f"Missing {statistics_path}, {name} left out of {out_path.name}")
            continue
        with open(statistics_path, encoding="utf-8") as f:
            yearly_results = json.load(f)

        for year in YEARS:
            c = yearly_results[str(year)]
            n = c["total"]
            row = {"keyword": name, "year": year, "n_docs": n}
            for label in LABELS:
                row[label] = c[label]
                row[f"{label}_%"] = f"{round(100 * c[label] / n, 1) if n else 0.0:.1f}"
            rows.append(row)

    if not rows:
        print(f"No statistics found, {out_path.name} not written")
        return

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Table saved: {out_path}")


def main():
    if "--tables-only" not in sys.argv:
        from sentiment_model import load_model

        if not INPUT_DIR.is_dir():
            sys.exit(f"Put the tweet CSV files in {INPUT_DIR.resolve()}")

        tokenizer, model = load_model()
        for group in GROUPS.values():
            for name, search_terms in group.items():
                process(name, search_terms, tokenizer, model)
        print("\nAll keywords and parties processed.")

    for filename, group in GROUPS.items():
        write_yearly_table(group, RESULTS_DIR / filename)


if __name__ == "__main__":
    main()
