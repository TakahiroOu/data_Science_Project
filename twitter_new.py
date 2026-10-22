import csv
import html
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "fergusq/finbert-finnsentiment"
BATCH_SIZE = 8
NEUTRAL_THRESHOLD = 0.98

KEYWORDS = [
    "hallitus",
    "presidentti",
    "Venäjä",
    "puolue",
    "ulkopolitiikka",
    "talouspolitiikka",
    "turvallisuuspolitiikka",
    "perussuomalainen",
    "pääministeri",
    "maahanmuuttopolitiikka",
]

YEARS = range(2013, 2023)


def load_model():
    print("Loading FinBERT model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        use_fast=False
    )
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()

    print("Model loaded successfully.")
    return tokenizer, model


def find_input_files(folder, keyword):
    pattern = re.compile(
        rf"^{re.escape(keyword)}(\d+)\.csv$",
        re.IGNORECASE
    )

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

            if year.isdigit() and 2013 <= int(year) <= 2022:
                tweets.append({
                    "text": text,
                    "date": date,
                    "year": year
                })

    return tweets


def classify(texts, tokenizer, model):
    label2id = {
        label.lower(): i
        for i, label in model.config.id2label.items()
    }

    neu = label2id["neutral"]
    pos = label2id["positive"]
    neg = label2id["negative"]

    labels = []
    scores = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.inference_mode():
            probs = model(**enc).logits.softmax(dim=-1)

        for p in probs:
            if p[neu] >= NEUTRAL_THRESHOLD:
                labels.append("neutral")
                scores.append(p[neu].item())
            elif p[pos] > p[neg]:
                labels.append("positive")
                scores.append(p[pos].item())
            else:
                labels.append("negative")
                scores.append(p[neg].item())

        processed = min(start + BATCH_SIZE, len(texts))
        print(
            f"\rClassified {processed}/{len(texts)} tweets",
            end="",
            flush=True
        )

    print()
    return labels, scores


def process_keyword(keyword, folder, tokenizer, model):
    input_files = find_input_files(folder, keyword)

    if not input_files:
        print(f"\nNo CSV files found for keyword: {keyword}")
        print("Skipping this keyword.")
        return

    print(f"\n{'=' * 50}")
    print(f"Processing keyword: {keyword}")
    print("Input files:", ", ".join(p.name for p in input_files))

    results = {
        "positive": [],
        "negative": [],
        "neutral": []
    }

    yearly_results = {
        str(year): {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "total": 0
        }
        for year in YEARS
    }

    total_tweets = 0

    for in_path in input_files:
        tweets = read_tweets(in_path)
        print(f"\n{in_path.name}: {len(tweets)} tweets")

        if not tweets:
            continue

        texts = [tweet["text"] for tweet in tweets]
        labels, scores = classify(texts, tokenizer, model)

        for tweet, label, score in zip(tweets, labels, scores):
            results[label].append({
                "text": tweet["text"],
                "date": tweet["date"],
                "year": tweet["year"],
                "score": score
            })

            yearly_results[tweet["year"]][label] += 1
            yearly_results[tweet["year"]]["total"] += 1

        total_tweets += len(tweets)
        print(f"Total processed for {keyword}: {total_tweets}")

    sentiment_path = folder / f"{keyword}_sentiment.json"

    with open(sentiment_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    statistics_path = folder / f"{keyword}_yearly_statistics.json"

    with open(statistics_path, "w", encoding="utf-8") as f:
        json.dump(yearly_results, f, ensure_ascii=False, indent=2)

    print(f"\nYearly statistics for {keyword}:")

    for year in YEARS:
        counts = yearly_results[str(year)]

        print(
            f"{year}: "
            f"Positive={counts['positive']}, "
            f"Negative={counts['negative']}, "
            f"Neutral={counts['neutral']}, "
            f"Total={counts['total']}"
        )

    print(f"Total tweets: {total_tweets}")
    print(f"Sentiment JSON saved: {sentiment_path.name}")
    print(f"Yearly statistics saved: {statistics_path.name}")


def main():
    folder = Path(r"C:\Users\mkygh_tm2pfj8\Downloads")

    tokenizer, model = load_model()

    for keyword in KEYWORDS:
        process_keyword(keyword, folder, tokenizer, model)

    print("\nAll keywords processed.")


if __name__ == "__main__":
    main()