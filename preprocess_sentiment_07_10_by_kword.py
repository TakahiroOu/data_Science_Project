"""
Extract Politiikka leads from a VRT file, classify them as
positive, negative, or neutral using FinBERT, and save the
results as JSON.

Only:
    keywords="Politiikka"

is included.

For each token:
    columns[0] = original word
    columns[2] = lemma

Usage:
    python ---
"""

import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "fergusq/finbert-finnsentiment"
BATCH_SIZE = 32
NEUTRAL_THRESHOLD = 0.98


def extract_leads(corpus):
    leads = []
    paragraph_type = "lead"
    lead = None
    content_re = re.compile(r'(\w+)="([^"]*)"')

    for line in corpus:
        if line.startswith("<sentence"):
            tag_info = dict(content_re.findall(line))
            if tag_info.get("paragraph_type") == paragraph_type:
                lead = []
        elif line.startswith("</sentence>"):
            if lead is not None:
                leads.append(lead)
            lead = None
        elif lead is not None:
            lead.append(line)

    return leads


def extract_words(headlines):
    words = []

    for headline in headlines:
        original_words = []
        lemma_words = []

        for line in headline:
            columns = line.split()

            if len(columns) >= 3:
                original_words.append(columns[0])
                lemma_words.append(columns[2])

        words.append({
            "original": original_words,
            "lemma": lemma_words
        })

    return words

KEYWORD = "politiikka"

def mentions_keyword(item):
    return any(KEYWORD in lemma.lower() for lemma in item["lemma"])

def words_to_text(original_words):
    text = " ".join(original_words)
    text = re.sub(r" ([,.:;!?])", r"\1", text)
    text = text.replace("( ", "(")
    return text.strip()


def classify(texts, neutral_threshold):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()

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

        with torch.no_grad():
            probs = model(**enc).logits.softmax(dim=-1)

        for p in probs:
            if p[neu] >= neutral_threshold:
                labels.append("neutral")
                scores.append(p[neu].item())
            elif p[pos] > p[neg]:
                labels.append("positive")
                scores.append(p[pos].item())
            else:
                labels.append("negative")
                scores.append(p[neg].item())

        print(
            f"\r{min(start + BATCH_SIZE, len(texts))}"
            f"/{len(texts)}",
            end="",
            flush=True
        )

    print()
    return labels, scores


def main():
    in_path = Path(sys.argv[1])

    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    else:
        output_path = in_path.parent / f"{in_path.stem}_sentiment.json"

    if len(sys.argv) > 3:
        neutral_threshold = float(sys.argv[3])
    else:
        neutral_threshold = NEUTRAL_THRESHOLD


    with open(in_path, encoding="utf-8") as f:
        leads = extract_leads(f)
    
    """content = in_path.read_text(encoding="utf-8")
    corpus = content.splitlines(keepends=True)

    leads = extract_leads(corpus)
    print(f"Politiikka leads: {len(leads)}")"""

    words = extract_words(leads)
    words = [w for w in words if mentions_keyword(w)]
    print(f"Leads mentioning '{KEYWORD}': {len(words)}")
    sentiment_texts = [
        words_to_text(item["original"])
        for item in words
    ]

    print(f"Neutral threshold: {neutral_threshold}")
    print("Classifying sentiment...")

    labels, scores = classify(
        sentiment_texts,
        neutral_threshold
    )

    results = {
        "positive": [],
        "negative": [],
        "neutral": []
    }

    for item, label in zip(words, labels):
        results[label].append(item["lemma"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(f"Positive: {len(results['positive'])}")
    print(f"Negative: {len(results['negative'])}")
    print(f"Neutral: {len(results['neutral'])}")
    print(
        f"Total: "
        f"{len(results['positive']) + len(results['negative']) + len(results['neutral'])}"
    )
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
