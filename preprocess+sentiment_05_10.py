"""
Extract Politiikka leads from a VRT file, classify them as
positive or negative using FinBERT, and save the results
as JSON.

Only:
    main_department="Politiikka"

is included.

For each token:
    columns[0] = original word
    columns[2] = lemma

The JSON contains both original words and lemmas so that
the lemmas can later be used for TF-IDF, word frequencies,
etc.

Usage:
    python split_sentiment.py data/2022/01.vrt [output.json]
"""

import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL = "fergusq/finbert-finnsentiment"
BATCH_SIZE = 32


def extract_leads(corpus):
    """
    Extract Politiikka leads.

    Only sentences with:
        paragraph_type="lead"

    from texts with:
        main_department="Politiikka"

    are returned.

    Returns:
        List of leads.
    """

    leads = []

    department = "Politiikka"
    paragraph_type = "lead"

    in_department = False
    lead = None

    content_re = re.compile(r'(\w+)="([^"]*)"')

    for line in corpus:

        if line.startswith("<text "):

            metas = dict(content_re.findall(line))

            in_department = (
                metas.get("main_department") == department
            )

            lead = None

        elif line.startswith("</text>"):

            in_department = False
            lead = None

        elif not in_department:

            continue

        elif line.startswith("<sentence"):

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
    """
    Extract both original words and lemma words.

    VRT columns:
        columns[0] = original word
        columns[2] = lemma

    Returns:
        List of dictionaries, one for each lead.

    Example:
        {
            "original": ["Presidentti", "on", "suosittu"],
            "lemma": ["presidentti", "olla", "suosittu"]
        }
    """

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


def words_to_text(original_words):
    """
    Convert original words into text for FinBERT.
    """

    text = " ".join(original_words)

    # Remove spaces before punctuation.
    text = re.sub(
        r" ([,.:;!?])",
        r"\1",
        text
    )

    text = text.replace("( ", "(")

    return text.strip()


def classify(texts):
    """
    Classify texts as positive or negative.

    Neutral is not used.

    The class with the higher probability between
    positive and negative is selected.
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL
    )

    model.eval()

    label2id = {
        label.lower(): i
        for i, label in model.config.id2label.items()
    }

    pos = label2id["positive"]
    neg = label2id["negative"]

    labels = []
    scores = []

    for start in range(0, len(texts), BATCH_SIZE):

        batch = texts[
            start:start + BATCH_SIZE
        ]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():

            probs = model(
                **enc
            ).logits.softmax(dim=-1)

        for p in probs:

            if p[pos] > p[neg]:

                labels.append("positive")
                scores.append(
                    p[pos].item()
                )

            else:

                labels.append("negative")
                scores.append(
                    p[neg].item()
                )

        print(
            f"\r{min(start + BATCH_SIZE, len(texts))}"
            f"/{len(texts)}",
            end="",
            flush=True
        )

    print()

    return labels, scores


def main():

    # ---------------------------------------------------------
    # Input
    # ---------------------------------------------------------

    in_path = Path(sys.argv[1])

    if len(sys.argv) > 2:

        output_path = Path(sys.argv[2])

    else:

        output_path = (
            in_path.parent /
            f"{in_path.stem}_sentiment.json"
        )

    # ---------------------------------------------------------
    # Read VRT
    # ---------------------------------------------------------

    content = in_path.read_text(
        encoding="utf-8"
    )

    corpus = content.splitlines(
        keepends=True
    )

    # ---------------------------------------------------------
    # Extract Politiikka leads
    # ---------------------------------------------------------

    leads = extract_leads(corpus)

    print(
        f"Politiikka leads: {len(leads)}"
    )

    # ---------------------------------------------------------
    # Extract original words + lemmas
    # ---------------------------------------------------------

    words = extract_words(leads)

    # ---------------------------------------------------------
    # Prepare text for FinBERT
    #
    # Use original words for sentiment classification.
    # ---------------------------------------------------------

    sentiment_texts = [
        words_to_text(item["original"])
        for item in words
    ]

    # ---------------------------------------------------------
    # Sentiment classification
    # ---------------------------------------------------------

    labels, scores = classify(
        sentiment_texts
    )

    # ---------------------------------------------------------
    # Build JSON structure
    # ---------------------------------------------------------

    results = {
        "positive": [],
        "negative": []
    }

    for item, label, score in zip(
        words,
        labels,
        scores
    ):

        results[label].append({
            "original": item["original"],
            "lemma": item["lemma"],
            "sentiment_score": score
        })

    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    print()
    print(
        f"Positive: {len(results['positive'])}"
    )

    print(
        f"Negative: {len(results['negative'])}"
    )

    print(
        f"Total: "
        f"{len(results['positive']) + len(results['negative'])}"
    )

    print(
        f"\nSaved to: {output_path}"
    )


if __name__ == "__main__":
    main()
