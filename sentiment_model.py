"""FinBERT sentiment classifier shared by preprocess_sentiment.py (news)
and twitter_sentiment.py (tweets), so both are labelled the same way.

A text is neutral only if P(neutral) >= neutral_threshold; otherwise it goes
to whichever of positive/negative has the higher probability.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "fergusq/finbert-finnsentiment"
BATCH_SIZE = 32
NEUTRAL_THRESHOLD = 0.98


def load_model():
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()
    print("Model loaded successfully.")
    return tokenizer, model


def classify(texts, tokenizer, model, neutral_threshold=NEUTRAL_THRESHOLD,
             batch_size=BATCH_SIZE):
    label2id = {label.lower(): i for i, label in model.config.id2label.items()}
    neu = label2id["neutral"]
    pos = label2id["positive"]
    neg = label2id["negative"]

    labels = []
    scores = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=128, return_tensors="pt")

        with torch.inference_mode():
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

        processed = min(start + batch_size, len(texts))
        print(f"\rClassified {processed}/{len(texts)}", end="", flush=True)

    print()
    return labels, scores
