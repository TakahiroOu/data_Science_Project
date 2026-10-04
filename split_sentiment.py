"""Split a VRT file into positive / negative / neutral VRT files by headline sentiment.

Each <text> block is classified by its main headline (first type="heading" sentence
inside <paragraph type="headline">) using the FinBERT FinnSentiment model.
A headline is neutral only if P(neutral) >= NEUTRAL_THRESHOLD; otherwise it goes
to whichever of positive/negative has the higher probability.

Usage: python split_sentiment.py data/2022/01.vrt [out_dir] [neutral_threshold]
"""
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "fergusq/finbert-finnsentiment"
BATCH_SIZE = 32
NEUTRAL_THRESHOLD = 0.98

HEADLINE_RE = re.compile(
    r'<paragraph[^>]*type="headline"[^>]*>.*?<sentence[^>]*type="heading"[^>]*>\n(.*?)</sentence>',
    re.DOTALL,
)


def read_texts(path):
    """Return (header_lines, list of <text>...</text> blocks)."""
    content = Path(path).read_text(encoding="utf-8")
    first_text = content.find("<text ")
    header = content[:first_text]
    texts = re.findall(r"<text .*?</text>\n?", content, re.DOTALL)
    return header, texts


def headline_of(text_block):
    """Rebuild the headline string from the token column of the VRT."""
    match = HEADLINE_RE.search(text_block)
    if not match:
        return ""
    tokens = [line.split("\t")[0] for line in match.group(1).splitlines() if "\t" in line]
    text = " ".join(tokens)
    return re.sub(r" ([,.:;!?)])", r"\1", text).replace("( ", "(")


def classify(headlines, neutral_threshold):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()
    label2id = {l.lower(): i for i, l in model.config.id2label.items()}
    neu, pos, neg = label2id["neutral"], label2id["positive"], label2id["negative"]

    labels, scores = [], []
    for start in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[start:start + BATCH_SIZE]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
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
        print(f"\r{min(start + BATCH_SIZE, len(headlines))}/{len(headlines)}", end="", flush=True)
    print()
    return labels, scores


def main():
    in_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else in_path.parent / "sentiment"
    neutral_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else NEUTRAL_THRESHOLD
    out_dir.mkdir(parents=True, exist_ok=True)

    header, texts = read_texts(in_path)
    headlines = [headline_of(t) for t in texts]
    labels, scores = classify(headlines, neutral_threshold)

    outputs = {}
    for label in sorted(set(labels)):
        outputs[label] = open(out_dir / f"{in_path.stem}_{label}.vrt", "w", encoding="utf-8")
        outputs[label].write(header)
    for text, label in zip(texts, labels):
        outputs[label].write(text)
    for f in outputs.values():
        f.close()

    for label in outputs:
        print(f"{label}: {labels.count(label)} -> {outputs[label].name}")
    for label in outputs:
        print(f"\nTop {label} examples:")
        top = sorted((s, h) for h, l, s in zip(headlines, labels, scores) if l == label)[-5:]
        for s, h in reversed(top):
            print(f"  {s:.2f}  {h}")


if __name__ == "__main__":
    main()
