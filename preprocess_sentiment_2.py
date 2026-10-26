"""
Extract politics-related leads from a VRT file, classify them as
positive, negative, or neutral using FinBERT, and save the
results as JSON.

The department tag is missing for most articles before 2022, so leads
are selected by keyword instead: a lead is kept if one of its lemmas
starts with one of the keywords at the bottom of this file.

For each token:
    columns[0] = original word
    columns[2] = lemma

Usage:
    python preprocess_sentiment_2.py data/2022/01.vrt [output.json] [neutral_threshold]
"""

import json
import re
import sys
from pathlib import Path

from sentiment_model import NEUTRAL_THRESHOLD, classify, load_model


def extract_leads(corpus, keyword):
    leads = []
    paragraph_type = "lead"
    department = ""
    in_department = False
    lead = None
    content_re = re.compile(r'(\w+)="([^"]*)"')

    for line in corpus:
        if line.startswith("<text "):
            metas = dict(content_re.findall(line))
            in_department = True
            lead = None
        elif line.startswith("</text>"):
            in_department = True
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


def mentions(item, keyword):
    # True if a lemma starts with the keyword. Multi-word keywords
    # such as "liike nyt" are matched against consecutive lemmas.
    lemma_text = " " + " ".join(item["lemma"]).lower()
    return " " + keyword.lower() in lemma_text



def words_to_text(original_words):
    text = " ".join(original_words)
    text = re.sub(r" ([,.:;!?])", r"\1", text)
    text = text.replace("( ", "(")
    return text.strip()


def main(keywords):
    in_path = Path(sys.argv[1])

    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    else:
        output_path = in_path.parent / f"{in_path.stem}_sentiment.json"

    if len(sys.argv) > 3:
        neutral_threshold = float(sys.argv[3])
    else:
        neutral_threshold = NEUTRAL_THRESHOLD

    with open(in_path, encoding="utf-8", errors="replace") as corpus:
        leads = extract_leads(corpus, keywords)
    print(f"Leads: {len(leads)}")

    words = extract_words(leads)

    # Which leads mention each keyword
    matches = {
        keyword: [i for i, item in enumerate(words) if mentions(item, keyword)]
        for keyword in keywords
    }
    for keyword, ids in matches.items():
        print(f"  {keyword}: {len(ids)}")

    # Classify each matching lead once, even if it matches several keywords
    matched_ids = sorted(set().union(*matches.values()))
    print(f"Leads mentioning any keyword: {len(matched_ids)}")

    sentiment_texts = [
        words_to_text(words[i]["original"])
        for i in matched_ids
    ]

    print(f"Neutral threshold: {neutral_threshold}")
    print("Classifying sentiment...")

    tokenizer, model = load_model()
    labels, scores = classify(
        sentiment_texts,
        tokenizer,
        model,
        neutral_threshold
    )
    label_of = dict(zip(matched_ids, labels))

    # Top level: every matching lead once (the format common.load_docs reads).
    # "keywords": the same leads grouped by the keyword they matched.
    results = {
        "positive": [],
        "negative": [],
        "neutral": [],
        "keywords": {
            keyword: {"positive": [], "negative": [], "neutral": []}
            for keyword in keywords
        }
    }

    for i in matched_ids:
        results[label_of[i]].append(words[i]["lemma"])

    for keyword, ids in matches.items():
        for i in ids:
            results["keywords"][keyword][label_of[i]].append(words[i]["lemma"])

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
    # The data in data/*_2.json was made with this list, including "venäjä" and "vihre"
    keywords = ["venäjä", "politii", "poliitti", "sdp", "sosiaalidemokraat", "perussuomalai", "kokoomu",
                "keskusta", "rkp", "liike nyt", "vasemmistoliit", "vihre", "hallitus", "presidentti", "puolue",
                "ulkopolitiikka", "talouspolitiikka", "maahanmuutto", "pääministeri", "kdp", "kristillisdemokraat"]
    main(keywords)

