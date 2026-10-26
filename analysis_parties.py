"""News sentiment per party and year, in the same format as results/twitter_party_yearly.csv.

Reads the keyword-filtered leads in data/*_2.json (made with preprocess_sentiment_2.py)
for the same years as the tweets, and writes results/news_party_yearly.csv.
"""
import re
from pathlib import Path

import pandas as pd

from analysis import clean_text, keyword_label_by_year
from common import NEWS_PARTY_RESULTS, load_data

YEARS = range(2013, 2023)

# Party name (as in the Twitter results) -> regexes matched at the start of a lemma.
# "keskusta" alone mostly means "city centre" and "vihreä" also means the colour green,
# so Keskusta uses only unambiguous forms and Vihreät only the exact lemma.
PARTIES = {
    "SDP": [r"sdp", r"sosiaalidemokraat"],
    "Vihreät": [r"vihreä(?!\S)"],
    "Vasemmistoliitto": [r"vasemmistoliit"],
    "Keskusta": [r"keskustapuolue", r"keskustalai", r"kepu"],
    "RKP": [r"rkp"],
    "Kokoomus": [r"kokoomu"],
    "Kristillisdemokraatit": [r"kristillisdemokraat", r"kd(?!\w)", r"kdp"],
    "Perussuomalaiset": [r"perussuomalai"],
    "Liike Nyt": [r"liike nyt"],
}


def party_label_by_year(df, parties, text_col="clean"):
    """Label counts and shares per party and year."""
    tables = []
    for party, terms in parties.items():
        pattern = r"(?<!\S)(?:" + "|".join(terms) + ")"
        subset = df[df[text_col].str.contains(pattern, regex=True)]
        # With no keywords, keyword_label_by_year returns only the
        # "ALL DOCUMENTS" rows, i.e. the counts for this subset
        yearly = keyword_label_by_year(subset, [], text_col=text_col)
        yearly["keyword"] = party
        tables.append(yearly)
    return pd.concat(tables, ignore_index=True)


def main():
    df = load_data("data", source="news", pattern="*_2.json")
    df = df[df["year"].isin(YEARS)].copy()
    df["clean"] = df["text"].map(clean_text)

    yearly = party_label_by_year(df, PARTIES)
    print(yearly.groupby("keyword")["n_docs"].sum().to_string())

    yearly.to_csv(NEWS_PARTY_RESULTS, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {Path(NEWS_PARTY_RESULTS).resolve()}")


if __name__ == "__main__":
    main()
