import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("positive", "neutral", "negative")

URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_LETTER_RE = re.compile(r"[^a-zäöå\s]")
SPACE_RE = re.compile(r"\s+")
YEAR_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")

# lemmas
KEYWORDS = [
    "hallitus", "presidentti", "venäjä", "puolue",
    "ulkopolitiikka", "talouspolitiikka", "turvallisuuspolitiikka",
    "maahanmuuttopolitiikka", "perussuomalainen", "pääministeri", 
    ]


def load_data(data_dir, source=None):
    """Read all json-files in the folder. One row per document with
    text (lemmas joined with spaces), label, year (from the file name) and source."""
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
    df["source"] = source or Path(data_dir).name
    print(f"[{df['source'].iloc[0]}] Loaded {len(files)} files, {len(df)} documents")
    print(df["label"].value_counts().to_string())
    return df


def clean_text(text):
    """Lowercase, remove URLs and non-letters, normalise whitespace."""
    text = URL_RE.sub(" ", str(text).lower())
    text = NON_LETTER_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def build_pattern(keyword, match="prefix"):
    """Regex for one keyword, for text that is space-separated lemmas.

    exact    - whole word only            ("vaali" does not match "vaalitulos")
    prefix   - word starts with keyword   ("vaali" matches "vaalitulos")
    contains - keyword anywhere in a word ("puolue" matches "oppositiopuolue" too; widest net)
    """
    kw = re.escape(keyword.lower().strip())
    if match == "exact":
        return rf"(?<!\S){kw}(?!\S)"
    if match == "prefix":
        return rf"(?<!\S){kw}"
    if match == "contains":
        return kw
    raise ValueError("match must be 'exact', 'prefix' or 'contains'")


def filter_by_keyword(df, keyword, text_col="clean", match="prefix"):
    return df[df[text_col].str.contains(build_pattern(keyword, match), regex=True)]


def keyword_label_relation(df, keywords, text_col="clean", label_col="label", match="prefix"):
    """Overall label distribution per keyword (all years together), with lift.

    lift = share of label among documents with the keyword / share of label in all documents.
    """
    baseline = df[label_col].value_counts(normalize=True).reindex(LABELS).fillna(0)

    rows, subsets = [], {}
    for kw in keywords:
        kw = kw.lower().strip()
        subset = filter_by_keyword(df, kw, text_col, match)
        subsets[kw] = subset
        counts = subset[label_col].value_counts().reindex(LABELS).fillna(0).astype(int)
        n = int(counts.sum())

        row = {"keyword": kw, "n_docs": n}
        for name in LABELS:
            row[name] = int(counts[name])
            row[f"{name}_%"] = round(100 * counts[name] / n, 1) if n else np.nan
            row[f"{name}_lift"] = round(counts[name] / n / baseline[name], 2) if n and baseline[name] else np.nan
        row["pos_neg_ratio"] = round(counts["positive"] / counts["negative"], 2) if counts["negative"] else np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("keyword"), subsets


def keyword_label_by_year(df, keywords, text_col="clean", label_col="label",
                          year_col="year", match="prefix"):
    """Proportions of positive / neutral / negative labels per keyword per year.

    Returns a long table: keyword, year, n_docs, count and share (%) of each label.
    Also includes an 'ALL DOCUMENTS' row per year as a baseline for comparison.
    """
    kw_list = ["ALL DOCUMENTS"] + [kw.lower().strip() for kw in keywords]

    rows = []
    for kw in kw_list:
        subset = df if kw == "ALL DOCUMENTS" else filter_by_keyword(df, kw, text_col, match)
        subset = subset.dropna(subset=[year_col])

        counts = (subset.groupby([year_col, label_col]).size()
                  .unstack(fill_value=0).reindex(columns=list(LABELS), fill_value=0))
        counts["n_docs"] = counts.sum(axis=1)

        for year, r in counts.iterrows():
            n = int(r["n_docs"])
            row = {"keyword": kw, "year": int(year), "n_docs": n}
            for name in LABELS:
                row[name] = int(r[name])
                row[f"{name}_%"] = round(100 * r[name] / n, 1) if n else np.nan
            rows.append(row)

    out = pd.DataFrame(rows).sort_values(["keyword", "year"]).reset_index(drop=True)
    return out


def share_pivot(yearly, label):
    """Pivot of one label's share (%): rows = keyword, columns = year."""
    return yearly.pivot(index="keyword", columns="year", values=f"{label}_%")


def analyse(df, name, keywords, match, out_dir):
    df = df.copy()
    df["clean"] = df["text"].map(clean_text)

    summary, _ = keyword_label_relation(df, keywords, match=match)
    print(f"\n=== {name}: labels per keyword (all years) ===")
    print(summary.to_string())

    yearly = keyword_label_by_year(df, keywords, match=match)
    print(f"\n=== {name}: label proportions per keyword per year ===")
    print(yearly.to_string(index=False))

    for label in LABELS:
        print(f"\n--- {name}: {label} share (%) per keyword and year ---")
        print(share_pivot(yearly, label).to_string())

    out_dir.mkdir(exist_ok=True)
    summary.to_csv(out_dir / f"{name}_keyword_summary.csv", encoding="utf-8-sig")
    yearly.to_csv(out_dir / f"{name}_keyword_yearly.csv", index=False, encoding="utf-8-sig")
    return summary, yearly


def main():
    # Adjust the folders to where your json files are.
    df_news = load_data("data/news", source="news")
    #df_twitter = load_data("data/twitter", source="twitter")

    # match: "exact" | "prefix" (default, catches compounds like hallitusohjelma)
    #        | "contains" (widest, also catches oppositiopuolue)
    match = "prefix"
    out_dir = Path("results")

    analyse(df_news, "news", KEYWORDS, match, out_dir)
    #analyse(df_twitter, "twitter", KEYWORDS, match, out_dir)
    print(f"\nCSV files saved in {out_dir.resolve()}")


if __name__ == "__main__":
    main()