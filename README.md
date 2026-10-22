# Sentiment of Finnish political topics: YLE news vs Twitter (2013–2022)

This project compares how Finnish political topics are discussed in the news and on Twitter.
We classify YLE political news leads and Finnish tweets as **positive**, **neutral** or **negative**
with a Finnish BERT sentiment model, and follow ten political keywords and nine parties year by year.

The full description of the data, method and results is in [technical_report.md](technical_report.md).

## Data

| Source | What is used | In this repo |
|---|---|---|
| YLE news corpus (Finnish Broadcasting Company), VRT files | Leads of articles in the main department *Politiikka* (politics), lemmatized | Yes: `data/2013.json` … `data/2022.json` (3,831 leads, already labelled) |
| Tweets ([Nordic Tweet Stream](https://ceur-ws.org/Vol-2084/short10.pdf), Laitinen et al. 2018) | Tweets found with each keyword or party name as the search term | No: the raw tweets are not published. Only the yearly counts are in `results/` |

**Keywords** (lemmas): hallitus, presidentti, venäjä, puolue, ulkopolitiikka, talouspolitiikka,
turvallisuuspolitiikka, maahanmuuttopolitiikka, perussuomalainen, pääministeri

**Parties**: SDP, Vihreät, Vasemmistoliitto, Keskusta, RKP, Kokoomus, Kristillisdemokraatit,
Perussuomalaiset, Liike Nyt

## Method

1. **Sentiment**: every news lead and tweet is classified with
   [`fergusq/finbert-finnsentiment`](https://huggingface.co/fergusq/finbert-finnsentiment).
   A text is *neutral* only if the model gives neutral a probability of at least 0.98.
   Otherwise it gets whichever of positive or negative is more likely.
   News and tweets use the same classifier (`sentiment_model.py`), so the two can be compared.
2. **Keyword analysis**: for each keyword and year, count the positive, neutral and negative
   documents and their shares (%). For news, a lead matches a keyword if one of its lemmas
   starts with the keyword, so compounds like *hallitusohjelma* also count for *hallitus*.
3. **TF-IDF**: find the most characteristic words of positive, neutral and negative news leads.
4. **Visualization**: yearly bar charts of news vs Twitter for each keyword, Twitter charts for
   each party, and a heatmap of net sentiment, (positive − negative) / all, over all years.

## Project structure

```
data/                        labelled YLE news leads, one json file per year
data/twitter/                raw tweet CSV files (not in the repo, see .gitignore)
results/                     yearly sentiment tables (CSV)
results/graphs/              graphs (created when you run the scripts)

preprocess_sentiment.py      YLE VRT file -> politics leads -> sentiment -> data/<year>.json
twitter_sentiment.py         tweets -> sentiment -> results/twitter_*_yearly.csv
analysis.py                  news sentiment per keyword and year -> results/news_keyword_*.csv
tfidf.py                     TF-IDF and word frequencies per sentiment (printed)
visualization_yearly.py      yearly bar charts per keyword and party
visualization_heatmap.py     news vs Twitter net sentiment heatmap
sentiment_model.py           FinBERT classifier shared by the two sentiment scripts
common.py                    shared settings, keywords and data loading
index.py                     runs the pipeline in order
technical_report.md          report: data, preprocessing and results
```

## How to run

Requires Python 3. Run all commands from the project folder.

```bash
pip install -r requirements.txt
python index.py
```

`index.py` runs `analysis.py`, `tfidf.py`, `visualization_yearly.py` and `visualization_heatmap.py`
using the data already in `data/` and `results/`. The graphs are saved in `results/graphs/`.
Each script can also be run on its own, e.g. `python visualization_heatmap.py`.

**Re-running the sentiment classification** (needs `torch` and `transformers`, and is slow without a GPU):

- News: you need the YLE corpus VRT files, which are not included. For each file:
  ```bash
  python preprocess_sentiment.py <file.vrt> <output.json> [neutral_threshold]
  ```
- Tweets: put the tweet CSV files in `data/twitter/`, named `<searchterm><number>.csv`
  (e.g. `hallitus1.csv`, `SDP1.csv`), with the tweet text in the first column and the date
  in the second. Then run:
  ```bash
  python index.py --twitter
  ```
  `python twitter_sentiment.py --tables-only` rebuilds the tables in `results/` from the saved
  sentiment files without classifying again.

## Results in short

- News leads are mostly neutral: 3,303 neutral, 432 negative and 96 positive (about 86% neutral).
- Twitter is much more negative. For 8 of the 10 keywords, net sentiment on Twitter is lower than
  in the news. The largest gaps are *perussuomalainen*, *venäjä* and *puolue*. Only *presidentti*
  and *turvallisuuspolitiikka* are slightly positive on Twitter.

See [technical_report.md](technical_report.md) for the full results.
