# Sentiment of Finnish political topics: YLE news vs Twitter (2013–2022)

This project compares how Finnish politics is discussed in the news and on Twitter.
We classify YLE news leads and Finnish tweets as **positive**, **neutral** or **negative**
with a Finnish BERT sentiment model, and follow ten political topics and nine parliamentary parties
year by year.

The full description of the data, method and results is in [technical_report.md](technical_report.md).

Read about the project in https://medium.com/@anniinataiv/what-can-we-learn-from-finnish-politics-analysing-10-years-of-news-and-social-media-2aacd96b46c5
 
## Data

| Source | What is used | In this repo |
|---|---|---|
| YLE news corpus (Finnish Broadcasting Company), VRT files | Leads (opening paragraphs) of articles that mention political keywords or parties, lemmatized | Yes: `data/<year>_2.json` for 2012–2022, already labelled. The analysis uses 2013–2022 (68,820 leads) |
| Tweets ([Nordic Tweet Stream](https://ceur-ws.org/Vol-2084/short10.pdf), Laitinen et al. 2018) | Tweets found with each topic or party name as the search term: 61,750 topic tweets and 45,309 party tweets | No: the raw tweets are not published. Only the yearly counts are in `results/` |

Our first version of the news data selected only articles in the *Politiikka* department (3,831 leads),
made with `preprocess_sentiment.py`. It was used to choose the keywords with TF-IDF and is no longer in
the repo.

**Topics**: hallitus (government), presidentti (president), venäjä (Russia), puolue (party),
ulkopolitiikka (foreign policy), talouspolitiikka (economic policy), turvallisuuspolitiikka (security
policy), maahanmuuttopolitiikka (immigration), perussuomalainen (Finns Party member), pääministeri
(prime minister)

**Parties**: SDP, Vihreät, Vasemmistoliitto, Keskusta, RKP, Kokoomus, Kristillisdemokraatit,
Perussuomalaiset, Liike Nyt

## Method

1. **Selecting the news**: the department tag is missing for many articles in the older YLE files, so
   `preprocess_sentiment_2.py` selects leads that contain political keywords or party names in their
   lemmas instead.
2. **Sentiment**: every news lead and tweet is classified with
   [`fergusq/finbert-finnsentiment`](https://huggingface.co/fergusq/finbert-finnsentiment).
   A text is *neutral* only if the model gives neutral a probability of at least 0.98.
   Otherwise it gets whichever of positive or negative is more likely.
   News and tweets use the same classifier (`sentiment_model.py`), so the two can be compared.
3. **Topic analysis** (`analysis.py`): for each topic and year, count the positive, neutral and
   negative leads and their shares (%). A lead matches a topic if the topic word appears anywhere in
   its lemmas, so different word forms and compounds are included.
4. **Party analysis** (`analysis_parties.py`): the same counts for each party in the news, matched by
   the start of a lemma. *Keskusta* alone usually means "city centre" and *vihreä* also means the
   colour green, so Keskusta is matched only by *keskustapuolue*, *keskustalainen* and *kepu*, and
   Vihreät by the exact lemma *vihreä*.
5. **TF-IDF** (`tfidf.py`): the most characteristic words of positive, neutral and negative news leads.
6. **Visualization**: yearly bar charts of news vs Twitter for each topic and party, heatmaps, and the
   graphs used in our blog (share of negative texts per year, topic and party, and per party over
   time with government periods marked).

## Project structure

```
data/                          labelled YLE news leads, one json file per year
data/twitter/                  raw tweet CSV files (not in the repo, see .gitignore)
results/                       yearly sentiment tables (CSV)
results/graphs/keywords/       news vs Twitter per topic + heatmap
results/graphs/parties/        news vs Twitter per party + heatmap
results/graphs/blog/           graphs used in the blog

preprocess_sentiment_2.py      YLE VRT file -> keyword-filtered leads -> sentiment -> data/<year>_2.json
preprocess_sentiment.py        first version: Politiikka department only (data no longer in the repo)
twitter_sentiment.py           tweets -> sentiment -> results/twitter_*_yearly.csv
analysis.py                    news sentiment per topic and year -> results/news_keyword_*.csv
analysis_parties.py            news sentiment per party and year -> results/news_party_yearly.csv
tfidf.py                       TF-IDF and word frequencies per sentiment (printed)
visualization_yearly.py        yearly bar charts per topic and party
visualization_heatmap.py       news vs Twitter heatmaps per topic and party
visualization_blog.py          graphs used in the blog
sentiment_model.py             FinBERT classifier shared by the sentiment scripts
common.py                      shared settings, topics and data loading
index.py                       runs the pipeline in order
technical_report.md            report: data, preprocessing and results
```

## How to run

Requires Python 3. Run all commands from the project folder.

```bash
pip install -r requirements.txt
python index.py
```

`index.py` runs `analysis.py`, `analysis_parties.py`, `tfidf.py` and the three visualization scripts,
using the data already in `data/` and `results/`. The graphs are saved in `results/graphs/`.
Each script can also be run on its own, e.g. `python visualization_blog.py`. The visualization
scripts only read the tables in `results/`.

**Re-running the sentiment classification** (needs `torch` and `transformers`, and is slow without a GPU):

- News: you need the YLE corpus VRT files, which are not included. For each file:
  ```bash
  python preprocess_sentiment_2.py <file.vrt> <output.json> [neutral_threshold]
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

- **The news stayed calm.** About 88% of news leads are neutral, and only 5–10% were negative in any
  year from 2013 to 2022.
- **Twitter became more negative.** The share of negative political tweets rose from 28% in 2013 to
  49% in 2022. The most negative topics on Twitter were the Finns Party (48% negative), the government
  (46%), immigration (44%) and Russia (40%); security policy (14%) and the president (24%) were the
  calmest.
- **Parties.** In the news, only 4–11% of leads about any party were negative. On Twitter, negative
  tweets grew most for the parties in government: all five parties of the 2019–2023 government
  reached 42–66% negative tweets in 2022, compared with 36% for Kokoomus and 28% for the Finns Party.

See [technical_report.md](technical_report.md) for the full results.
