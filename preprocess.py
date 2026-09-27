import pandas as pd
import re
from collections import Counter
from itertools import islice
from nltk.corpus import stopwords


def extract_headlines(corpus):
    """
    Args: Corpus from the vrt-file with headings, image captions, textbody etc.

    Returns: List, extracted headlines by the tag <sentence>, where type="heading".

    """

    headlines = []
    for hdl in re.findall(r"<sentence[^>]*type=\"heading\"[^>]*>(.*?)</sentence>", str(corpus.read()), re.DOTALL):
        headlines.append(hdl)

    print(headlines[2]) #for testing
    print(len(headlines)) #for testing

    return headlines

def extract_lemmas(headlines):
    lemmas = []

    for headline in headlines:
        for line in headline.strip().splitlines():
            columns = line.split()
            if len(columns) >= 3:
                lemmas.append(columns[2])
    print(lemmas[0:30]) #for testing
    print(len(lemmas)) #for testing
    return lemmas

def take(n, iterable):
        return list(islice(iterable, n))
    


if __name__ == "__main__":

    # TODO: take a whole year at a time
    
    corpus = open(r"path_to_vrt", "r", encoding="utf8")   #for testing, can be automated later
    headlines = extract_headlines(corpus)
    lemmas = extract_lemmas(headlines)
    frequencies = Counter(lemmas)
    descending = {k: v for k, v in sorted(frequencies.items(), key=lambda item: item[1], reverse=True)}

    suomi_stopwords  = stopwords.words('finnish')
    no_stopwords = {key: value for key, value in descending.items() if key not in [".", ",", ":", '"', "?", '-', '–']}
    no_stopwords = {key: value for key, value in no_stopwords.items() if key not in suomi_stopwords}

    res = take(30, no_stopwords.items()) # for testing
    print(res)                           # for testing
    print(len(no_stopwords))             # for testing

    dataf = pd.DataFrame([no_stopwords])
    print(dataf)
    dataf.to_json("february.json", orient="records", indent=4)

