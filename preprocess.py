import pandas as pd
import re
from collections import Counter
from itertools import islice
from nltk.corpus import stopwords


def extract_leads(corpus):
    """
    Args: Corpus from the vrt-file with headings, image captions, textbody etc.

    Returns: List, extracted leads by the tag <sentence>, where type="lead".

    """

    leads = []
    department = "Politiikka"
    paragraph_type = "lead"
    in_department = False
    lead = None
    content_re = re.compile(r'(\w+)="([^"]*)"')

    # set the lead to none, until we find according the paragraph type, 
    # then go line by line until the lead is collected --> tag ending </ reached 
    
    for line in corpus: # find metadata from the <text-tag
        if line.startswith("<text "):
            metas = dict(content_re.findall(line))
            in_department = metas.get("main_department") == department
            lead = None

        elif line.startswith("</text>"):
            in_department = False
            lead = None

        # go line by line until find tag <sentence>, where paragraph_type == lead
        elif not in_department: 
            continue

        elif line.startswith("<sentence"):
            tag_info = dict(content_re.findall(line))
            if tag_info.get("paragraph_type") == paragraph_type:
                lead = []

         # if all lines for the lead are processed -> append the whole lead to lead and set lead = None for the next lead
        elif line.startswith("</sentence>"): 
            if lead is not None:
                leads.append(lead)
            lead = None

        elif lead is not None:
            lead.append(line)

    print(leads[2]) #for testing
    print(len(leads)) #for testing

    return leads

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
    headlines = extract_leads(corpus)
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

