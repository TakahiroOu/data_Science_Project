import pandas as pd
import re
from collections import Counter
from itertools import islice
from nltk.corpus import stopwords
import nltk


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
        lead_lemmas = []

        for line in headline:
            columns = line.split()

            if len(columns) >= 3:
                lead_lemmas.append(columns[2])
        lemmas.append(lead_lemmas)
    print(lemmas[0:10]) #for testing
    print(len(lemmas)) #for testing
    return lemmas


def take(n, iterable):
    return list(islice(iterable, n))

if __name__ == "__main__":
    corpus = open(r"C:\Users\~~username~~\Downloads\01.vrt", "r", encoding="utf8")   #for testing, can be automated later
    headlines = extract_leads(corpus)
    lemmas = extract_lemmas(headlines)

    nltk.download('stopwords')
    suomi_stopwords  = stopwords.words('finnish') 

    no_stopwords = []
    for lead in lemmas:
        cleaned_lead = [
            lemma for lemma in lead
            if lemma not in [".", ",", ":", '"', "?", '-', '–']
            and lemma not in suomi_stopwords
        ]
        no_stopwords.append(cleaned_lead)

    print(no_stopwords[0])       # for testing
    print(no_stopwords[0:10])    # for testing
    print(len(no_stopwords))     # for testing
