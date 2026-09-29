import pandas as pd 
import re 
from collections import Counter 
from itertools import islice 
 
 
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
 
    for line in corpus:
        if line.startswith("<text "):
            metas = dict(content_re.findall(line)) 
            in_department = metas.get("main_department") == department 
            lead = None 
 
        elif line.startswith("</text>"):
            in_department = False 
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
 
    print(leads[2]) 
    print(len(leads)) 
 
    return leads 


def extract_words(headlines): 
    words = [] 
 
    for headline in headlines: 
        lead_words = [] 
 
        for line in headline: 
            columns = line.split() 
 
            if len(columns) >= 1: 
                lead_words.append(columns[0]) 
 
        words.append(lead_words) 
 
    print(words[0:10]) 
    print(len(words)) 
 
    return words 
 

if __name__ == "__main__": 
    corpus = open(r"C:\Users\~~username~~\Downloads\01.vrt", "r", encoding="utf8")
    
    headlines = extract_leads(corpus) 
    words = extract_words(headlines) 
 
    print(words[0]) 
    print(words[0:10]) 
    print(len(words))
