import pandas as pd
import re



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


if __name__ == "__main__":
    corpus = open("01.vrt", "r", encoding="utf8")   #for testing, can be automated later
    extract_leads(corpus)