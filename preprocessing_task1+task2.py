import pandas as pd
import re



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


if __name__ == "__main__":
    corpus = open(r"C:\Users\~~~\Downloads\01.vrt", "r", encoding="utf8")   #for testing, can be automated later
    headlines = extract_headlines(corpus)
    lemmas = extract_lemmas(headlines)
