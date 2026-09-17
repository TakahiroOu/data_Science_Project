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


if __name__ == "__main__":
    corpus = open("01.vrt", "r", encoding="utf8")   #for testing, can be automated later
    extract_headlines(corpus)