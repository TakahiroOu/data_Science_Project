"""Run the whole pipeline in order.

    python index.py            # analysis, TF-IDF and graphs from data/ and results/
    python index.py --twitter  # first classify the tweets in data/twitter/ again

The news sentiment step is run separately, once per YLE VRT file, because the
corpus is not in the repo:
    python preprocess_sentiment.py <file.vrt> data/<year>.json
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    "analysis.py",               # results/news_keyword_*.csv
    "tfidf.py",                  # TF-IDF and word frequencies (printed)
    "visualization_yearly.py",   # results/graphs/ bar charts
    "visualization_heatmap.py",  # results/graphs/ heatmap
]


def main():
    steps = STEPS
    if "--twitter" in sys.argv:
        steps = ["twitter_sentiment.py"] + steps

    # Save the graphs without opening a window for each one
    env = dict(os.environ, MPLBACKEND="Agg")

    for script in steps:
        print(f"\n{'=' * 50}\nRunning {script}\n{'=' * 50}")
        subprocess.run([sys.executable, script], cwd=ROOT, env=env, check=True)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
