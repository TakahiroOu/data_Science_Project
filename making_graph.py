
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Input files
NEWS_FILE = r"C:\Users\mkygh_tm2pfj8\Downloads\news.csv"
TWITTER_FILE = r"C:\Users\mkygh_tm2pfj8\Downloads\twitter.txt"

# Output folder
OUTPUT_DIR = Path(
    r"C:\Users\mkygh_tm2pfj8\Downloads\sentiment_graphs"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read the files
news = pd.read_csv(NEWS_FILE)
twitter = pd.read_csv(TWITTER_FILE, sep="\t")

# Keep only the columns needed for plotting
columns = [
    "keyword", "year", "n_docs",
    "positive", "neutral", "negative"
]

news = news[columns].copy()
twitter = twitter[columns].copy()

# Identify each source
news["source"] = "News"
twitter["source"] = "Twitter"

# Combine the datasets
df = pd.concat([news, twitter], ignore_index=True)

df["keyword"] = df["keyword"].str.lower().str.strip()
df["year"] = df["year"].astype(int)

years = list(range(2013, 2023))
sentiments = ["positive", "neutral", "negative"]

# Gentle pastel colors
colors = {
    "positive": "#8FAED1",  # Soft coral red
    "neutral": "#F2D58A",   # Soft pastel yellow
    "negative": "#E8A09A",  # Soft muted blue
}

# Create one graph per keyword
for keyword in sorted(df["keyword"].unique()):

    data = df[df["keyword"] == keyword]

    fig, ax = plt.subplots(figsize=(15, 7))

    x = np.arange(len(years))
    width = 0.32
    gap = 0.04

    # News on the left, Twitter on the right,
    # with a small gap between the two bars.
    sources = [
        ("News", -(width + gap) / 2),
        ("Twitter", (width + gap) / 2),
    ]

    for source, offset in sources:

        subset = (
            data[data["source"] == source]
            .set_index("year")
            .reindex(years)
        )

        # Use counts, not the supplied percentages
        counts = subset[sentiments].fillna(0)

        # Convert counts to percentages
        totals = counts.sum(axis=1)

        percentages = (
            counts.div(totals.replace(0, np.nan), axis=0) * 100
        ).fillna(0)

        bottom = np.zeros(len(years))

        for sentiment in sentiments:

            values = percentages[sentiment].to_numpy()

            ax.bar(
                x + offset,
                values,
                width=width,
                bottom=bottom,
                color=colors[sentiment],
                edgecolor="white",
                linewidth=0.5,
                label=(
                    sentiment.capitalize()
                    if source == "News"
                    else "_nolegend_"
                ),
            )

            bottom += values

    # Title
    ax.set_title(
        keyword,
        fontsize=16,
        fontweight="bold",
        pad=25,
    )

    # Explain the position of the two bars
    ax.text(
        0.5,
        1.02,
        "News (left bar)  |  Twitter (right bar)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    # Axis labels and ticks
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Share of sentiment (%)", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))

    # Light horizontal gridlines
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )
    ax.set_axisbelow(True)

    # Legend outside the plot, on the right
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        title="Sentiment",
        frameon=False,
    )

    # Leave space on the right for the legend
    fig.tight_layout(rect=[0, 0, 0.86, 1])

    # Save one high-resolution image per keyword
    filename = OUTPUT_DIR / f"{keyword}.png"

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

print(f"Created {df['keyword'].nunique()} graphs.")
print(f"Saved to: {OUTPUT_DIR}")
