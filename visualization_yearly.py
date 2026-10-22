import matplotlib.pyplot as plt
import numpy as np

from common import GRAPH_DIR, load_results

OUTPUT_DIR = GRAPH_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# News and Twitter per keyword, and Twitter per party
df, parties = load_results()

years = list(range(2013, 2023))
sentiments = ["positive", "neutral", "negative"]

# Gentle pastel colors
colors = {
    "positive": "#8FAED1",  # Soft muted blue
    "neutral": "#F2D58A",   # Soft pastel yellow
    "negative": "#E8A09A",  # Soft coral red
}


def plot_sentiment(data, title, subtitle, sources, width, filename):
    """Stacked sentiment shares per year, one bar per source.

    sources: list of (source name, x offset of its bar)
    """
    fig, ax = plt.subplots(figsize=(15, 7))

    x = np.arange(len(years))

    for i, (source, offset) in enumerate(sources):

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
                    if i == 0
                    else "_nolegend_"
                ),
            )

            bottom += values

    # Title
    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=25,
    )

    # Explain the position of the bars
    ax.text(
        0.5,
        1.02,
        subtitle,
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

    # Save one high-resolution image
    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# One graph per keyword: News on the left, Twitter on the right,
# with a small gap between the two bars.
width = 0.32
gap = 0.04

for keyword in sorted(df["keyword"].unique()):
    plot_sentiment(
        df[df["keyword"] == keyword],
        keyword,
        "News (left bar)  |  Twitter (right bar)",
        [("News", -(width + gap) / 2), ("Twitter", (width + gap) / 2)],
        width,
        OUTPUT_DIR / f"{keyword}.png",
    )

# One graph per party, Twitter only
for party in parties["keyword"].unique():
    plot_sentiment(
        parties[parties["keyword"] == party],
        party,
        "Twitter",
        [("Twitter", 0)],
        0.6,
        OUTPUT_DIR / f"party_{party}.png",
    )

print(f"Created {df['keyword'].nunique()} keyword graphs "
      f"and {parties['keyword'].nunique()} party graphs.")
print(f"Saved to: {OUTPUT_DIR}")
