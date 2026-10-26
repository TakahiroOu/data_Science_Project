"""Graphs for the blog: share of negative texts (%), news vs Twitter.

- negative_share_by_year.png   all topics combined, per year
- negative_share_topics.png    per topic
- negative_share_parties.png   per party
- party_trends.png             per party and year, with government periods and elections
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import GRAPH_DIR, load_results

OUTPUT_DIR = GRAPH_DIR / "blog"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"News": "#2a78d6", "Twitter": "#eb6834"}
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#888780"
GOVERNMENT_COLOR = "#e8e6df"

TOPICS = {
    "hallitus": "Government (hallitus)",
    "maahanmuuttopolitiikka": "Immigration (maahanmuutto)",
    "perussuomalainen": "Finns Party member (perussuomalainen)",
    "presidentti": "President (presidentti)",
    "puolue": "Party (puolue)",
    "pääministeri": "Prime minister (pääministeri)",
    "talouspolitiikka": "Economic policy (talouspolitiikka)",
    "turvallisuuspolitiikka": "Security policy (turvallisuuspolitiikka)",
    "ulkopolitiikka": "Foreign policy (ulkopolitiikka)",
    "venäjä": "Russia (venäjä)",
}

# Periods in government, as fractions of a year. Governments: Katainen/Stubb until 29.5.2015,
# Sipilä 29.5.2015-6.6.2019 (Perussuomalaiset left 13.6.2017), Rinne/Marin from 6.6.2019.
GOVERNMENT = {
    "Keskusta": [(2015.41, 2022.9)],
    "Kokoomus": [(2012.5, 2019.43)],
    "Perussuomalaiset": [(2015.41, 2017.45)],
    "SDP": [(2012.5, 2015.41), (2019.43, 2022.9)],
    "Vihreät": [(2012.5, 2014.71), (2019.43, 2022.9)],
    "Vasemmistoliitto": [(2012.5, 2014.23), (2019.43, 2022.9)],
    "RKP": [(2012.5, 2015.41), (2019.43, 2022.9)],
    "Kristillisdemokraatit": [(2012.5, 2015.41)],
    "Liike Nyt": [],
}
PARTY_ORDER = ["Keskusta", "SDP", "Vasemmistoliitto", "Vihreät", "RKP",
               "Kokoomus", "Perussuomalaiset", "Liike Nyt", "Kristillisdemokraatit"]
FEW_TWEETS = {"Kristillisdemokraatit": "few tweets (2–27 per year)",
              "Liike Nyt": "founded 2018, few tweets"}

df, parties = load_results()


def negative_share(data, by):
    totals = data.groupby(by)[["negative", "n_docs"]].sum()
    return (100 * totals["negative"] / totals["n_docs"]).unstack("source")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)


def plot_by_year(filename):
    share = negative_share(df, ["year", "source"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for source in ["News", "Twitter"]:
        ax.plot(share.index, share[source], color=COLORS[source], linewidth=2, marker="o",
                markersize=7, markeredgecolor="white", markeredgewidth=2, label=source)
        last = share[source].iloc[-1]
        ax.text(share.index[-1] + 0.2, last, f"{source} {last:.0f}%",
                va="center", fontsize=11, color=INK)
    ax.set_title("Share of negative political texts per year",
                 fontsize=15, fontweight="bold", loc="left", pad=12)
    ax.set_ylabel("Negative texts (%)", fontsize=11)
    ax.set_xticks(share.index)
    ax.set_xlim(share.index[0] - 0.4, share.index[-1] + 1.6)
    ax.set_ylim(0, 60)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    style(ax)
    ax.text(0.0, -0.13, "10 political topics combined, 2013–2022.",
            transform=ax.transAxes, fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dots(data, title, filename, labels=None):
    """News dot and Twitter dot per row, joined by a line showing the gap."""
    share = negative_share(data, ["keyword", "source"]).sort_values("Twitter")
    names = [labels.get(k, k) if labels else k for k in share.index]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(share) + 1.6))
    y = range(len(share))
    ax.hlines(y, share["News"], share["Twitter"], color=GRID, linewidth=2, alpha=0.6)
    for source in ["News", "Twitter"]:
        ax.scatter(share[source], y, s=90, color=COLORS[source], edgecolor="white",
                   linewidth=2, zorder=3, label=source)
    for i, (news, twitter) in enumerate(zip(share["News"], share["Twitter"])):
        ax.text(news - 1.2, i, f"{news:.0f}%", ha="right", va="center", fontsize=9.5, color=INK)
        ax.text(twitter + 1.2, i, f"{twitter:.0f}%", ha="left", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlim(-6, 65)
    ax.set_xlabel("Negative texts (%)", fontsize=11)
    ax.set_title(title, fontsize=15, fontweight="bold", loc="left", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_party_trends(filename):
    """One small panel per party: negative share per year, government periods shaded."""
    data = parties.assign(share=100 * parties["negative"] / parties["n_docs"])
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True)
    for ax, party in zip(axes.flat, PARTY_ORDER):
        for start, end in GOVERNMENT[party]:
            ax.axvspan(start, end, color=GOVERNMENT_COLOR, zorder=0)
        for year in (2015, 2019):
            ax.axvline(year + 0.3, color="#b4b2a9", linestyle="--", linewidth=1, zorder=1)
        for source in ["News", "Twitter"]:
            d = data[(data["keyword"] == party) & (data["source"] == source)].sort_values("year")
            ax.plot(d["year"], d["share"], color=COLORS[source], linewidth=2,
                    marker="o", markersize=4, zorder=3)
        ax.set_title(party, fontsize=12, fontweight="bold", loc="left", color=INK)
        if party in FEW_TWEETS:
            ax.text(0.02, 0.92, FEW_TWEETS[party], transform=ax.transAxes, fontsize=8.5, color=MUTED)
        ax.set_xlim(2012.5, 2022.9)
        ax.set_ylim(0, 75)
        ax.set_xticks([2013, 2015, 2017, 2019, 2021])
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Negative texts (%)")

    handles = [
        plt.Line2D([], [], color=COLORS["Twitter"], linewidth=2, marker="o", markersize=4, label="Twitter"),
        plt.Line2D([], [], color=COLORS["News"], linewidth=2, marker="o", markersize=4, label="News"),
        Patch(color=GOVERNMENT_COLOR, label="Party in government"),
        plt.Line2D([], [], color="#b4b2a9", linestyle="--", label="Parliamentary election (2015, 2019)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Negative texts per party and year", fontsize=16, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


plot_by_year("negative_share_by_year.png")
plot_dots(df, "Negative texts per topic: news vs Twitter", "negative_share_topics.png", TOPICS)
plot_dots(parties, "Negative texts per party: news vs Twitter", "negative_share_parties.png")
plot_party_trends("party_trends.png")

print(f"Saved to: {OUTPUT_DIR}")
