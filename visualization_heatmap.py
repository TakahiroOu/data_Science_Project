"""Overview heatmap of net sentiment per keyword, news vs Twitter (no time aspect)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from common import GRAPH_DIR, load_results

OUTPUT_DIR = GRAPH_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df, _ = load_results()

# net sentiment = (positive - negative) / all documents, over all years
totals = df.groupby(["keyword", "source"])[["positive", "negative", "n_docs"]].sum()
net = (
    (totals["positive"] - totals["negative"]) / totals["n_docs"]
).unstack("source")
net["Difference (Twitter - News)"] = net["Twitter"] - net["News"]

# Red = negative, blue = positive, white = 0
limit = np.nanmax(np.abs(net.to_numpy()))
norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

fig, ax = plt.subplots(figsize=(9, 7))
image = ax.imshow(net.to_numpy(), cmap="RdBu", norm=norm, aspect="auto")

ax.set_xticks(np.arange(len(net.columns)))
ax.set_xticklabels(net.columns)
ax.set_yticks(np.arange(len(net.index)))
ax.set_yticklabels(net.index)

for row in range(len(net.index)):
    for col in range(len(net.columns)):
        value = net.iat[row, col]
        # White text on the darkest cells so it stays readable
        ax.text(col, row, f"{value:.2f}",
                ha="center", va="center", fontsize=10,
                color="white" if abs(value) > 0.6 * limit else "black")

ax.set_title(
    "Net sentiment 2013-2022: (positive - negative) / all",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
fig.colorbar(image, ax=ax, label="Net sentiment")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "heatmap_news_vs_twitter.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print(f"Saved to: {OUTPUT_DIR / 'heatmap_news_vs_twitter.png'}")
