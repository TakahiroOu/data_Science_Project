"""some ideas here:
pos/neut/neg -> graph comparison between news/tweets during 10 years per keyword
heatmap for overall analysis -> no time aspect

"""
import plotly.express as px
import plotly.graph_objects as go  # hover over datapoint


dark_blue = "#2f3140"
blue = "#545d71"
orange = "#d58c66"
grey = "#bfc0c0"


# example
# Overview heatmap, no time
avg = wide.groupby(["keyword", "dataset"]).apply(
    lambda g: (g["positive"].sum() - g["negative"].sum()) / g["n"].sum()
).unstack("dataset")
avg["difference"] = avg.iloc[:, 1] - avg.iloc[:, 0]

fig = px.imshow(avg, color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                text_auto=".2f", aspect="auto")
fig.show()
