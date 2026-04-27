import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

IN_PATH = "result.tsv"
LONG_PATH = "dist_long.tsv"
OUTDIR  = "plots"

df = pd.read_csv(IN_PATH, sep="\t")

df1 = pd.melt(
    df,
    id_vars = ["Precision"],
    value_vars = ["Intra_spk_dist", "Inter_spk_dist"],
    var_name="Type",
    value_name="Distance",
)
df1["Type"] = df1["Type"].str.split("_").str[0]

dist_df = pd.read_csv(LONG_PATH, sep="\t")

plt.figure()
sns.barplot(
    data=df1,
    x="Precision",
    y="Distance",
    hue="Type"
)
plt.title("Average Cosine Distance")
plt.xlabel("Precision")
plt.ylabel("Cosine distance")
plt.savefig(f"{OUTDIR}/dist.png")
plt.close()

plt.figure()
sns.barplot(
    data=df,
    x="Precision",
    y="Run_time_s",
)
plt.title("Total Running Time")
plt.xlabel("Precision")
plt.ylabel("Time(s)")
plt.savefig(f"{OUTDIR}/time.png")
plt.close()

plt.figure()
sns.barplot(
    data=df,
    x="Precision",
    y="Size_MB"
)
plt.title("Tensor Size")
plt.xlabel("Precision")
plt.ylabel("Size(MB)")
plt.savefig(f"{OUTDIR}/size.png")
plt.close()


# ----- New distribution plots -----
# KDE plot per precision
g = sns.FacetGrid(dist_df, col="Precision", hue="Type", col_wrap=2, sharex=True, sharey=True)
g.map(sns.kdeplot, "Distance", fill=True, alpha=0.4)
g.add_legend()
g.fig.suptitle("Distance Distributions (KDE)", y=1.02)
g.savefig(f"{OUTDIR}/dist_kde.png")
plt.close(g.fig)

# Histogram plot per precision
g2 = sns.FacetGrid(dist_df, col="Precision", hue="Type", col_wrap=2, sharex=True, sharey=True)
g2.map(sns.histplot, "Distance", bins=40, alpha=0.5, stat="density")
g2.add_legend()
g2.fig.suptitle("Distance Distributions (Histogram)", y=1.02)
g2.savefig(f"{OUTDIR}/dist_hist.png")
plt.close(g2.fig)