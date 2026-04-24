import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

IN_PATH = "result.tsv"
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
