import argparse
import torch
import time
import os

import pandas as pd
import torch.nn.functional as F
from itertools import combinations
from convert import dequantize_uint8

REPS_PATH = {
    "f64": "reps_out/reps_f64.pt",
    "f32": "reps_out/reps_f32.pt",
    "f16": "reps_out/reps_f16.pt",
    "u8": "reps_out/reps_u8.pt",
}

MD_PATH = "metadata.tsv"


def avg_intra_cosine_distance(group):
    reps = list(group["word_rep"])
    if len(reps) < 2:
        return float("nan")

    dists = []
    for x, y in combinations(reps, 2):
        sim = F.cosine_similarity(x, y, dim=0).item()
        dists.append(1.0 - sim)

    return sum(dists) / len(dists)


def avg_inter_cosine_distance(group):
    reps = list(group["word_rep"])
    spk = list(group["speaker"])
    if len(reps) < 2:
        return float("nan")

    dists = []
    for i, j in combinations(range(len(reps)), 2):
        if spk[i] == spk[j]:
            continue
        sim = F.cosine_similarity(reps[i], reps[j], dim=0).item()
        dists.append(1.0 - sim)
    return sum(dists) / len(dists) if dists else float("nan")


def load_reps(path, precision):
    if precision != "u8":
        ts = torch.load(path)
        return [t for t in ts]

    pack = torch.load(path)
    q = pack["q"]
    x_min = pack["x_min"]
    scale = pack["scale"]
    reps = dequantize_uint8(q, x_min, scale)
    return [t for t in reps]


def compute_run(metadata, reps_path, precision):
    df = metadata.copy()
    word_reps = load_reps(reps_path, precision)
    df["word_rep"] = word_reps

    t0 = time.perf_counter()
    intra_dist = df.groupby(["speaker", "word"])\
                   .apply(avg_intra_cosine_distance)
    inter_dist = df.groupby("word").apply(avg_inter_cosine_distance)

    intra_mean = intra_dist.mean()
    inter_mean = inter_dist.mean()
    ratio = inter_mean / intra_mean
    t1 = time.perf_counter()

    return t1-t0, intra_mean, inter_mean, ratio


if __name__ == "__main__":

    df = pd.read_csv(MD_PATH, sep="\t")
    df["word"] = df["word"].str.strip()

    keys = []
    intras = []
    inters = []
    ratios = []
    time_ls = []
    sizes = []
    for k, v in REPS_PATH.items():
        keys.append(k)
        t, intra, inter, ratio = compute_run(df, v, k)
        intras.append(intra)
        inters.append(inter)
        ratios.append(ratio)
        time_ls.append(t)
        sizes.append(os.path.getsize(v) / (1024**2))

    res_df = pd.DataFrame({
        "Precision": keys,
        "Intra_spk_dist": intras,
        "Inter_spk_dist": inters,
        "Ratio": ratios,
        "Run_time_s": time_ls,
        "Size_MB": sizes
    })

    res_df.to_csv("result.tsv", sep="\t")
