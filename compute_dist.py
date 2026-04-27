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


def intra_cosine_distance(group):
    reps = list(group["word_rep"])
    if len(reps) < 2:
        return float("nan")

    dists = []
    for x, y in combinations(reps, 2):
        sim = F.cosine_similarity(x, y, dim=0).item()
        dists.append(1.0 - sim)

    return dists


def inter_cosine_distance(group):
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
    return dists


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


def flatten(series_of_lists):
    out = []
    for l in series_of_lists:
        if isinstance(l, list):
            out.extend(l)
    return out


def compute_run(metadata, reps_path, precision):
    df = metadata.copy()
    word_reps = load_reps(reps_path, precision)
    df["word_rep"] = word_reps

    t0 = time.perf_counter()
    intra_lists = df.groupby(["speaker", "word"])\
                    .apply(intra_cosine_distance)
    inter_lists = df.groupby("word")\
                    .apply(inter_cosine_distance)
    t1 = time.perf_counter()

    intra_flat = flatten(intra_lists)
    inter_flat = flatten(inter_lists)

    intra_mean = sum(intra_flat) / len(intra_flat) if intra_flat else float("nan")
    inter_mean = sum(inter_flat) / len(inter_flat) if inter_flat else float("nan")
    ratio = inter_mean / intra_mean if intra_flat and inter_flat else float("nan")

    return {
        "time_s": t1 - t0,
        "intra_flat": intra_flat,
        "inter_flat": inter_flat,
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "ratio": ratio,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=MD_PATH)
    parser.add_argument("--out-summary", default="result.tsv")
    parser.add_argument("--out-dist", default="dist_long.tsv")
    args = parser.parse_args()

    meta = pd.read_csv(args.metadata, sep="\t")
    meta["word"] = meta["word"].str.strip()

    summary_rows = []
    dist_rows = []

    for k, v in REPS_PATH.items():
        size_mb = os.path.getsize(v) / (1024 ** 2)
        res = compute_run(meta, v, k)

        summary_rows.append({
            "Precision": k,
            "Intra_spk_dist": res["intra_mean"],
            "Inter_spk_dist": res["inter_mean"],
            "Ratio": res["ratio"],
            "Run_time_s": res["time_s"],
            "Size_MB": size_mb,
        })

        for d in res["intra_flat"]:
            dist_rows.append({
                "Precision": k,
                "Type": "Intra",
                "Distance": d,
            })
        for d in res["inter_flat"]:
            dist_rows.append({
                "Precision": k,
                "Type": "Inter",
                "Distance": d,
            })
    
    pd.DataFrame(summary_rows).to_csv(args.out_summary, sep="\t", index=False)
    pd.DataFrame(dist_rows).to_csv(args.out_dist, sep="\t", index=False)

