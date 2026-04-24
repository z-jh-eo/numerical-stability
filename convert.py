import argparse
import torch
import pandas as pd
from pathlib import Path


def quantize_uint8(x: torch.Tensor):
    x_min = x.min(dim=1, keepdim=True).values
    x_max = x.max(dim=1, keepdim=True).values
    scale = (x_max - x_min) / 255.0
    scale = torch.clamp(scale, min=1e-8)
    q = ((x - x_min) / scale).round().clamp(0, 225).to(torch.uint8)
    return q, x_min.squeeze(1), scale.squeeze(1)


def dequantize_uint8(q, x_min, scale):
    return q.float() * scale.unsqueeze(1) + x_min.unsqueeze(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="metadata.tsv")
    parser.add_argument("--reps", default="word_reps.pt")
    parser.add_argument("--outdir", default="reps_out")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata, sep="\t")
    reps = torch.load(args.reps)

    assert len(df) == reps.shape[0], "Metadata rows != reps rows"

    torch.save(
        reps.to(torch.float64), f"{args.outdir}/reps_f64.pt"
    )
    torch.save(
        reps.to(torch.float32), f"{args.outdir}/reps_f32.pt"
    )
    torch.save(
        reps.to(torch.float16), f"{args.outdir}/reps_f16.pt"
    )
    q, x_min, scale = quantize_uint8(reps.to(torch.float32))
    torch.save(
        {"q": q, "x_min": x_min, "scale": scale}, f"{args.outdir}/reps_u8.pt"
    )
    print("Representations saved to:", args.outdir)
