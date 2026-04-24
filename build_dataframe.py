import argparse
import os
import pandas as pd


def run(data_dir: str) -> None:

    DATA_DIR = data_dir

    speakers: list[str] = []

    with os.scandir(DATA_DIR) as d:
        for e in d:
            if e.is_dir():
                speakers.append(
                    DATA_DIR+'/'+e.name
                )

    full_dfs: list[pandas.DataFrame] = []

    for s in speakers:
        metadata: list[str] = []
        dfs: list[pandas.DataFrame] = []
        with os.scandir(s) as d:
            for e in d:
                if e.is_file() and e.name.endswith("csv"):
                    metadata.append(s+"/"+e.name)

        for e in metadata:
            df = pd.read_csv(e, sep=";", names=[
                             "word", "start_time", "end_time"])
            df["recording"] = os.path.dirname(e)+"/"\
                + "_".join(os.path.basename(e).split("_")[:4])+".wav"
            dfs.append(df)

        df = pd.concat(dfs)
        df["speaker"] = os.path.basename(s)

        full_dfs.append(df)

    df = pd.concat(full_dfs)\
        .sort_values(by=["speaker", "recording"])\
        .reset_index(drop=True)

    df.to_csv("rec_metadata.tsv", sep="\t")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "-d", type=str,
                        default="./wav_et_textgrids/FRcorp_textgrids_only")
    args = parser.parse_args()

    run(args.data_dir)
