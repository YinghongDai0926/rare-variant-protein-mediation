#!/usr/bin/env python3
import argparse, glob
from pathlib import Path
import pandas as pd
import numpy as np

def bh_fdr(p):
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n+1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.indir) / "chr*.combined.cis_pqtl.per_protein.tsv.gz")))
    if not files:
        raise RuntimeError("No chr result files found.")

    df = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    df = df.dropna(subset=["p"]).copy()

    M = df.shape[0]
    df["fdr_q_global"] = bh_fdr(df["p"].to_numpy())
    df["bonf_p_global"] = np.minimum(df["p"].to_numpy() * M, 1.0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    print(f"[OK] merged {len(files)} files, tests={M}, wrote: {args.out}")

if __name__ == "__main__":
    main()

