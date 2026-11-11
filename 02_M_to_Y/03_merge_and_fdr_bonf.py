#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chunk_glob", required=True,
        help='Glob pattern, e.g. "/path/inc_chunks/inc_chunk_*.tsv"'
    )
    ap.add_argument(
        "--out_prefix", required=True,
        help='Output prefix, e.g. "/path/results/inc_proteome_phenome" (no .tsv)'
    )
    ap.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold for q-values (default 0.05)"
    )
    ap.add_argument(
        "--sort_full_by", choices=["p", "fdr_q", "bonf_q"], default="p",
        help="How to sort the full (unfiltered) output (default: p)"
    )
    return ap.parse_args()


def main():
    args = parse_args()

    files = sorted(glob.glob(args.chunk_glob))
    if not files:
        raise FileNotFoundError(f"No files matched: {args.chunk_glob}")

    dfs = []
    input_columns = []
    for f in files:
        try:
            if os.path.getsize(f) == 0:
                continue
        except OSError:
            continue

        try:
            df = pd.read_csv(f, sep="\t")
        except pd.errors.EmptyDataError:
            continue
        input_columns = list(dict.fromkeys(input_columns + list(df.columns)))
        if df.shape[0] == 0:
            continue
        dfs.append(df)

    if not dfs:
        # still write empty outputs for pipeline consistency
        empty = pd.DataFrame(columns=list(dict.fromkeys(input_columns + ["icd3", "protein", "p", "fdr_q", "bonf_q"])))
        empty.to_csv(f"{args.out_prefix}.all.tsv", sep="\t", index=False)
        empty.to_csv(f"{args.out_prefix}.fdr_q_lt{args.alpha}.tsv", sep="\t", index=False)
        empty.to_csv(f"{args.out_prefix}.bonf_q_lt{args.alpha}.tsv", sep="\t", index=False)
        print("[WARN] All chunks empty -> wrote empty outputs")
        return

    out = pd.concat(dfs, ignore_index=True)

    if "p" not in out.columns:
        raise ValueError("Column 'p' not found in merged results.")

    # clean p
    out["p"] = pd.to_numeric(out["p"], errors="coerce")
    out = out.dropna(subset=["p"])
    out = out[(out["p"] >= 0) & (out["p"] <= 1)].copy()

    if out.shape[0] == 0:
        empty = pd.DataFrame(columns=list(dict.fromkeys(input_columns + ["icd3", "protein", "p", "fdr_q", "bonf_q"])))
        empty.to_csv(f"{args.out_prefix}.all.tsv", sep="\t", index=False)
        empty.to_csv(f"{args.out_prefix}.fdr_q_lt{args.alpha}.tsv", sep="\t", index=False)
        empty.to_csv(f"{args.out_prefix}.bonf_q_lt{args.alpha}.tsv", sep="\t", index=False)
        print("[WARN] No valid p-values after cleaning -> wrote empty outputs")
        return

    m = out.shape[0]

    # BH-FDR
    out["fdr_q"] = multipletests(out["p"].values, method="fdr_bh")[1]

    # Bonferroni (adjusted p-value)
    out["bonf_q"] = np.minimum(out["p"].values * m, 1.0)

    # ----- (1) Full output (unfiltered) -----
    full_path = f"{args.out_prefix}.all.tsv"
    out_full = out.sort_values([args.sort_full_by, "p"], ascending=[True, True]).copy()
    out_full.to_csv(full_path, sep="\t", index=False)

    # ----- (2) FDR-filtered output -----
    fdr_path = f"{args.out_prefix}.fdr_q_lt{args.alpha}.tsv"
    out_fdr = out[out["fdr_q"] < args.alpha].copy()
    out_fdr.sort_values(["fdr_q", "p"], ascending=[True, True], inplace=True)
    out_fdr.to_csv(fdr_path, sep="\t", index=False)

    # ----- (3) Bonferroni-filtered output -----
    bonf_path = f"{args.out_prefix}.bonf_q_lt{args.alpha}.tsv"
    out_bonf = out[out["bonf_q"] < args.alpha].copy()
    out_bonf.sort_values(["bonf_q", "p"], ascending=[True, True], inplace=True)
    out_bonf.to_csv(bonf_path, sep="\t", index=False)

    print(f"[DONE] merged rows={m:,}")
    print(f"  - Full          : {full_path} (rows={len(out_full):,})")
    print(f"  - FDR q<{args.alpha}: {fdr_path} (rows={len(out_fdr):,})")
    print(f"  - Bonf q<{args.alpha}: {bonf_path} (rows={len(out_bonf):,})")


if __name__ == "__main__":
    main()
