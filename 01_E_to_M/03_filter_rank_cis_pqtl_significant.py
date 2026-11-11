#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Filter and rank cis pQTL results by global FDR and Bonferroni."
    )
    ap.add_argument("--in_tsv", required=True,
                    help="Input TSV with global corrections, e.g. cis_pqtl.per_protein.ALL.global_corrected.tsv")
    ap.add_argument("--out_prefix", required=True,
                    help="Output prefix, e.g. /path/cis_pqtl.per_protein")
    ap.add_argument("--fdr_thr", type=float, default=0.05, help="FDR threshold (default 0.05)")
    ap.add_argument("--bonf_thr", type=float, default=0.05, help="Bonferroni threshold on bonf_p_global (default 0.05)")
    ap.add_argument("--write_full", action="store_true",
                    help="Also write full (unscreened) table sorted by p.")
    args = ap.parse_args()

    df = pd.read_csv(args.in_tsv, sep="\t", dtype=str)
    # coerce numeric columns robustly
    for col in ["n_used", "n_carriers_used"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["beta", "se", "t", "p", "fdr_q_global", "bonf_p_global"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = {"chr", "gene", "protein", "p", "fdr_q_global", "bonf_p_global"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise RuntimeError(f"Missing required columns in input: {missing}")

    # Optional: full output
    if args.write_full:
        out_full = f"{args.out_prefix}.ALL.sorted_by_p.tsv"
        df_full = df.sort_values(["p", "fdr_q_global", "bonf_p_global"], ascending=[True, True, True])
        df_full.to_csv(out_full, sep="\t", index=False)
        print(f"[OK] wrote full table: {out_full} (n={df_full.shape[0]})")

    # FDR significant
    df_fdr = df[df["fdr_q_global"].notna() & (df["fdr_q_global"] < args.fdr_thr)].copy()
    df_fdr = df_fdr.sort_values(["fdr_q_global", "p"], ascending=[True, True])
    out_fdr = f"{args.out_prefix}.FDR_lt{args.fdr_thr}.sorted.tsv"
    df_fdr.to_csv(out_fdr, sep="\t", index=False)
    print(f"[OK] wrote FDR-significant: {out_fdr} (n={df_fdr.shape[0]})")

    # Bonferroni significant
    df_bonf = df[df["bonf_p_global"].notna() & (df["bonf_p_global"] < args.bonf_thr)].copy()
    df_bonf = df_bonf.sort_values(["bonf_p_global", "p"], ascending=[True, True])
    out_bonf = f"{args.out_prefix}.BONF_lt{args.bonf_thr}.sorted.tsv"
    df_bonf.to_csv(out_bonf, sep="\t", index=False)
    print(f"[OK] wrote Bonferroni-significant: {out_bonf} (n={df_bonf.shape[0]})")

    # Quick comparison summary
    fdr_keys = set(zip(df_fdr["chr"], df_fdr["gene"], df_fdr["protein"]))
    bonf_keys = set(zip(df_bonf["chr"], df_bonf["gene"], df_bonf["protein"]))
    inter = len(fdr_keys & bonf_keys)
    print(f"[INFO] Overlap: FDR ∩ Bonf = {inter} (FDR={len(fdr_keys)}, Bonf={len(bonf_keys)})")


if __name__ == "__main__":
    main()

