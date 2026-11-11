#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inputs
  1) chr{CHR}.gene_carriers.filtered.tsv (LoF + P/LP combined, already removed pure-AR genes)
     columns: gene, sample, carrier(=1)
  2) wes_olink_covariates_analytic.tsv
     columns: sample, age, sex, PC1..PC10, proteins...

Model (cis only, gene name must match protein column exactly):
  protein ~ carrier(gene) + age + sex + PC1..PC10
  BUT: for each protein, we only use samples where that protein is non-missing.

Covariate imputation:
  age: median
  sex: mode
  PC1..PC10: 0

Output (per chr):
  chr{CHR}.combined.cis_pqtl.per_protein.tsv.gz
  columns:
    chr gene protein n_used n_carriers_used beta se t p
"""

import argparse
import gzip
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist


COV_COLS_DEFAULT = ["age", "sex"] + [f"PC{i}" for i in range(1, 11)]


def coerce_numeric_or_binary(x: pd.Series, name: str) -> pd.Series:
    """Robust coercion: numeric if possible, else map binary-like strings to 0/1."""
    s = x.copy()
    num = pd.to_numeric(s, errors="coerce")

    if num.isna().any():
        s2 = (
            s.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.replace("\u00A0", " ", regex=False)
            .str.strip()
        )
        s2 = s2.str.replace(r"[^0-9eE\+\-\.]", "", regex=True)
        num2 = pd.to_numeric(s2, errors="coerce")
        if num2.isna().sum() < num.isna().sum():
            num = num2

    uniq = pd.Series(s.dropna().astype(str).str.strip().unique())
    uniq = uniq[uniq != ""]
    if num.isna().any() and len(uniq) == 2:
        a, b = uniq.iloc[0].lower(), uniq.iloc[1].lower()
        s_lower = s.astype(str).str.strip().str.lower()

        if set([a, b]).issubset({"male", "female"}):
            return s_lower.map({"female": 0.0, "male": 1.0})
        if set([a, b]).issubset({"m", "f"}):
            return s_lower.map({"f": 0.0, "m": 1.0})
        if set([a, b]).issubset({"1", "2"}):
            return s_lower.map({"1": 1.0, "2": 0.0})
        if set([a, b]).issubset({"0", "1"}):
            return s_lower.map({"0": 0.0, "1": 1.0})

        ordered = sorted([a, b])
        mapping = {ordered[0]: 0.0, ordered[1]: 1.0}
        return s_lower.map(mapping)

    return num.astype(float)


def fit_beta_for_carrier(y: np.ndarray, x: np.ndarray, C: np.ndarray) -> tuple:
    """
    Fit y ~ x + C (C excludes intercept), return (beta, se, t, p).
    y,x: 1d arrays length n
    C:   2d array shape (n, p) covariates
    """
    n = y.shape[0]
    # add intercept
    Xc = np.column_stack([np.ones((n, 1), dtype=float), C])
    # residualize x and y on covariates
    # beta_cov = (Xc'Xc)^-1 Xc'y
    XtX_inv = np.linalg.pinv(Xc.T @ Xc)

    bx = XtX_inv @ (Xc.T @ x)
    x_hat = Xc @ bx
    x_res = x - x_hat

    by = XtX_inv @ (Xc.T @ y)
    y_hat = Xc @ by
    y_res = y - y_hat

    xss = float(np.dot(x_res, x_res))
    if xss <= 1e-12:
        return (np.nan, np.nan, np.nan, np.nan)

    beta = float(np.dot(x_res, y_res) / xss)

    df = n - Xc.shape[1] - 1  # subtract covariates+intercept and x itself
    if df <= 0:
        return (np.nan, np.nan, np.nan, np.nan)

    rss = float(np.dot(y_res - beta * x_res, y_res - beta * x_res))
    rss = max(rss, 0.0)
    se = float(np.sqrt((rss / df) / xss))
    if se <= 0 or not np.isfinite(se):
        return (beta, np.nan, np.nan, np.nan)

    tval = beta / se
    p = float(2.0 * t_dist.sf(abs(tval), df))
    return (beta, se, float(tval), p)


def read_header_cols(tsv_path: str) -> List[str]:
    return list(pd.read_csv(tsv_path, sep="\t", nrows=0).columns)


def load_carriers_as_index_dict(
    carriers_tsv: str,
    sample_index: Dict[str, int],
    allowed_genes: set,
    chunksize: int = 500_000
) -> Dict[str, np.ndarray]:
    """
    Read long carriers TSV and return gene -> array(row_indices) for carrier samples.
    De-duplicate sample indices per gene.
    """
    gene_to_set: Dict[str, set] = {}
    usecols = ["gene", "sample", "carrier"]

    it = pd.read_csv(
        carriers_tsv,
        sep="\t",
        usecols=usecols,
        dtype={"gene": str, "sample": str},
        chunksize=chunksize
    )

    for chunk in it:
        if "carrier" in chunk.columns:
            chunk = chunk[chunk["carrier"].astype(str) != "0"]
        chunk = chunk[chunk["gene"].isin(allowed_genes)]
        if chunk.empty:
            continue

        # map sample to row index in cov/protein table
        idx = chunk["sample"].map(sample_index)
        m = idx.notna()
        if not m.any():
            continue

        chunk = chunk.loc[m, :]
        idx = idx.loc[m].astype(np.int32).to_numpy()

        for g, i in zip(chunk["gene"].to_numpy(), idx):
            if g not in gene_to_set:
                gene_to_set[g] = set()
            gene_to_set[g].add(int(i))

    gene_to_idx = {g: np.fromiter(sorted(s), dtype=np.int32) for g, s in gene_to_set.items()}
    return gene_to_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chr", required=True, help="1-22, X, Y")
    ap.add_argument("--carriers_tsv_template", required=True, help=".../chr{CHR}.gene_carriers.filtered.tsv")
    ap.add_argument("--covar_protein_tsv", required=True, help="wes_olink_covariates_analytic.tsv")
    ap.add_argument("--sample_col", default="sample")
    ap.add_argument("--cov_cols", nargs="+", default=COV_COLS_DEFAULT)
    ap.add_argument("--min_carriers_used", type=int, default=10,
                    help="Min carriers among non-missing protein samples (default 10)")
    ap.add_argument("--min_samples_used", type=int, default=50, help="Minimum nonmissing samples per protein (original threshold: 50)")
    ap.add_argument("--chunksize", type=int, default=500_000)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    CHR = str(args.chr)
    carriers_tsv = args.carriers_tsv_template.format(CHR=CHR)
    if not os.path.exists(carriers_tsv):
        raise FileNotFoundError(f"Carriers file not found: {carriers_tsv}")
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"chr{CHR}.combined.cis_pqtl.per_protein.tsv.gz")

    # Header to know protein columns
    header_cols = read_header_cols(args.covar_protein_tsv)
    colset = set(header_cols)
    required_cols = [args.sample_col] + args.cov_cols
    miss = [c for c in required_cols if c not in colset]
    if miss:
        raise ValueError(f"Missing required columns in cov/protein file: {miss}")

    protein_colset = set(header_cols) - set(required_cols)

    # Read covariates + ALL protein columns? No:
    # We will first read only sample+cov to build sample_index quickly
    df_sc = pd.read_csv(args.covar_protein_tsv, sep="\t", usecols=required_cols, dtype={args.sample_col: str})
    df_sc = df_sc.drop_duplicates(subset=[args.sample_col]).set_index(args.sample_col, drop=True)
    samples = df_sc.index.astype(str).tolist()
    sample_index = {s: i for i, s in enumerate(samples)}
    n0 = len(samples)

    # Allowed genes for cis: must exist as protein column
    # We still need to know which genes appear in carriers file, but we'll filter during reading
    allowed_genes = protein_colset

    # Read carriers -> gene -> carrier row indices
    gene_to_idx = load_carriers_as_index_dict(
        carriers_tsv=carriers_tsv,
        sample_index=sample_index,
        allowed_genes=allowed_genes,
        chunksize=args.chunksize
    )

    genes = sorted(gene_to_idx.keys())
    if len(genes) == 0:
        with gzip.open(out_path, "wt") as f:
            f.write("chr\tgene\tprotein\tn_used\tn_carriers_used\tbeta\tse\tt\tp\n")
        print(f"[OK] chr{CHR}: no cis-matched genes in carriers. Wrote empty: {out_path}")
        return

    # Now read covariates + only those protein columns needed for this chromosome
    usecols = required_cols + genes
    df = pd.read_csv(args.covar_protein_tsv, sep="\t", usecols=usecols, dtype={args.sample_col: str})
    df = df.drop_duplicates(subset=[args.sample_col]).set_index(args.sample_col, drop=True)
    # align to initial sample order
    df = df.loc[samples, :]

    # Coerce + impute covariates
    C_df = df[args.cov_cols].copy()
    for c in args.cov_cols:
        C_df[c] = coerce_numeric_or_binary(C_df[c], c)

    # Impute as requested
    if "age" in C_df.columns:
        age_med = float(np.nanmedian(C_df["age"].to_numpy(dtype=float)))
        C_df["age"] = C_df["age"].fillna(age_med)
    for pc in [f"PC{i}" for i in range(1, 11)]:
        if pc in C_df.columns:
            C_df[pc] = C_df[pc].fillna(0.0)
    if "sex" in C_df.columns:
        if C_df["sex"].notna().any():
            sex_mode = float(C_df.loc[C_df["sex"].notna(), "sex"].mode().iloc[0])
        else:
            sex_mode = 0.0
        C_df["sex"] = C_df["sex"].fillna(sex_mode)

    # Still any NA in covariates? Drop those few rows globally (should be tiny)
    mask_cov = ~C_df.isna().any(axis=1)
    if (~mask_cov).any():
        n_drop = int((~mask_cov).sum())
        print(f"[WARN] Dropping {n_drop} samples due to covariate NA even after imputation (chr{CHR}).",
              file=sys.stderr)
        C_df = C_df.loc[mask_cov, :]
        df = df.loc[mask_cov, :]
        # update samples & indices
        samples = C_df.index.astype(str).tolist()
        sample_index = {s: i for i, s in enumerate(samples)}
        # also remap carrier indices: easiest is to rebuild gene_to_idx (rare)
        gene_to_idx = load_carriers_as_index_dict(
            carriers_tsv=carriers_tsv,
            sample_index=sample_index,
            allowed_genes=set(genes),
            chunksize=args.chunksize
        )
        genes = sorted(gene_to_idx.keys())
        if len(genes) == 0:
            with gzip.open(out_path, "wt") as f:
                f.write("chr\tgene\tprotein\tn_used\tn_carriers_used\tbeta\tse\tt\tp\n")
            print(f"[OK] chr{CHR}: no cis-matched genes after cov-drop. Wrote empty: {out_path}")
            return

    C_all = C_df.to_numpy(dtype=float)
    n = C_all.shape[0]

    # Prepare output
    rows = []
    for g in genes:
        # y: protein values (may have NA)
        y = pd.to_numeric(df[g], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y)
        n_used = int(ok.sum())
        if n_used < args.min_samples_used:
            continue

        # x carrier vector for all n, then subset to ok
        x = np.zeros(n, dtype=float)
        idx = gene_to_idx.get(g, None)
        if idx is not None and idx.size > 0:
            x[idx] = 1.0
        x_ok = x[ok]
        y_ok = y[ok]
        C_ok = C_all[ok, :]

        n_car = int(x_ok.sum())
        if n_car < args.min_carriers_used:
            continue

        beta, se, tval, p = fit_beta_for_carrier(y_ok, x_ok, C_ok)
        if not np.isfinite(p):
            continue

        rows.append([f"chr{CHR}", g, g, n_used, n_car, beta, se, tval, p])

    with gzip.open(out_path, "wt") as f:
        f.write("chr\tgene\tprotein\tn_used\tn_carriers_used\tbeta\tse\tt\tp\n")
        for r in rows:
            f.write(
                f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t"
                f"{r[5]:.6g}\t{r[6]:.6g}\t{r[7]:.6g}\t{r[8]:.6g}\n"
            )

    print(f"[OK] chr{CHR}: tested {len(rows)} cis genes (per-protein samples). wrote -> {out_path}")


if __name__ == "__main__":
    main()
