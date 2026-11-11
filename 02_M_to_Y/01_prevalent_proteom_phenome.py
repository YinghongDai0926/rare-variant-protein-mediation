#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Proteome–phenome (Prevalent) using logistic regression:
Y_k = 1 if first_occurrence_date(icd3=k) <= baseline_date, else 0
Model (per protein j, per phenotype k):
logit(P(Y_k=1)) ~ protein_j + age + sex + PC1..PC10

Outputs: OR, 95% CI, p-value (FDR can be computed after merging chunks)
"""

import argparse
import sys
import numpy as np
import pandas as pd

OUTPUT_COLUMNS = ['icd3', 'protein', 'n_cases', 'n_controls', 'beta', 'se', 'OR', 'CI95_L', 'CI95_U', 'p']
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Progress display is optional; model fitting is unchanged."""
        return iterable


FIXED_COLS = ["sample", "age", "sex"] + [f"PC{i}" for i in range(1, 11)]


def read_header_cols(path):
    with open(path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
    return header


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analytic_tsv", required=True, help="wes_olink_covariates_analytic.tsv")
    ap.add_argument("--icd_first_tsv", required=True, help="icd10_first_occurrence.tsv: sample icd3 first_date")
    ap.add_argument("--followup_tsv", required=True, help="followup_dates.tsv: sample baseline_date death_date censor_date")
    ap.add_argument("--out_tsv", required=True, help="Output TSV path")

    ap.add_argument("--min_cases", type=int, default=50, help="Minimum prevalent cases per icd3 to run")
    ap.add_argument("--min_controls", type=int, default=200, help="Minimum controls per icd3 to run")
    ap.add_argument("--icd3_list", default=None, help="Optional file with icd3 codes (one per line) to restrict")
    ap.add_argument("--protein_start", type=int, default=0, help="Start index (0-based) of protein columns slice")
    ap.add_argument("--protein_end", type=int, default=None, help="End index (exclusive) of protein columns slice")
    ap.add_argument("--compute_fdr_within_chunk", action="store_true",
                    help="If set, compute FDR within this run's results (NOTE: not global if chunked)")
    return ap.parse_args()


def main():
    args = parse_args()

    # --- Load follow-up (analytic sample only later) ---
    fu = pd.read_csv(args.followup_tsv, sep="\t", usecols=["sample", "baseline_date", "death_date", "censor_date"])
    for c in ["baseline_date", "death_date", "censor_date"]:
        fu[c] = pd.to_datetime(fu[c], errors="coerce")

    # censor_end = min(death_date, censor_date) but allow missing death_date
    fu["censor_end"] = fu["censor_date"]
    has_death = fu["death_date"].notna()
    fu.loc[has_death, "censor_end"] = np.minimum(fu.loc[has_death, "death_date"].values,
                                                 fu.loc[has_death, "censor_date"].values)
    fu = fu.set_index("sample")

    # --- Decide which protein columns to load (support chunking) ---
    header = read_header_cols(args.analytic_tsv)
    protein_cols_all = [c for c in header if c not in FIXED_COLS]
    if args.protein_end is None:
        args.protein_end = len(protein_cols_all)
    protein_cols = protein_cols_all[args.protein_start:args.protein_end]

    usecols = FIXED_COLS + protein_cols
    df = pd.read_csv(args.analytic_tsv, sep="\t", usecols=usecols, low_memory=False)
    df = df.set_index("sample")
    # Merge baseline date into analytic df
    df = df.join(fu[["baseline_date", "censor_end"]], how="inner")
    # Basic cleanup
    df["sex"] = pd.to_numeric(df["sex"], errors="coerce")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    for pc in [f"PC{i}" for i in range(1, 11)]:
        df[pc] = pd.to_numeric(df[pc], errors="coerce")

    # Drop rows missing baseline_date or covariates
    df = df.dropna(subset=["baseline_date", "age", "sex"] + [f"PC{i}" for i in range(1, 11)])

    sample_index = df.index

    # --- Load ICD first occurrence and restrict to analytic samples ---
    icd = pd.read_csv(args.icd_first_tsv, sep="\t", usecols=["sample", "icd3", "first_date"])
    icd = icd[icd["sample"].isin(sample_index)]
    icd["first_date"] = pd.to_datetime(icd["first_date"], errors="coerce")
    icd = icd.dropna(subset=["first_date", "icd3"])

    # Attach baseline_date for prevalent definition
    icd = icd.join(df[["baseline_date"]], on="sample", how="inner")
    icd["is_prevalent"] = icd["first_date"] <= icd["baseline_date"]

    # Optional ICD restriction list
    if args.icd3_list is not None:
        with open(args.icd3_list, "r") as f:
            keep_codes = set([line.strip() for line in f if line.strip()])
        icd = icd[icd["icd3"].isin(keep_codes)]

    # Count prevalent cases per icd3
    prev = icd[icd["is_prevalent"]].copy()
    case_counts = prev.groupby("icd3")["sample"].nunique().sort_values(ascending=False)

    # Prepare covariate matrix once
    cov_cols = ["age", "sex"] + [f"PC{i}" for i in range(1, 11)]
    X_cov = df[cov_cols].to_numpy(dtype=float)
    X_cov = sm.add_constant(X_cov, has_constant="add")  # intercept + covs

    results = []
    N = df.shape[0]

    # Iterate icd3 codes that pass thresholds
    codes = []
    for code, n_case in case_counts.items():
        n_ctrl = N - int(n_case)
        if n_case >= args.min_cases and n_ctrl >= args.min_controls:
            codes.append(code)

    if len(codes) == 0:
        print("No icd3 codes pass thresholds. Try lowering --min_cases/--min_controls.", file=sys.stderr)
        df_out = pd.DataFrame(columns=OUTPUT_COLUMNS)
        df_out.to_csv(args.out_tsv, sep="\t", index=False)
        return

    # For faster membership testing
    prev_groups = {code: set(prev.loc[prev["icd3"] == code, "sample"].unique()) for code in codes}

    # Run models
    for code in tqdm(codes, desc="Prevalent logistic (per icd3)"):
        case_set = prev_groups[code]
        y = sample_index.isin(case_set).astype(int)

        # Skip if all 0/1 due to edge cases
        if y.sum() < args.min_cases or y.sum() == 0 or y.sum() == len(y):
            continue

        for prot in protein_cols:
            x = pd.to_numeric(df[prot], errors="coerce").to_numpy(dtype=float)
            if np.all(np.isnan(x)):
                continue
            mask = ~np.isnan(x)
            y_sub = y[mask]
            if y_sub.sum() < args.min_cases or (len(y_sub) - y_sub.sum()) < args.min_controls:
                continue

            X = np.column_stack([x[mask], X_cov[mask, :]])  # protein + (const+covs)
            # NOTE: X already contains const in X_cov; we keep it as is.

            try:
                model = sm.GLM(y_sub, X, family=sm.families.Binomial())
                fit = model.fit(disp=0, maxiter=100)
                beta = fit.params[0]
                se = fit.bse[0]
                p = fit.pvalues[0]
                or_ = float(np.exp(beta))
                ci_l = float(np.exp(beta - 1.96 * se))
                ci_u = float(np.exp(beta + 1.96 * se))
                results.append([code, prot, int(y_sub.sum()), int(len(y_sub) - y_sub.sum()),
                                beta, se, or_, ci_l, ci_u, p])
            except Exception:
                # convergence/separation etc.
                continue

    out = pd.DataFrame(results, columns=[
        "icd3", "protein", "n_cases", "n_controls",
        "beta", "se", "OR", "CI95_L", "CI95_U", "p"
    ])

    if args.compute_fdr_within_chunk and len(out) > 0:
        out["q"] = multipletests(out["p"].values, method="fdr_bh")[1]

    out.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"[DONE] Wrote: {args.out_tsv} (rows={len(out)})")


if __name__ == "__main__":
    main()

