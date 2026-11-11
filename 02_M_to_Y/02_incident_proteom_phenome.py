#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Proteome–phenome (Incident) using Cox PH (statsmodels PHReg).

For each icd3 phenotype k:
- Exclude prevalent cases: first_date <= baseline_date
- Event = 1 if baseline_date < first_date <= censor_end
- Duration = min(first_date, censor_end) - baseline_date (days)
Model (per protein j, per phenotype k):
CoxPH: Surv(duration, event) ~ protein_j + age + sex + PC1..PC10

NOTE:
- Designed for chunking by protein columns.
- Global FDR should be computed AFTER merging all chunks.
"""

import argparse
import sys
import numpy as np
import pandas as pd

OUTPUT_COLUMNS = ['icd3', 'protein', 'n_events', 'n_atrisk', 'beta', 'se', 'HR', 'CI95_L', 'CI95_U', 'p']

from statsmodels.duration.hazard_regression import PHReg

FIXED_COLS = ["sample", "age", "sex"] + [f"PC{i}" for i in range(1, 11)]
COV_COLS = ["age", "sex"] + [f"PC{i}" for i in range(1, 11)]


def read_header_cols(path: str):
    with open(path, "r") as f:
        return f.readline().rstrip("\n").split("\t")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analytic_tsv", required=True)
    ap.add_argument("--icd_first_tsv", required=True)
    ap.add_argument("--followup_tsv", required=True)
    ap.add_argument("--out_tsv", required=True)

    ap.add_argument("--min_events", type=int, default=50)
    ap.add_argument("--min_samples", type=int, default=2000,
                    help="Minimum complete-case at-risk samples per protein (original threshold: 2000)")
    ap.add_argument("--icd3_list", default=None, help="Optional file with icd3 codes (one per line)")
    ap.add_argument("--protein_start", type=int, default=0)
    ap.add_argument("--protein_end", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=25, help="Print progress every N icd3 codes")
    return ap.parse_args()


def main():
    args = parse_args()

    # --- follow-up table ---
    fu = pd.read_csv(
        args.followup_tsv,
        sep="\t",
        usecols=["sample", "baseline_date", "death_date", "censor_date"],
    )
    for c in ["baseline_date", "death_date", "censor_date"]:
        fu[c] = pd.to_datetime(fu[c], errors="coerce")

    fu = fu.dropna(subset=["baseline_date", "censor_date"])
    # censor_end = min(death_date, censor_date) when death exists; else censor_date
    fu["censor_end"] = fu["censor_date"]
    has_death = fu["death_date"].notna()
    fu.loc[has_death, "censor_end"] = fu.loc[has_death, ["death_date", "censor_date"]].min(axis=1)

    fu = fu.dropna(subset=["baseline_date", "censor_end"])
    fu["t_censor_days"] = (fu["censor_end"] - fu["baseline_date"]).dt.days
    fu = fu[fu["t_censor_days"] > 0].copy()

    # --- analytic table (cov + protein slice) ---
    header = read_header_cols(args.analytic_tsv)
    protein_cols_all = [c for c in header if c not in FIXED_COLS]

    if args.protein_end is None:
        args.protein_end = len(protein_cols_all)
    # robust slicing even if end > len
    protein_cols = protein_cols_all[args.protein_start: min(args.protein_end, len(protein_cols_all))]

    if len(protein_cols) == 0:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(args.out_tsv, sep="\t", index=False)
        print(f"[WARN] Empty protein slice. Wrote empty: {args.out_tsv}", file=sys.stderr)
        return

    usecols = FIXED_COLS + protein_cols
    df = pd.read_csv(args.analytic_tsv, sep="\t", usecols=usecols, low_memory=False)

    # merge follow-up (inner join: keep only analytic samples with follow-up)
    df = df.merge(fu[["sample", "baseline_date", "censor_end", "t_censor_days"]], on="sample", how="inner")

    # numeric covariates
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["sex"] = pd.to_numeric(df["sex"], errors="coerce")
    for pc in [f"PC{i}" for i in range(1, 11)]:
        df[pc] = pd.to_numeric(df[pc], errors="coerce")

    df = df.dropna(subset=COV_COLS + ["baseline_date", "censor_end", "t_censor_days"]).copy()
    if df.shape[0] == 0:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(args.out_tsv, sep="\t", index=False)
        print(f"[WARN] No samples after QC. Wrote empty: {args.out_tsv}", file=sys.stderr)
        return

    # convert dates/arrays
    samples = df["sample"].astype(str).to_numpy()
    n = samples.shape[0]

    baseline = df["baseline_date"].to_numpy(dtype="datetime64[ns]")
    censor_end = df["censor_end"].to_numpy(dtype="datetime64[ns]")
    t_censor = df["t_censor_days"].to_numpy(dtype=float)

    cov = df[COV_COLS].to_numpy(dtype=float)

    # protein matrix for this slice
    prot_df = df[protein_cols].apply(pd.to_numeric, errors="coerce")
    prot_mat = prot_df.to_numpy(dtype=float)  # shape: (n, p)

    # sample -> row index
    sample_to_idx = pd.Series(np.arange(n, dtype=int), index=samples)

    # --- ICD first occurrence ---
    icd = pd.read_csv(args.icd_first_tsv, sep="\t", usecols=["sample", "icd3", "first_date"])
    icd["sample"] = icd["sample"].astype(str)
    icd["first_date"] = pd.to_datetime(icd["first_date"], errors="coerce")
    icd = icd.dropna(subset=["sample", "icd3", "first_date"])
    icd = icd[icd["sample"].isin(sample_to_idx.index)].copy()

    if args.icd3_list is not None:
        with open(args.icd3_list, "r") as f:
            keep = {line.strip() for line in f if line.strip()}
        icd = icd[icd["icd3"].isin(keep)].copy()

    if icd.shape[0] == 0:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(args.out_tsv, sep="\t", index=False)
        print(f"[WARN] No ICD records after filtering. Wrote empty: {args.out_tsv}", file=sys.stderr)
        return

    # map to row indices
    icd_idx = sample_to_idx.loc[icd["sample"]].to_numpy(dtype=int)
    first_date = icd["first_date"].to_numpy(dtype="datetime64[ns]")
    icd3 = icd["icd3"].astype(str).to_numpy()

    # prevalent / incident flags
    is_prev = first_date <= baseline[icd_idx]
    is_inc = (first_date > baseline[icd_idx]) & (first_date <= censor_end[icd_idx])

    # event counts per icd3 (incident only)
    inc_codes, inc_counts = np.unique(icd3[is_inc], return_counts=True)
    codes = inc_codes[inc_counts >= args.min_events]

    if codes.size == 0:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(args.out_tsv, sep="\t", index=False)
        print("[WARN] No icd3 codes pass --min_events. Try lowering it.", file=sys.stderr)
        return

    # Pre-build index lists for each code (prev + inc)
    prev_idx_by_code = {}
    inc_idx_by_code = {}
    inc_date_by_code = {}

    for code in codes:
        m_code = (icd3 == code)
        prev_idx_by_code[code] = icd_idx[m_code & is_prev]
        inc_idx_by_code[code] = icd_idx[m_code & is_inc]
        inc_date_by_code[code] = first_date[m_code & is_inc]

    results = []

    # reusable buffers
    event_full = np.zeros(n, dtype=np.int8)
    duration_full = np.empty(n, dtype=float)

    for i, code in enumerate(codes, start=1):
        # at-risk = exclude prevalent
        at_risk = np.ones(n, dtype=bool)
        prev_idx = prev_idx_by_code[code]
        if prev_idx.size > 0:
            at_risk[np.unique(prev_idx)] = False

        # init (censor)
        duration_full[:] = t_censor
        event_full[:] = 0

        # set incident events
        inc_idx = inc_idx_by_code[code]
        inc_dt = inc_date_by_code[code]
        if inc_idx.size < args.min_events:
            continue

        # compute event times
        t_event = (inc_dt - baseline[inc_idx]).astype("timedelta64[D]").astype(float)
        valid = t_event > 0
        if valid.sum() < args.min_events:
            continue

        inc_idx = inc_idx[valid]
        t_event = t_event[valid]

        event_full[inc_idx] = 1
        duration_full[inc_idx] = t_event

        # final usable mask for this phenotype
        base_mask = at_risk & np.isfinite(duration_full) & (duration_full > 0)

        n_events = int(event_full[base_mask].sum())
        if n_events < args.min_events:
            continue

        # Fit Cox for each protein in the slice
        for pj, prot in enumerate(protein_cols):
            x = prot_mat[:, pj]
            m = base_mask & np.isfinite(x)
            if m.sum() < args.min_samples:
                continue

            endog = duration_full[m]
            status = event_full[m]
            exog = np.column_stack([x[m], cov[m]])

            # require enough events after removing missing protein
            if int(status.sum()) < args.min_events:
                continue

            try:
                fit = PHReg(endog, exog, status=status).fit(disp=0)
                beta = float(fit.params[0])
                se = float(fit.bse[0])
                p = float(fit.pvalues[0])

                hr = float(np.exp(beta))
                ci_l = float(np.exp(beta - 1.96 * se))
                ci_u = float(np.exp(beta + 1.96 * se))

                results.append([
                    code, prot, int(status.sum()), int(m.sum()),
                    beta, se, hr, ci_l, ci_u, p
                ])
            except Exception:
                # skip non-convergent / numerical issues
                continue

        if args.log_every > 0 and (i % args.log_every == 0):
            print(f"[PROGRESS] icd3 {i}/{len(codes)} done (code={code}), results so far={len(results)}",
                  file=sys.stderr)

    out = pd.DataFrame(results, columns=[
        "icd3", "protein", "n_events", "n_atrisk",
        "beta", "se", "HR", "CI95_L", "CI95_U", "p"
    ])
    out.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"[DONE] Wrote: {args.out_tsv} (rows={len(out)})", file=sys.stderr)


if __name__ == "__main__":
    main()
