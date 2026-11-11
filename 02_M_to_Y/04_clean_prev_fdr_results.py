#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(
        description="Clean prev_proteome_phenome results by removing numerically anomalous rows while retaining valid extreme-significance underflow rows."
    )
    ap.add_argument("--all_tsv", required=True, help="prev_proteome_phenome.all.tsv")
    ap.add_argument("--orig_fdr_tsv", required=True, help="prev_proteome_phenome.fdr_q_lt0.05.tsv")
    ap.add_argument("--out_cleaned_tsv", required=True, help="Output cleaned FDR file")
    ap.add_argument("--out_anomaly_tsv", default=None, help="Output anomaly rows TSV")
    ap.add_argument("--out_qc_txt", default=None, help="Output QC text file")
    ap.add_argument("--beta_abs_hard_cut", type=float, default=10.0,
                    help="Hard cutoff for absolute beta to define anomaly (default 10)")
    ap.add_argument("--se_hard_cut", type=float, default=5.0,
                    help="Hard cutoff for SE to define anomaly (default 5)")
    ap.add_argument("--tol_log_or", type=float, default=5e-3,
                    help="Tolerance for consistency between beta and log(OR)")
    ap.add_argument("--tol_ci", type=float, default=5e-2,
                    help="Tolerance for consistency between beta±1.96*se and log(CI)")
    ap.add_argument("--z_underflow_min", type=float, default=30.0,
                    help="Minimum |beta/se| to accept p=0 as likely numeric underflow of an extremely significant valid result")
    return ap.parse_args()


def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def finite(x):
    return pd.notna(x) and np.isfinite(x)


def classify_row(row, beta_abs_hard_cut, se_hard_cut, tol_log_or, tol_ci, z_underflow_min):
    """
    Return:
      - anomaly_flag: bool
      - classification: str
      - reasons: list[str]
    """
    reasons = []

    beta = row["beta"]
    se = row["se"]
    or_ = row["OR"]
    ci_l = row["CI95_L"]
    ci_u = row["CI95_U"]
    p = row["p"]
    fdr_q = row["fdr_q"]
    bonf_q = row["bonf_q"]
    n_cases = row["n_cases"]
    n_controls = row["n_controls"]

    required = {
        "beta": beta, "se": se, "OR": or_, "CI95_L": ci_l, "CI95_U": ci_u,
        "p": p, "fdr_q": fdr_q, "bonf_q": bonf_q, "n_cases": n_cases, "n_controls": n_controls
    }
    for k, v in required.items():
        if not finite(v):
            reasons.append(f"nonfinite_{k}")

    if reasons:
        return True, "model_anomaly", reasons

    if n_cases <= 0:
        reasons.append("n_cases_le_0")
    if n_controls <= 0:
        reasons.append("n_controls_le_0")

    if se <= 0:
        reasons.append("se_le_0")
    if p < 0 or p > 1:
        reasons.append("p_out_of_range")
    if fdr_q < 0 or fdr_q > 1:
        reasons.append("fdr_out_of_range")
    if bonf_q < 0 or bonf_q > 1:
        reasons.append("bonf_out_of_range")

    if or_ <= 0:
        reasons.append("or_le_0")
    if ci_l <= 0:
        reasons.append("ci_l_le_0")
    if ci_u <= 0:
        reasons.append("ci_u_le_0")
    if ci_l > ci_u:
        reasons.append("ci_l_gt_ci_u")

    if abs(beta) > beta_abs_hard_cut:
        reasons.append("abs_beta_extreme")
    if se > se_hard_cut:
        reasons.append("se_extreme")

    # consistency checks on log scale
    if or_ > 0:
        log_or = math.log(or_)
        if abs(log_or - beta) > tol_log_or:
            reasons.append("beta_or_inconsistent")

    if ci_l > 0 and ci_u > 0:
        target_l = beta - 1.96 * se
        target_u = beta + 1.96 * se
        obs_l = math.log(ci_l)
        obs_u = math.log(ci_u)
        if abs(obs_l - target_l) > tol_ci or abs(obs_u - target_u) > tol_ci:
            reasons.append("beta_se_ci_inconsistent")

    # p=0 handling: distinguish valid underflow from anomaly
    z_abs = np.inf if se == 0 else abs(beta / se)

    if p == 0:
        if (or_ == 0) or (ci_l == 0) or (ci_u == 0):
            reasons.append("p0_with_zero_effect_metrics")
        elif z_abs < z_underflow_min:
            reasons.append("p0_but_z_not_extreme_enough")

    anomaly_flag = len(reasons) > 0

    if not anomaly_flag:
        if p == 0:
            classification = "extreme_significance_underflow_valid"
        else:
            classification = "normal_valid"
    else:
        classification = "model_anomaly"

    return anomaly_flag, classification, reasons


def summarize_anomalies(df_anom):
    if df_anom.empty:
        return {
            "n_anomaly_rows": 0,
            "median_n_cases": "NA",
            "median_n_controls": "NA",
            "median_abs_beta": "NA",
            "median_se": "NA",
            "median_or": "NA",
            "n_p_eq_0": 0
        }
    return {
        "n_anomaly_rows": int(df_anom.shape[0]),
        "median_n_cases": float(df_anom["n_cases"].median()),
        "median_n_controls": float(df_anom["n_controls"].median()),
        "median_abs_beta": float(df_anom["beta"].abs().median()),
        "median_se": float(df_anom["se"].median()),
        "median_or": float(df_anom["OR"].median()),
        "n_p_eq_0": int((df_anom["p"] == 0).sum())
    }


def main():
    args = parse_args()

    out_cleaned = args.out_cleaned_tsv
    if args.out_anomaly_tsv is None:
        args.out_anomaly_tsv = out_cleaned.replace(".tsv", ".anomalies.tsv")
    if args.out_qc_txt is None:
        args.out_qc_txt = out_cleaned.replace(".tsv", ".qc.txt")

    df_all = pd.read_csv(args.all_tsv, sep="\t")
    df_orig_fdr = pd.read_csv(args.orig_fdr_tsv, sep="\t")

    required_cols = ["icd3", "protein", "n_cases", "n_controls", "beta", "se", "OR", "CI95_L", "CI95_U", "p", "fdr_q", "bonf_q"]
    missing = [c for c in required_cols if c not in df_all.columns]
    if missing:
        raise ValueError(f"Missing columns in all_tsv: {missing}")

    df_all = safe_numeric(df_all, ["n_cases", "n_controls", "beta", "se", "OR", "CI95_L", "CI95_U", "p", "fdr_q", "bonf_q"])
    df_all["row_id"] = np.arange(df_all.shape[0])

    # classify all rows
    records = []
    for _, row in df_all.iterrows():
        anomaly_flag, classification, reasons = classify_row(
            row=row,
            beta_abs_hard_cut=args.beta_abs_hard_cut,
            se_hard_cut=args.se_hard_cut,
            tol_log_or=args.tol_log_or,
            tol_ci=args.tol_ci,
            z_underflow_min=args.z_underflow_min
        )
        records.append((anomaly_flag, classification, ";".join(reasons)))

    df_all["is_anomaly"] = [r[0] for r in records]
    df_all["classification"] = [r[1] for r in records]
    df_all["anomaly_reasons"] = [r[2] for r in records]

    # cleaned FDR rows
    df_cleaned_fdr = df_all[(df_all["fdr_q"] < 0.05) & (~df_all["is_anomaly"])].copy()
    df_cleaned_fdr.sort_values(["fdr_q", "p"], ascending=[True, True], inplace=True)
    df_cleaned_fdr.to_csv(out_cleaned, sep="\t", index=False)

    # anomaly rows among all
    df_anom_all = df_all[df_all["is_anomaly"]].copy()
    df_anom_all.sort_values(["fdr_q", "p"], ascending=[True, True], inplace=True)
    df_anom_all.to_csv(args.out_anomaly_tsv, sep="\t", index=False)

    # compare with original FDR file
    key_cols = ["icd3", "protein"]
    for k in key_cols:
        if k not in df_orig_fdr.columns:
            raise ValueError(f"Missing {k} in orig_fdr_tsv")

    orig_keys = df_orig_fdr[key_cols].drop_duplicates().copy()
    cleaned_keys = df_cleaned_fdr[key_cols].drop_duplicates().copy()

    removed_from_orig = orig_keys.merge(cleaned_keys, on=key_cols, how="left", indicator=True)
    removed_from_orig = removed_from_orig[removed_from_orig["_merge"] == "left_only"][key_cols].copy()

    df_removed = df_orig_fdr.merge(removed_from_orig, on=key_cols, how="inner")
    df_removed = df_removed.merge(
        df_all[key_cols + ["classification", "anomaly_reasons"]],
        on=key_cols,
        how="left"
    )

    # p=0 rows among original fdr
    n_orig_fdr_p0 = int((pd.to_numeric(df_orig_fdr["p"], errors="coerce") == 0).sum())
    valid_underflow_rows = df_all[(df_all["classification"] == "extreme_significance_underflow_valid") & (df_all["fdr_q"] < 0.05)].copy()
    anomaly_fdr_rows = df_all[(df_all["is_anomaly"]) & (df_all["fdr_q"] < 0.05)].copy()

    # reason breakdown
    reason_counts = {}
    if not anomaly_fdr_rows.empty:
        for rs in anomaly_fdr_rows["anomaly_reasons"].fillna(""):
            for r in str(rs).split(";"):
                r = r.strip()
                if r:
                    reason_counts[r] = reason_counts.get(r, 0) + 1

    summary_all = summarize_anomalies(df_anom_all)
    summary_fdr = summarize_anomalies(anomaly_fdr_rows)

    with open(args.out_qc_txt, "w") as f:
        f.write("QC comparison for prevalent proteome-phenome cleaned results\n")
        f.write("===========================================================\n\n")
        f.write(f"Input all.tsv                : {args.all_tsv}\n")
        f.write(f"Input original FDR file      : {args.orig_fdr_tsv}\n")
        f.write(f"Output cleaned FDR file      : {out_cleaned}\n")
        f.write(f"Output anomaly rows TSV      : {args.out_anomaly_tsv}\n\n")

        f.write("[Counts]\n")
        f.write(f"Total rows in all.tsv                        : {df_all.shape[0]:,}\n")
        f.write(f"Original rows in fdr_q<0.05 file             : {df_orig_fdr.shape[0]:,}\n")
        f.write(f"Rows in cleaned fdr_q<0.05 file              : {df_cleaned_fdr.shape[0]:,}\n")
        f.write(f"Rows removed from original FDR file          : {df_removed.shape[0]:,}\n")
        f.write(f"Total anomaly rows in all.tsv                : {df_anom_all.shape[0]:,}\n")
        f.write(f"Anomaly rows among original/eligible FDR set : {anomaly_fdr_rows.shape[0]:,}\n\n")

        f.write("[p=0 distinction]\n")
        f.write(f"Rows with p==0 in original FDR file                    : {n_orig_fdr_p0:,}\n")
        f.write(f"Rows classified as valid extreme-significance underflow: {valid_underflow_rows.shape[0]:,}\n")
        f.write(f"Rows classified as anomalous among FDR rows            : {anomaly_fdr_rows.shape[0]:,}\n\n")

        f.write("[Anomaly feature summary - all anomalies]\n")
        for k, v in summary_all.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("[Anomaly feature summary - FDR anomalies only]\n")
        for k, v in summary_fdr.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("[Anomaly reason breakdown among FDR anomalies]\n")
        if reason_counts:
            for k, v in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"{k}: {v}\n")
        else:
            f.write("No anomalous FDR rows detected.\n")
        f.write("\n")

        if not df_removed.empty:
            f.write("[Removed rows characteristics]\n")
            f.write(f"Median n_cases    : {pd.to_numeric(df_removed['n_cases'], errors='coerce').median()}\n")
            f.write(f"Median n_controls : {pd.to_numeric(df_removed['n_controls'], errors='coerce').median()}\n")
            f.write(f"Median |beta|     : {pd.to_numeric(df_removed['beta'], errors='coerce').abs().median()}\n")
            f.write(f"Median se         : {pd.to_numeric(df_removed['se'], errors='coerce').median()}\n")
            f.write(f"Median OR         : {pd.to_numeric(df_removed['OR'], errors='coerce').median()}\n\n")

            f.write("[Top 20 removed rows by |beta|]\n")
            tmp = df_removed.copy()
            tmp["abs_beta"] = pd.to_numeric(tmp["beta"], errors="coerce").abs()
            top20 = tmp.sort_values("abs_beta", ascending=False).head(20)
            cols_show = ["icd3", "protein", "n_cases", "n_controls", "beta", "se", "OR", "CI95_L", "CI95_U", "p", "fdr_q", "classification", "anomaly_reasons"]
            f.write(top20[cols_show].to_string(index=False))
            f.write("\n")
        else:
            f.write("[Removed rows characteristics]\nNo rows removed from original FDR file.\n")

    print(f"[DONE] cleaned FDR rows = {df_cleaned_fdr.shape[0]:,}")
    print(f"[DONE] anomaly rows (all) = {df_anom_all.shape[0]:,}")
    print(f"[DONE] anomaly rows among FDR = {anomaly_fdr_rows.shape[0]:,}")
    print(f"[DONE] cleaned file written to: {out_cleaned}")
    print(f"[DONE] anomaly rows written to: {args.out_anomaly_tsv}")
    print(f"[DONE] QC text written to: {args.out_qc_txt}")


if __name__ == "__main__":
    main()
