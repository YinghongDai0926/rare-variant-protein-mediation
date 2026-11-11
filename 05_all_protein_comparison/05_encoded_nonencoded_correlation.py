#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_encoded_nonencoded_correlation.py

Purpose
-------
Assess whether the best non-encoded mediator is correlated with the encoded protein.

This step helps interpret stronger non-encoded mediator signals:
    1. shared-module / downstream signal
    2. network-extended signal

Inputs
------
1. 04_step5_correlation_runset_encoded_best_nonencoded.tsv
2. wes_olink_covariates_analytic.tsv

Outputs
-------
1. 05_encoded_nonencoded_correlation.tsv
2. 05_encoded_nonencoded_correlation_overall_summary.tsv
3. 05_encoded_nonencoded_correlation_by_priority_class.tsv
4. 05_figure_correlation_distribution.tsv
5. 05_figure_pathway_interpretation_summary.tsv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


DEFAULT_COVARIATES = ["age", "sex"] + [f"PC{i}" for i in range(1, 11)]


NUMERIC_COLS = [
    "encoded_rank_by_indirect_p",
    "encoded_indirect_p_analytic",
    "encoded_abs_indirect_logHR",
    "encoded_abs_PM",
    "best_nonencoded_indirect_p_analytic",
    "best_nonencoded_abs_indirect_logHR",
    "best_nonencoded_abs_PM",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Correlation analysis between encoded and best non-encoded mediators."
    )

    parser.add_argument(
        "--correlation_runset",
        required=True,
        help="Path to 04_step5_correlation_runset_encoded_best_nonencoded.tsv.",
    )

    parser.add_argument(
        "--covariate_protein_table",
        required=True,
        help="Path to wes_olink_covariates_analytic.tsv.",
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--prefix",
        default="05",
        help="Output prefix. Default: 05.",
    )

    parser.add_argument(
        "--covariates",
        nargs="+",
        default=DEFAULT_COVARIATES,
        help="Covariates used for residual correlation.",
    )

    parser.add_argument(
        "--min_complete_case",
        type=int,
        default=200,
        help="Minimum complete-case sample size for correlation analysis.",
    )

    parser.add_argument(
        "--high_corr_cutoff",
        type=float,
        default=0.7,
        help="Absolute correlation cutoff for high correlation.",
    )

    parser.add_argument(
        "--moderate_corr_cutoff",
        type=float,
        default=0.4,
        help="Absolute correlation cutoff for moderate correlation.",
    )

    return parser.parse_args()


def convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def require_columns(df: pd.DataFrame, cols: list):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def safe_corr(x, y, method="spearman"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan, np.nan

    if np.nanstd(x) <= 0 or np.nanstd(y) <= 0:
        return np.nan, np.nan

    if method == "spearman":
        r, p = spearmanr(x, y)
    elif method == "pearson":
        r, p = pearsonr(x, y)
    else:
        raise ValueError("Unknown correlation method.")

    return float(r), float(p)


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Residualize y on X using ordinary least squares.

    Scientific logic:
    Protein-protein correlation can be partly driven by age, sex, and ancestry PCs.
    Residual correlation removes these shared covariate effects before assessing
    whether encoded and non-encoded proteins are correlated.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    X = np.column_stack([np.ones(X.shape[0]), X])

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted

    return resid


def assign_correlation_group(abs_r, high_cutoff, moderate_cutoff):
    if pd.isna(abs_r) or not np.isfinite(abs_r):
        return "correlation_failed"

    if abs_r >= high_cutoff:
        return "high_correlation"

    if abs_r >= moderate_cutoff:
        return "moderate_correlation"

    return "low_correlation"


def assign_pathway_interpretation(priority_class, corr_group):
    priority_class = str(priority_class)
    corr_group = str(corr_group)

    if priority_class in {
        "encoded_top1",
        "encoded_top5_and_effect_leading",
        "encoded_top10_and_effect_leading",
    }:
        return "encoded_leading"

    if priority_class in {
        "nonencoded_stronger_but_encoded_top10",
        "nonencoded_leading",
    }:
        if corr_group in {"high_correlation", "moderate_correlation"}:
            return "shared_module_or_downstream"
        if corr_group == "low_correlation":
            return "network_extended"
        return "unclassified_nonencoded_stronger"

    return "unclassified"


def load_needed_columns(table_path: str, runset: pd.DataFrame, covariates: list) -> pd.DataFrame:
    proteins = sorted(
        set(runset["encoded_protein"].astype(str))
        | set(runset["nonencoded_protein"].astype(str))
    )

    usecols = ["sample"] + covariates + proteins

    df = pd.read_csv(
        table_path,
        sep="\t",
        usecols=usecols,
        dtype={"sample": str},
    )

    df["sample"] = df["sample"].astype(str)

    for col in covariates + proteins:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def run_correlation_for_pair(row, protein_df: pd.DataFrame, covariates: list, args):
    encoded = str(row["encoded_protein"])
    nonencoded = str(row["nonencoded_protein"])

    cols = [encoded, nonencoded] + covariates

    dat = protein_df.loc[:, cols].dropna().copy()

    out = {
        "pair_id": row["pair_id"],
        "gene": row["gene"],
        "icd3": row["icd3"],
        "encoded_protein": encoded,
        "nonencoded_protein": nonencoded,
        "tier_label": row.get("tier_label", ""),
        "encoded_rank_by_indirect_p": row.get("encoded_rank_by_indirect_p", np.nan),
        "encoded_abs_indirect_logHR": row.get("encoded_abs_indirect_logHR", np.nan),
        "encoded_abs_PM": row.get("encoded_abs_PM", np.nan),
        "best_nonencoded_abs_indirect_logHR": row.get("best_nonencoded_abs_indirect_logHR", np.nan),
        "best_nonencoded_abs_PM": row.get("best_nonencoded_abs_PM", np.nan),
        "encoded_vs_nonencoded_priority_class": row.get("encoded_vs_nonencoded_priority_class", ""),
        "n_complete_case_correlation": int(dat.shape[0]),
        "correlation_success": 0,
        "fail_reason": "",
    }

    if dat.shape[0] < args.min_complete_case:
        out["fail_reason"] = "low_complete_case_correlation"
        return out

    x = dat[encoded].to_numpy(dtype=float)
    y = dat[nonencoded].to_numpy(dtype=float)

    raw_spearman_r, raw_spearman_p = safe_corr(x, y, method="spearman")
    raw_pearson_r, raw_pearson_p = safe_corr(x, y, method="pearson")

    X_cov = dat[covariates].to_numpy(dtype=float)

    try:
        x_resid = residualize(x, X_cov)
        y_resid = residualize(y, X_cov)

        adjusted_spearman_r, adjusted_spearman_p = safe_corr(
            x_resid,
            y_resid,
            method="spearman",
        )

        adjusted_pearson_r, adjusted_pearson_p = safe_corr(
            x_resid,
            y_resid,
            method="pearson",
        )

    except Exception as e:
        out["fail_reason"] = f"residual_correlation_failed::{str(e)[:200]}"
        return out

    out.update(
        {
            "raw_spearman_r": raw_spearman_r,
            "raw_spearman_p": raw_spearman_p,
            "raw_abs_spearman_r": abs(raw_spearman_r) if np.isfinite(raw_spearman_r) else np.nan,

            "raw_pearson_r": raw_pearson_r,
            "raw_pearson_p": raw_pearson_p,
            "raw_abs_pearson_r": abs(raw_pearson_r) if np.isfinite(raw_pearson_r) else np.nan,

            "adjusted_spearman_r": adjusted_spearman_r,
            "adjusted_spearman_p": adjusted_spearman_p,
            "adjusted_abs_spearman_r": abs(adjusted_spearman_r)
            if np.isfinite(adjusted_spearman_r)
            else np.nan,

            "adjusted_pearson_r": adjusted_pearson_r,
            "adjusted_pearson_p": adjusted_pearson_p,
            "adjusted_abs_pearson_r": abs(adjusted_pearson_r)
            if np.isfinite(adjusted_pearson_r)
            else np.nan,
        }
    )

    out["correlation_group"] = assign_correlation_group(
        out["adjusted_abs_spearman_r"],
        high_cutoff=args.high_corr_cutoff,
        moderate_cutoff=args.moderate_corr_cutoff,
    )

    out["pathway_interpretation_class"] = assign_pathway_interpretation(
        out["encoded_vs_nonencoded_priority_class"],
        out["correlation_group"],
    )

    out["correlation_success"] = int(out["correlation_group"] != "correlation_failed")

    if out["correlation_success"] != 1:
        out["fail_reason"] = "nonfinite_correlation"

    return out


def summarize_overall(res: pd.DataFrame) -> pd.DataFrame:
    valid = res["correlation_success"].eq(1)

    rows = [
        ("n_pairs", res.shape[0]),
        ("n_correlation_success", int(valid.sum())),
        ("n_correlation_failed", int((~valid).sum())),
    ]

    for group, n in res.loc[valid, "correlation_group"].value_counts().items():
        rows.append((f"correlation_group::{group}::n", int(n)))
        rows.append((f"correlation_group::{group}::proportion", n / int(valid.sum())))

    for cls, n in res.loc[valid, "pathway_interpretation_class"].value_counts().items():
        rows.append((f"pathway_class::{cls}::n", int(n)))
        rows.append((f"pathway_class::{cls}::proportion", n / int(valid.sum())))

    nonencoded_stronger = res["encoded_vs_nonencoded_priority_class"].isin(
        ["nonencoded_stronger_but_encoded_top10", "nonencoded_leading"]
    )

    sub = res.loc[valid & nonencoded_stronger].copy()
    rows.append(("n_nonencoded_stronger_pairs", int(sub.shape[0])))

    if sub.shape[0] > 0:
        for group, n in sub["correlation_group"].value_counts().items():
            rows.append((f"nonencoded_stronger::correlation_group::{group}::n", int(n)))
            rows.append((f"nonencoded_stronger::correlation_group::{group}::proportion", n / sub.shape[0]))

        for cls, n in sub["pathway_interpretation_class"].value_counts().items():
            rows.append((f"nonencoded_stronger::pathway_class::{cls}::n", int(n)))
            rows.append((f"nonencoded_stronger::pathway_class::{cls}::proportion", n / sub.shape[0]))

    rows += [
        (
            "median_adjusted_abs_spearman_r",
            res.loc[valid, "adjusted_abs_spearman_r"].median(),
        ),
        (
            "median_raw_abs_spearman_r",
            res.loc[valid, "raw_abs_spearman_r"].median(),
        ),
    ]

    return pd.DataFrame(rows, columns=["metric", "value"])


def summarize_by_priority_class(res: pd.DataFrame) -> pd.DataFrame:
    valid = res["correlation_success"].eq(1)
    sub = res.loc[valid].copy()

    rows = []

    for cls, g in sub.groupby("encoded_vs_nonencoded_priority_class", dropna=False):
        n = g.shape[0]

        rows.append(
            {
                "encoded_vs_nonencoded_priority_class": cls,
                "n_pairs": n,
                "median_adjusted_abs_spearman_r": g["adjusted_abs_spearman_r"].median(),
                "median_raw_abs_spearman_r": g["raw_abs_spearman_r"].median(),
                "n_high_correlation": int((g["correlation_group"] == "high_correlation").sum()),
                "n_moderate_correlation": int((g["correlation_group"] == "moderate_correlation").sum()),
                "n_low_correlation": int((g["correlation_group"] == "low_correlation").sum()),
                "prop_high_correlation": (g["correlation_group"] == "high_correlation").mean(),
                "prop_moderate_correlation": (g["correlation_group"] == "moderate_correlation").mean(),
                "prop_low_correlation": (g["correlation_group"] == "low_correlation").mean(),
            }
        )

    return pd.DataFrame(rows)


def build_figure_tables(res: pd.DataFrame):
    fig_corr = res.loc[
        res["correlation_success"].eq(1),
        [
            "pair_id",
            "gene",
            "icd3",
            "encoded_protein",
            "nonencoded_protein",
            "encoded_vs_nonencoded_priority_class",
            "correlation_group",
            "pathway_interpretation_class",
            "raw_spearman_r",
            "raw_abs_spearman_r",
            "adjusted_spearman_r",
            "adjusted_abs_spearman_r",
            "encoded_rank_by_indirect_p",
            "encoded_abs_indirect_logHR",
            "best_nonencoded_abs_indirect_logHR",
            "encoded_abs_PM",
            "best_nonencoded_abs_PM",
        ],
    ].copy()

    fig_pathway = (
        res.loc[res["correlation_success"].eq(1), "pathway_interpretation_class"]
        .value_counts()
        .reset_index()
    )
    fig_pathway.columns = ["pathway_interpretation_class", "n_pairs"]
    fig_pathway["proportion"] = fig_pathway["n_pairs"] / fig_pathway["n_pairs"].sum()

    return fig_corr, fig_pathway


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runset = pd.read_csv(args.correlation_runset, sep="\t")
    runset = convert_numeric(runset)

    required = [
        "pair_id",
        "gene",
        "icd3",
        "encoded_protein",
        "nonencoded_protein",
        "encoded_vs_nonencoded_priority_class",
    ]
    require_columns(runset, required)

    protein_df = load_needed_columns(
        table_path=args.covariate_protein_table,
        runset=runset,
        covariates=args.covariates,
    )

    results = []

    for _, row in runset.iterrows():
        out = run_correlation_for_pair(
            row=row,
            protein_df=protein_df,
            covariates=args.covariates,
            args=args,
        )
        results.append(out)

    res = pd.DataFrame(results)

    overall = summarize_overall(res)
    by_priority = summarize_by_priority_class(res)
    fig_corr, fig_pathway = build_figure_tables(res)

    out_res = outdir / f"{args.prefix}_encoded_nonencoded_correlation.tsv"
    out_overall = outdir / f"{args.prefix}_encoded_nonencoded_correlation_overall_summary.tsv"
    out_by_priority = outdir / f"{args.prefix}_encoded_nonencoded_correlation_by_priority_class.tsv"
    out_fig_corr = outdir / f"{args.prefix}_figure_correlation_distribution.tsv"
    out_fig_pathway = outdir / f"{args.prefix}_figure_pathway_interpretation_summary.tsv"

    res.to_csv(out_res, sep="\t", index=False)
    overall.to_csv(out_overall, sep="\t", index=False)
    by_priority.to_csv(out_by_priority, sep="\t", index=False)
    fig_corr.to_csv(out_fig_corr, sep="\t", index=False)
    fig_pathway.to_csv(out_fig_pathway, sep="\t", index=False)

    print("[Done] Step5 encoded-nonencoded correlation analysis completed.")
    print(f"[Output] {out_res}")
    print(f"[Output] {out_overall}")
    print(f"[Output] {out_by_priority}")
    print(f"[Output] {out_fig_corr}")
    print(f"[Output] {out_fig_pathway}")

    print("[QC]")
    for _, r in overall.iterrows():
        print(f"{r['metric']}\t{r['value']}")


if __name__ == "__main__":
    main()
