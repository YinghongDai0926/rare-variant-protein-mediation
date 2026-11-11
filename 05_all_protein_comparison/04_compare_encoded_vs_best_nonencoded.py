#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_compare_encoded_vs_best_nonencoded.py

Purpose
-------
Compare encoded protein with the best non-encoded mediator within each
gene-disease pair.

This script does not fit new models. It uses Step3 ranking results.

Main outputs
------------
1. 04_encoded_vs_best_nonencoded_comparison.tsv
2. 04_encoded_vs_best_nonencoded_overall_summary.tsv
3. 04_encoded_vs_best_nonencoded_tier_summary.tsv
4. 04_step5_correlation_runset_encoded_best_nonencoded.tsv
5. Figure input tables
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


NUMERIC_COLS = [
    "n_tested_proteins_total",
    "n_tested_proteins_valid",

    "encoded_valid_for_ranking",
    "encoded_model_success",
    "encoded_n_complete_case",
    "encoded_n_events_complete_case",
    "encoded_n_carriers_complete_case",
    "encoded_n_carrier_events_complete_case",

    "encoded_indirect_logHR",
    "encoded_indirect_hr",
    "encoded_abs_indirect_logHR",
    "encoded_PM",
    "encoded_abs_PM",
    "encoded_indirect_p_analytic",
    "encoded_indirect_q_analytic_global",
    "encoded_indirect_q_analytic_within_pair",
    "encoded_rank_by_indirect_p",
    "encoded_rank_by_abs_indirect_logHR",
    "encoded_rank_by_abs_PM",
    "encoded_percentile_by_indirect_p",
    "encoded_percentile_by_abs_indirect_logHR",
    "encoded_percentile_by_abs_PM",

    "encoded_is_top1_by_indirect_p",
    "encoded_is_top5_by_indirect_p",
    "encoded_is_top10_by_indirect_p",
    "encoded_is_top1_by_abs_logHR",
    "encoded_is_top5_by_abs_logHR",
    "encoded_is_top10_by_abs_logHR",

    "best_nonencoded_indirect_logHR",
    "best_nonencoded_indirect_hr",
    "best_nonencoded_abs_indirect_logHR",
    "best_nonencoded_proportion_mediated_logscale",
    "best_nonencoded_abs_proportion_mediated_logscale",
    "best_nonencoded_indirect_p_analytic",
    "best_nonencoded_indirect_q_analytic_global",
    "best_nonencoded_indirect_q_analytic_within_pair",
    "best_nonencoded_rank_by_indirect_p",
    "best_nonencoded_rank_by_abs_indirect_logHR",
    "best_nonencoded_rank_by_abs_PM",
    "best_nonencoded_percentile_by_indirect_p",
    "best_nonencoded_percentile_by_abs_indirect_logHR",
    "best_nonencoded_percentile_by_abs_PM",
    "best_nonencoded_model_success",
    "best_nonencoded_n_complete_case",
    "best_nonencoded_n_events_complete_case",
    "best_nonencoded_n_carriers_complete_case",
    "best_nonencoded_n_carrier_events_complete_case",

    "encoded_stronger_than_best_nonencoded_by_abs_logHR",
    "encoded_stronger_than_best_nonencoded_by_abs_PM",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare encoded protein with best non-encoded mediator."
    )

    parser.add_argument(
        "--encoded_summary",
        required=True,
        help="Path to 03_encoded_protein_ranking_summary.tsv.",
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--prefix",
        default="04",
        help="Output prefix. Default: 04.",
    )

    return parser.parse_args()


def safe_divide(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    return np.where((b.notna()) & np.isfinite(b) & (b != 0), a / b, np.nan)


def convert_numeric(df):
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def add_comparison_metrics(df):
    df = df.copy()

    df["has_best_nonencoded"] = (
        df["best_nonencoded_mediator_protein"].notna()
        & df["best_nonencoded_mediator_protein"].astype(str).ne("")
    ).astype(int)

    df["encoded_minus_best_nonencoded_abs_logHR"] = (
        df["encoded_abs_indirect_logHR"]
        - df["best_nonencoded_abs_indirect_logHR"]
    )

    df["encoded_minus_best_nonencoded_abs_PM"] = (
        df["encoded_abs_PM"]
        - df["best_nonencoded_abs_proportion_mediated_logscale"]
    )

    df["encoded_to_best_nonencoded_abs_logHR_ratio"] = safe_divide(
        df["encoded_abs_indirect_logHR"],
        df["best_nonencoded_abs_indirect_logHR"],
    )

    df["encoded_to_best_nonencoded_abs_PM_ratio"] = safe_divide(
        df["encoded_abs_PM"],
        df["best_nonencoded_abs_proportion_mediated_logscale"],
    )

    df["encoded_p_better_than_best_nonencoded"] = (
        df["encoded_indirect_p_analytic"]
        <= df["best_nonencoded_indirect_p_analytic"]
    ).astype("Int64")

    df["encoded_abs_logHR_better_than_best_nonencoded"] = (
        df["encoded_abs_indirect_logHR"]
        >= df["best_nonencoded_abs_indirect_logHR"]
    ).astype("Int64")

    df["encoded_abs_PM_better_than_best_nonencoded"] = (
        df["encoded_abs_PM"]
        >= df["best_nonencoded_abs_proportion_mediated_logscale"]
    ).astype("Int64")

    return df


def assign_rank_category(row):
    if row.get("encoded_valid_for_ranking", 0) != 1:
        return "encoded_invalid"

    rank = row.get("encoded_rank_by_indirect_p", np.nan)

    if pd.isna(rank):
        return "encoded_invalid"
    if rank == 1:
        return "top1"
    if rank <= 5:
        return "top2_5"
    if rank <= 10:
        return "top6_10"
    return "outside_top10"


def assign_priority_class(row):
    if row.get("encoded_valid_for_ranking", 0) != 1:
        return "encoded_invalid"

    rank = row.get("encoded_rank_by_indirect_p", np.nan)

    if pd.isna(rank):
        return "encoded_invalid"

    encoded_better_loghr = row.get(
        "encoded_abs_logHR_better_than_best_nonencoded",
        pd.NA,
    )

    encoded_better_pm = row.get(
        "encoded_abs_PM_better_than_best_nonencoded",
        pd.NA,
    )

    if rank == 1:
        return "encoded_top1"

    if rank <= 5 and encoded_better_loghr == 1:
        return "encoded_top5_and_effect_leading"

    if rank <= 10 and encoded_better_loghr == 1:
        return "encoded_top10_and_effect_leading"

    if rank <= 10 and (encoded_better_loghr == 0 or encoded_better_pm == 0):
        return "nonencoded_stronger_but_encoded_top10"

    if rank > 10:
        return "nonencoded_leading"

    return "unclassified"


def add_categories(df):
    df = df.copy()

    df["encoded_rank_category_by_indirect_p"] = df.apply(
        assign_rank_category,
        axis=1,
    )

    df["encoded_vs_nonencoded_priority_class"] = df.apply(
        assign_priority_class,
        axis=1,
    )

    return df


def summarize_overall(df):
    valid = df["encoded_valid_for_ranking"].eq(1)

    def n_true(col):
        if col not in df.columns:
            return np.nan
        return int(df.loc[valid, col].fillna(0).astype(int).sum())

    def prop_true(col):
        denom = int(valid.sum())
        if denom == 0:
            return np.nan
        return n_true(col) / denom

    rows = [
        ("n_gene_disease_pairs", df["pair_id"].nunique()),
        ("n_encoded_valid_for_ranking", int(valid.sum())),
        ("n_has_best_nonencoded", int(df.loc[valid, "has_best_nonencoded"].sum())),

        ("n_encoded_top1_by_indirect_p", n_true("encoded_is_top1_by_indirect_p")),
        ("n_encoded_top5_by_indirect_p", n_true("encoded_is_top5_by_indirect_p")),
        ("n_encoded_top10_by_indirect_p", n_true("encoded_is_top10_by_indirect_p")),
        ("prop_encoded_top1_by_indirect_p", prop_true("encoded_is_top1_by_indirect_p")),
        ("prop_encoded_top5_by_indirect_p", prop_true("encoded_is_top5_by_indirect_p")),
        ("prop_encoded_top10_by_indirect_p", prop_true("encoded_is_top10_by_indirect_p")),

        (
            "n_encoded_abs_logHR_better_than_best_nonencoded",
            n_true("encoded_abs_logHR_better_than_best_nonencoded"),
        ),
        (
            "prop_encoded_abs_logHR_better_than_best_nonencoded",
            prop_true("encoded_abs_logHR_better_than_best_nonencoded"),
        ),
        (
            "n_encoded_abs_PM_better_than_best_nonencoded",
            n_true("encoded_abs_PM_better_than_best_nonencoded"),
        ),
        (
            "prop_encoded_abs_PM_better_than_best_nonencoded",
            prop_true("encoded_abs_PM_better_than_best_nonencoded"),
        ),

        (
            "median_encoded_rank_by_indirect_p",
            df.loc[valid, "encoded_rank_by_indirect_p"].median(),
        ),
        (
            "median_encoded_percentile_by_indirect_p",
            df.loc[valid, "encoded_percentile_by_indirect_p"].median(),
        ),
        (
            "median_encoded_abs_indirect_logHR",
            df.loc[valid, "encoded_abs_indirect_logHR"].median(),
        ),
        (
            "median_best_nonencoded_abs_indirect_logHR",
            df.loc[valid, "best_nonencoded_abs_indirect_logHR"].median(),
        ),
        (
            "median_encoded_abs_PM",
            df.loc[valid, "encoded_abs_PM"].median(),
        ),
        (
            "median_best_nonencoded_abs_PM",
            df.loc[valid, "best_nonencoded_abs_proportion_mediated_logscale"].median(),
        ),
        (
            "median_encoded_to_best_nonencoded_abs_logHR_ratio",
            np.nanmedian(df.loc[valid, "encoded_to_best_nonencoded_abs_logHR_ratio"]),
        ),
        (
            "median_encoded_to_best_nonencoded_abs_PM_ratio",
            np.nanmedian(df.loc[valid, "encoded_to_best_nonencoded_abs_PM_ratio"]),
        ),
    ]

    out = pd.DataFrame(rows, columns=["metric", "value"])

    class_counts = (
        df.loc[valid, "encoded_vs_nonencoded_priority_class"]
        .value_counts(dropna=False)
        .reset_index()
    )
    class_counts.columns = ["priority_class", "n"]

    denom = int(valid.sum())
    class_counts["proportion"] = class_counts["n"] / denom if denom > 0 else np.nan

    for _, r in class_counts.iterrows():
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [
                        {
                            "metric": f"priority_class::{r['priority_class']}::n",
                            "value": r["n"],
                        },
                        {
                            "metric": f"priority_class::{r['priority_class']}::proportion",
                            "value": r["proportion"],
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )

    return out


def summarize_by_tier(df):
    if "tier_label" not in df.columns:
        return pd.DataFrame()

    valid = df["encoded_valid_for_ranking"].eq(1)
    sub = df.loc[valid].copy()

    if sub.empty:
        return pd.DataFrame()

    rows = []

    for tier, g in sub.groupby("tier_label", dropna=False):
        n = g.shape[0]

        rows.append(
            {
                "tier_label": tier,
                "n_pairs": n,
                "n_encoded_top1_by_indirect_p": int(g["encoded_is_top1_by_indirect_p"].sum()),
                "prop_encoded_top1_by_indirect_p": g["encoded_is_top1_by_indirect_p"].mean(),
                "n_encoded_top5_by_indirect_p": int(g["encoded_is_top5_by_indirect_p"].sum()),
                "prop_encoded_top5_by_indirect_p": g["encoded_is_top5_by_indirect_p"].mean(),
                "n_encoded_top10_by_indirect_p": int(g["encoded_is_top10_by_indirect_p"].sum()),
                "prop_encoded_top10_by_indirect_p": g["encoded_is_top10_by_indirect_p"].mean(),
                "n_encoded_abs_logHR_better_than_best_nonencoded": int(
                    g["encoded_abs_logHR_better_than_best_nonencoded"].sum()
                ),
                "prop_encoded_abs_logHR_better_than_best_nonencoded": g[
                    "encoded_abs_logHR_better_than_best_nonencoded"
                ].mean(),
                "median_encoded_rank_by_indirect_p": g["encoded_rank_by_indirect_p"].median(),
                "median_encoded_percentile_by_indirect_p": g[
                    "encoded_percentile_by_indirect_p"
                ].median(),
                "median_encoded_abs_indirect_logHR": g["encoded_abs_indirect_logHR"].median(),
                "median_best_nonencoded_abs_indirect_logHR": g[
                    "best_nonencoded_abs_indirect_logHR"
                ].median(),
                "median_encoded_abs_PM": g["encoded_abs_PM"].median(),
                "median_best_nonencoded_abs_PM": g[
                    "best_nonencoded_abs_proportion_mediated_logscale"
                ].median(),
            }
        )

    return pd.DataFrame(rows)


def build_step5_correlation_runset(df):
    keep = [
        "pair_id",
        "gene",
        "icd3",
        "encoded_protein",
        "best_nonencoded_mediator_protein",
        "tier_label",
        "encoded_rank_by_indirect_p",
        "encoded_indirect_p_analytic",
        "encoded_abs_indirect_logHR",
        "encoded_abs_PM",
        "best_nonencoded_indirect_p_analytic",
        "best_nonencoded_abs_indirect_logHR",
        "best_nonencoded_abs_proportion_mediated_logscale",
        "encoded_vs_nonencoded_priority_class",
    ]

    keep = [c for c in keep if c in df.columns]

    out = df.loc[
        df["encoded_valid_for_ranking"].eq(1)
        & df["has_best_nonencoded"].eq(1),
        keep,
    ].copy()

    out = out.rename(
        columns={
            "best_nonencoded_mediator_protein": "nonencoded_protein",
            "best_nonencoded_abs_proportion_mediated_logscale": "best_nonencoded_abs_PM",
        }
    )

    return out


def build_figure_inputs(df):
    valid = df["encoded_valid_for_ranking"].eq(1)
    sub = df.loc[valid].copy()

    scatter_cols = [
        "pair_id",
        "gene",
        "icd3",
        "encoded_protein",
        "tier_label",
        "encoded_abs_indirect_logHR",
        "best_nonencoded_abs_indirect_logHR",
        "encoded_abs_PM",
        "best_nonencoded_abs_proportion_mediated_logscale",
        "encoded_rank_by_indirect_p",
        "encoded_percentile_by_indirect_p",
        "encoded_vs_nonencoded_priority_class",
    ]

    scatter = sub.loc[:, [c for c in scatter_cols if c in sub.columns]].copy()

    rank_class = (
        sub["encoded_rank_category_by_indirect_p"]
        .value_counts()
        .reset_index()
    )
    rank_class.columns = ["encoded_rank_category_by_indirect_p", "n_pairs"]
    rank_class["proportion"] = rank_class["n_pairs"] / sub.shape[0]

    priority_class = (
        sub["encoded_vs_nonencoded_priority_class"]
        .value_counts()
        .reset_index()
    )
    priority_class.columns = ["encoded_vs_nonencoded_priority_class", "n_pairs"]
    priority_class["proportion"] = priority_class["n_pairs"] / sub.shape[0]

    return scatter, rank_class, priority_class


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.encoded_summary, sep="\t")
    df = convert_numeric(df)

    required = [
        "pair_id",
        "gene",
        "icd3",
        "encoded_protein",
        "encoded_valid_for_ranking",
        "encoded_rank_by_indirect_p",
        "encoded_abs_indirect_logHR",
        "encoded_abs_PM",
        "encoded_indirect_p_analytic",
        "best_nonencoded_mediator_protein",
        "best_nonencoded_abs_indirect_logHR",
        "best_nonencoded_abs_proportion_mediated_logscale",
        "best_nonencoded_indirect_p_analytic",
    ]
    require_columns(df, required)

    df = add_comparison_metrics(df)
    df = add_categories(df)

    overall = summarize_overall(df)
    tier_summary = summarize_by_tier(df)
    step5_runset = build_step5_correlation_runset(df)
    scatter, rank_class, priority_class = build_figure_inputs(df)

    out_comparison = outdir / f"{args.prefix}_encoded_vs_best_nonencoded_comparison.tsv"
    out_overall = outdir / f"{args.prefix}_encoded_vs_best_nonencoded_overall_summary.tsv"
    out_tier = outdir / f"{args.prefix}_encoded_vs_best_nonencoded_tier_summary.tsv"
    out_step5 = outdir / f"{args.prefix}_step5_correlation_runset_encoded_best_nonencoded.tsv"
    out_scatter = outdir / f"{args.prefix}_figure_encoded_vs_best_nonencoded_scatter.tsv"
    out_rank_class = outdir / f"{args.prefix}_figure_encoded_rank_category_summary.tsv"
    out_priority_class = outdir / f"{args.prefix}_figure_encoded_priority_class_summary.tsv"

    df.to_csv(out_comparison, sep="\t", index=False)
    overall.to_csv(out_overall, sep="\t", index=False)
    tier_summary.to_csv(out_tier, sep="\t", index=False)
    step5_runset.to_csv(out_step5, sep="\t", index=False)
    scatter.to_csv(out_scatter, sep="\t", index=False)
    rank_class.to_csv(out_rank_class, sep="\t", index=False)
    priority_class.to_csv(out_priority_class, sep="\t", index=False)

    print("[Done] Step4 encoded vs best non-encoded comparison completed.")
    print(f"[Output] {out_comparison}")
    print(f"[Output] {out_overall}")
    print(f"[Output] {out_tier}")
    print(f"[Output] {out_step5}")
    print(f"[Output] {out_scatter}")
    print(f"[Output] {out_rank_class}")
    print(f"[Output] {out_priority_class}")

    print("[QC]")
    for _, r in overall.iterrows():
        print(f"{r['metric']}\t{r['value']}")


if __name__ == "__main__":
    main()
