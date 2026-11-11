#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_final_pathway_classification.py

Purpose
-------
Integrate Step3-Step7 results and assign final biological pathway classes.

Inputs
------
1. 04_encoded_vs_best_nonencoded_comparison.tsv
2. 05_encoded_nonencoded_correlation.tsv
3. 06_conditional_mediation_encoded_vs_nonencoded.tsv
4. 07_residual_mediation_pair_summary.tsv

Outputs
-------
1. 08_final_pathway_classification.tsv
2. 08_final_pathway_classification_overall_summary.tsv
3. 08_final_pathway_classification_by_tier.tsv
4. 08_figure_final_pathway_class_summary.tsv
5. 08_figure_final_coarse_class_summary.tsv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


NUMERIC_COLS = [
    "encoded_rank_by_indirect_p",
    "encoded_percentile_by_indirect_p",
    "encoded_abs_indirect_logHR",
    "best_nonencoded_abs_indirect_logHR",
    "encoded_abs_PM",
    "best_nonencoded_abs_proportion_mediated_logscale",
    "encoded_abs_logHR_better_than_best_nonencoded",
    "encoded_abs_PM_better_than_best_nonencoded",

    "adjusted_abs_spearman_r",
    "raw_abs_spearman_r",

    "conditional_model_success",
    "encoded_conditional_indirect_p",
    "nonencoded_conditional_indirect_p",
    "encoded_conditional_abs_indirect_logHR",
    "nonencoded_conditional_abs_indirect_logHR",

    "encoded_specific_success",
    "nonencoded_specific_success",
    "encoded_specific_indirect_p",
    "nonencoded_specific_indirect_p",
    "encoded_specific_abs_indirect_logHR",
    "nonencoded_specific_abs_indirect_logHR",
    "encoded_specific_retained_nominal",
    "nonencoded_specific_retained_nominal",
    "encoded_residualization_r2_after_nonencoded",
    "nonencoded_residualization_r2_after_encoded",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final pathway classification for encoded vs non-encoded mediation analysis."
    )

    parser.add_argument("--step4_comparison", required=True)
    parser.add_argument("--step5_correlation", required=True)
    parser.add_argument("--step6_conditional", required=True)
    parser.add_argument("--step7_residual_pair", required=True)

    parser.add_argument("--outdir", required=True)
    parser.add_argument("--prefix", default="08")

    return parser.parse_args()


def read_table(path):
    return pd.read_csv(path, sep="\t", dtype=str)


def convert_numeric(df):
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pick_existing(df, cols):
    return [c for c in cols if c in df.columns]


def require_cols(df, cols, table_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} missing required columns: {', '.join(missing)}")


def prepare_step4(path):
    df = read_table(path)

    require_cols(
        df,
        [
            "pair_id",
            "gene",
            "icd3",
            "encoded_protein",
            "encoded_vs_nonencoded_priority_class",
            "encoded_rank_by_indirect_p",
        ],
        "Step4 comparison",
    )

    if "best_nonencoded_mediator_protein" in df.columns and "nonencoded_protein" not in df.columns:
        df = df.rename(columns={"best_nonencoded_mediator_protein": "nonencoded_protein"})

    keep = [
        "pair_id",
        "gene",
        "icd3",
        "encoded_protein",
        "nonencoded_protein",
        "tier_label",
        "original_trio_id",
        "encoded_vs_nonencoded_priority_class",
        "encoded_rank_category_by_indirect_p",
        "encoded_rank_by_indirect_p",
        "encoded_percentile_by_indirect_p",
        "encoded_abs_indirect_logHR",
        "best_nonencoded_abs_indirect_logHR",
        "encoded_abs_PM",
        "best_nonencoded_abs_proportion_mediated_logscale",
        "encoded_abs_logHR_better_than_best_nonencoded",
        "encoded_abs_PM_better_than_best_nonencoded",
    ]

    return df.loc[:, pick_existing(df, keep)].drop_duplicates("pair_id").copy()


def prepare_step5(path):
    df = read_table(path)

    require_cols(
        df,
        [
            "pair_id",
            "correlation_group",
            "pathway_interpretation_class",
        ],
        "Step5 correlation",
    )

    keep = [
        "pair_id",
        "correlation_group",
        "pathway_interpretation_class",
        "raw_abs_spearman_r",
        "adjusted_abs_spearman_r",
        "raw_spearman_r",
        "adjusted_spearman_r",
    ]

    return df.loc[:, pick_existing(df, keep)].drop_duplicates("pair_id").copy()


def prepare_step6(path):
    df = read_table(path)

    require_cols(
        df,
        [
            "pair_id",
            "conditional_model_success",
            "conditional_mediation_class",
        ],
        "Step6 conditional",
    )

    keep = [
        "pair_id",
        "conditional_model_success",
        "conditional_mediation_class",
        "encoded_conditional_indirect_p",
        "nonencoded_conditional_indirect_p",
        "encoded_conditional_abs_indirect_logHR",
        "nonencoded_conditional_abs_indirect_logHR",
        "encoded_conditional_indirect_hr",
        "nonencoded_conditional_indirect_hr",
    ]

    return df.loc[:, pick_existing(df, keep)].drop_duplicates("pair_id").copy()


def prepare_step7(path):
    df = read_table(path)

    require_cols(
        df,
        [
            "pair_id",
            "residual_specific_class",
        ],
        "Step7 residual pair summary",
    )

    keep = [
        "pair_id",
        "residual_specific_class",
        "encoded_specific_success",
        "nonencoded_specific_success",
        "encoded_specific_indirect_p",
        "nonencoded_specific_indirect_p",
        "encoded_specific_abs_indirect_logHR",
        "nonencoded_specific_abs_indirect_logHR",
        "encoded_specific_indirect_hr",
        "nonencoded_specific_indirect_hr",
        "encoded_specific_retained_nominal",
        "nonencoded_specific_retained_nominal",
        "encoded_residualization_r2_after_nonencoded",
        "nonencoded_residualization_r2_after_encoded",
    ]

    return df.loc[:, pick_existing(df, keep)].drop_duplicates("pair_id").copy()


def is_encoded_priority(priority_class):
    return str(priority_class) in {
        "encoded_top1",
        "encoded_top5_and_effect_leading",
        "encoded_top10_and_effect_leading",
    }


def is_nonencoded_stronger_top10(priority_class):
    return str(priority_class) == "nonencoded_stronger_but_encoded_top10"


def is_nonencoded_leading(priority_class):
    return str(priority_class) == "nonencoded_leading"


def bool_from_value(x):
    if pd.isna(x):
        return False
    try:
        return float(x) == 1.0
    except Exception:
        return str(x).strip().lower() in {"true", "t", "yes", "y"}


def add_flags(df):
    df = df.copy()

    df["flag_encoded_priority"] = df["encoded_vs_nonencoded_priority_class"].apply(
        is_encoded_priority
    )

    df["flag_nonencoded_stronger_top10"] = df["encoded_vs_nonencoded_priority_class"].apply(
        is_nonencoded_stronger_top10
    )

    df["flag_nonencoded_leading"] = df["encoded_vs_nonencoded_priority_class"].apply(
        is_nonencoded_leading
    )

    df["flag_low_correlation"] = df["correlation_group"].astype(str).eq("low_correlation")
    df["flag_high_or_moderate_correlation"] = df["correlation_group"].astype(str).isin(
        ["high_correlation", "moderate_correlation"]
    )

    df["flag_encoded_specific_retained"] = df["residual_specific_class"].astype(str).isin(
        ["both_specific_retained", "encoded_specific_only"]
    )

    df["flag_nonencoded_specific_retained"] = df["residual_specific_class"].astype(str).isin(
        ["both_specific_retained", "nonencoded_specific_only"]
    )

    if "encoded_specific_retained_nominal" in df.columns:
        df["flag_encoded_specific_retained"] = (
            df["flag_encoded_specific_retained"]
            | df["encoded_specific_retained_nominal"].apply(bool_from_value)
        )

    if "nonencoded_specific_retained_nominal" in df.columns:
        df["flag_nonencoded_specific_retained"] = (
            df["flag_nonencoded_specific_retained"]
            | df["nonencoded_specific_retained_nominal"].apply(bool_from_value)
        )

    df["flag_both_specific_retained"] = (
        df["flag_encoded_specific_retained"]
        & df["flag_nonencoded_specific_retained"]
    )

    return df


def classify_final(row):
    priority = str(row.get("encoded_vs_nonencoded_priority_class", ""))
    encoded_specific = bool(row.get("flag_encoded_specific_retained", False))
    nonencoded_specific = bool(row.get("flag_nonencoded_specific_retained", False))
    low_corr = bool(row.get("flag_low_correlation", False))
    high_mod_corr = bool(row.get("flag_high_or_moderate_correlation", False))

    if is_encoded_priority(priority):
        if encoded_specific and nonencoded_specific:
            return "encoded_leading_with_parallel_network"
        if encoded_specific and not nonencoded_specific:
            return "encoded_dominant"
        if not encoded_specific and nonencoded_specific:
            return "network_signal_despite_encoded_priority"
        return "encoded_leading_without_residual_specificity"

    if is_nonencoded_stronger_top10(priority):
        if low_corr and encoded_specific and nonencoded_specific:
            return "network_extended_with_encoded_support"
        if low_corr and nonencoded_specific and not encoded_specific:
            return "network_extended_nonencoded_specific"
        if low_corr and encoded_specific and not nonencoded_specific:
            return "encoded_specific_despite_nonencoded_stronger"
        if high_mod_corr:
            return "shared_module_nonencoded_stronger"
        return "nonencoded_stronger_top10_unresolved"

    if is_nonencoded_leading(priority):
        if low_corr and encoded_specific and nonencoded_specific:
            return "nonencoded_dominant_with_encoded_residual_support"
        if low_corr and nonencoded_specific and not encoded_specific:
            return "nonencoded_dominant_network_extended"
        if low_corr and encoded_specific and not nonencoded_specific:
            return "encoded_residual_support_but_rank_weak"
        if high_mod_corr:
            return "shared_module_nonencoded_leading"
        return "nonencoded_leading_unresolved"

    return "unclassified"


def coarse_class(final_class):
    final_class = str(final_class)

    if final_class in {
        "encoded_dominant",
        "encoded_leading_with_parallel_network",
        "encoded_leading_without_residual_specificity",
    }:
        return "encoded_prioritized"

    if final_class in {
        "network_signal_despite_encoded_priority",
        "network_extended_with_encoded_support",
        "network_extended_nonencoded_specific",
        "encoded_specific_despite_nonencoded_stronger",
        "nonencoded_dominant_with_encoded_residual_support",
        "nonencoded_dominant_network_extended",
        "encoded_residual_support_but_rank_weak",
    }:
        return "network_extended"

    if final_class in {
        "shared_module_nonencoded_stronger",
        "shared_module_nonencoded_leading",
    }:
        return "shared_module"

    return "unresolved"


def summarize_overall(df):
    n = df.shape[0]

    rows = [
        ("n_pairs", n),
        ("n_encoded_priority", int(df["flag_encoded_priority"].sum())),
        ("prop_encoded_priority", df["flag_encoded_priority"].mean()),
        ("n_nonencoded_stronger_top10", int(df["flag_nonencoded_stronger_top10"].sum())),
        ("prop_nonencoded_stronger_top10", df["flag_nonencoded_stronger_top10"].mean()),
        ("n_nonencoded_leading", int(df["flag_nonencoded_leading"].sum())),
        ("prop_nonencoded_leading", df["flag_nonencoded_leading"].mean()),
        ("n_encoded_specific_retained", int(df["flag_encoded_specific_retained"].sum())),
        ("prop_encoded_specific_retained", df["flag_encoded_specific_retained"].mean()),
        ("n_nonencoded_specific_retained", int(df["flag_nonencoded_specific_retained"].sum())),
        ("prop_nonencoded_specific_retained", df["flag_nonencoded_specific_retained"].mean()),
        ("n_both_specific_retained", int(df["flag_both_specific_retained"].sum())),
        ("prop_both_specific_retained", df["flag_both_specific_retained"].mean()),
    ]

    if "encoded_abs_indirect_logHR" in df.columns:
        rows.append(
            (
                "median_encoded_abs_indirect_logHR",
                df["encoded_abs_indirect_logHR"].median(),
            )
        )

    if "best_nonencoded_abs_indirect_logHR" in df.columns:
        rows.append(
            (
                "median_best_nonencoded_abs_indirect_logHR",
                df["best_nonencoded_abs_indirect_logHR"].median(),
            )
        )

    if "encoded_specific_abs_indirect_logHR" in df.columns:
        rows.append(
            (
                "median_encoded_specific_abs_indirect_logHR",
                df["encoded_specific_abs_indirect_logHR"].median(),
            )
        )

    if "nonencoded_specific_abs_indirect_logHR" in df.columns:
        rows.append(
            (
                "median_nonencoded_specific_abs_indirect_logHR",
                df["nonencoded_specific_abs_indirect_logHR"].median(),
            )
        )

    for cls, count in df["final_pathway_class"].value_counts().items():
        rows.append((f"final_pathway_class::{cls}::n", int(count)))
        rows.append((f"final_pathway_class::{cls}::proportion", count / n))

    for cls, count in df["final_coarse_class"].value_counts().items():
        rows.append((f"final_coarse_class::{cls}::n", int(count)))
        rows.append((f"final_coarse_class::{cls}::proportion", count / n))

    return pd.DataFrame(rows, columns=["metric", "value"])


def summarize_by_tier(df):
    if "tier_label" not in df.columns:
        return pd.DataFrame()

    rows = []

    for tier, g in df.groupby("tier_label", dropna=False):
        n = g.shape[0]

        rows.append(
            {
                "tier_label": tier,
                "n_pairs": n,
                "n_encoded_priority": int(g["flag_encoded_priority"].sum()),
                "prop_encoded_priority": g["flag_encoded_priority"].mean(),
                "n_encoded_specific_retained": int(g["flag_encoded_specific_retained"].sum()),
                "prop_encoded_specific_retained": g["flag_encoded_specific_retained"].mean(),
                "n_nonencoded_specific_retained": int(g["flag_nonencoded_specific_retained"].sum()),
                "prop_nonencoded_specific_retained": g["flag_nonencoded_specific_retained"].mean(),
                "n_encoded_prioritized_coarse": int((g["final_coarse_class"] == "encoded_prioritized").sum()),
                "prop_encoded_prioritized_coarse": (g["final_coarse_class"] == "encoded_prioritized").mean(),
                "n_network_extended_coarse": int((g["final_coarse_class"] == "network_extended").sum()),
                "prop_network_extended_coarse": (g["final_coarse_class"] == "network_extended").mean(),
                "n_unresolved_coarse": int((g["final_coarse_class"] == "unresolved").sum()),
                "prop_unresolved_coarse": (g["final_coarse_class"] == "unresolved").mean(),
            }
        )

    return pd.DataFrame(rows)


def build_figure_class_summary(df, colname):
    out = df[colname].value_counts().reset_index()
    out.columns = [colname, "n_pairs"]
    out["proportion"] = out["n_pairs"] / out["n_pairs"].sum()
    return out


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    step4 = prepare_step4(args.step4_comparison)
    step5 = prepare_step5(args.step5_correlation)
    step6 = prepare_step6(args.step6_conditional)
    step7 = prepare_step7(args.step7_residual_pair)

    df = (
        step4
        .merge(step5, on="pair_id", how="left")
        .merge(step6, on="pair_id", how="left")
        .merge(step7, on="pair_id", how="left")
    )

    df = convert_numeric(df)
    df = add_flags(df)

    df["final_pathway_class"] = df.apply(classify_final, axis=1)
    df["final_coarse_class"] = df["final_pathway_class"].apply(coarse_class)

    overall = summarize_overall(df)
    by_tier = summarize_by_tier(df)

    fig_fine = build_figure_class_summary(df, "final_pathway_class")
    fig_coarse = build_figure_class_summary(df, "final_coarse_class")

    out_class = outdir / f"{args.prefix}_final_pathway_classification.tsv"
    out_overall = outdir / f"{args.prefix}_final_pathway_classification_overall_summary.tsv"
    out_tier = outdir / f"{args.prefix}_final_pathway_classification_by_tier.tsv"
    out_fig_fine = outdir / f"{args.prefix}_figure_final_pathway_class_summary.tsv"
    out_fig_coarse = outdir / f"{args.prefix}_figure_final_coarse_class_summary.tsv"

    df.to_csv(out_class, sep="\t", index=False)
    overall.to_csv(out_overall, sep="\t", index=False)
    by_tier.to_csv(out_tier, sep="\t", index=False)
    fig_fine.to_csv(out_fig_fine, sep="\t", index=False)
    fig_coarse.to_csv(out_fig_coarse, sep="\t", index=False)

    print("[Done] Step8 final pathway classification completed.")
    print(f"[Output] {out_class}")
    print(f"[Output] {out_overall}")
    print(f"[Output] {out_tier}")
    print(f"[Output] {out_fig_fine}")
    print(f"[Output] {out_fig_coarse}")

    print("[QC]")
    for _, r in overall.iterrows():
        print(f"{r['metric']}\t{r['value']}")


if __name__ == "__main__":
    main()
