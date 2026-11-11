#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_aggregate_rank_all_protein_mediators.py

Purpose
-------
Aggregate all Step2 chunk outputs and rank encoded proteins against all measured
proteins within each gene-disease pair.

Main question
-------------
For each supported gene-disease pair:
    Does the encoded protein rank among the leading single-protein mediators?

Primary ranking
---------------
rank_by_indirect_p:
    smaller indirect_p_analytic = stronger single-mediator evidence

Secondary rankings
------------------
rank_by_abs_indirect_logHR:
    larger abs(indirect_logHR) = stronger effect size

rank_by_abs_PM:
    larger abs(proportion_mediated_logscale) = larger mediation proportion

Inputs
------
1. 02_all_protein_mediation_scan.chunk*.tsv.gz
2. 01_supported_gene_disease_pairs.tsv

Outputs
-------
1. 03_all_protein_mediation_scan.all.tsv.gz
2. 03_encoded_protein_ranking_summary.tsv
3. 03_encoded_protein_ranking_overall_summary.tsv
4. 03_figure_encoded_rank_distribution.tsv
5. 03_figure_encoded_top_category_summary.tsv
6. 03_figure_encoded_vs_best_nonencoded.tsv
7. 03_figure_case_<gene>_<icd3>_all_protein_ranking.tsv
"""

import argparse
from pathlib import Path
import glob

import numpy as np
import pandas as pd


NUMERIC_COLS = [
    "is_encoded_protein",
    "n_total_rows",
    "n_complete_case",
    "n_events_complete_case",
    "n_carriers_complete_case",
    "n_noncarriers_complete_case",
    "n_carrier_events_complete_case",
    "n_carrier_nonevents_complete_case",
    "M_sd_complete_case",
    "mediator_model_success",
    "total_model_success",
    "direct_model_success",
    "model_success",
    "beta_EM",
    "se_EM",
    "p_EM",
    "beta_EY_total",
    "se_EY_total",
    "p_EY_total",
    "hr_EY_total",
    "beta_EY_direct",
    "se_EY_direct",
    "p_EY_direct",
    "hr_EY_direct",
    "beta_MY_conditional",
    "se_MY_conditional",
    "p_MY_conditional",
    "hr_MY_conditional",
    "indirect_logHR",
    "indirect_hr",
    "se_indirect_logHR_analytic",
    "z_indirect_analytic",
    "indirect_p_analytic",
    "proportion_mediated_logscale",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate all-protein mediation scan chunks and rank encoded proteins."
    )

    parser.add_argument(
        "--scan_dir",
        required=True,
        help="Directory containing 02_all_protein_mediation_scan.chunk*.tsv.gz.",
    )

    parser.add_argument(
        "--chunk_pattern",
        default="02_all_protein_mediation_scan.chunk*.tsv.gz",
        help="Chunk file pattern. Default: 02_all_protein_mediation_scan.chunk*.tsv.gz",
    )

    parser.add_argument(
        "--supported_pairs",
        required=True,
        help="Path to 01_supported_gene_disease_pairs.tsv.",
    )

    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "--prefix",
        default="03",
        help="Output prefix. Default: 03.",
    )

    parser.add_argument(
        "--case_gene",
        default="EGFR",
        help="Representative case gene. Default: EGFR.",
    )

    parser.add_argument(
        "--case_icd3",
        default="N17",
        help="Representative case ICD-3. Default: N17.",
    )

    return parser.parse_args()


def bh_fdr(pvals: pd.Series) -> pd.Series:
    """
    Benjamini-Hochberg FDR correction.

    Returns q values aligned to the input index.
    """
    p = pd.to_numeric(pvals, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype="float64")

    valid = p.notna() & np.isfinite(p) & (p >= 0) & (p <= 1)
    if valid.sum() == 0:
        return out

    pv = p.loc[valid].astype(float)
    order = np.argsort(pv.values)
    ranked = pv.values[order]
    n = len(ranked)

    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    valid_index = pv.index[order]
    out.loc[valid_index] = q

    return out


def read_chunks(scan_dir: str, chunk_pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(str(Path(scan_dir) / chunk_pattern)))

    if not paths:
        raise FileNotFoundError(
            f"No chunk files found under {scan_dir} with pattern {chunk_pattern}"
        )

    dfs = []
    for p in paths:
        df = pd.read_csv(p, sep="\t", compression="infer", dtype=str)
        df["source_chunk_file"] = Path(p).name
        dfs.append(df)

    out = pd.concat(dfs, axis=0, ignore_index=True)

    return out


def convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_validity_and_qvalues(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_encoded_protein"] = pd.to_numeric(
        df["is_encoded_protein"], errors="coerce"
    ).fillna(0).astype(int)

    df["abs_indirect_logHR"] = np.abs(df["indirect_logHR"])
    df["abs_proportion_mediated_logscale"] = np.abs(
        df["proportion_mediated_logscale"]
    )

    df["valid_for_ranking"] = (
        df["model_success"].eq(1)
        & df["indirect_p_analytic"].notna()
        & np.isfinite(df["indirect_p_analytic"])
        & df["indirect_logHR"].notna()
        & np.isfinite(df["indirect_logHR"])
        & df["proportion_mediated_logscale"].notna()
        & np.isfinite(df["proportion_mediated_logscale"])
    ).astype(int)

    df["indirect_q_analytic_global"] = np.nan
    df.loc[df["valid_for_ranking"].eq(1), "indirect_q_analytic_global"] = bh_fdr(
        df.loc[df["valid_for_ranking"].eq(1), "indirect_p_analytic"]
    )

    df["indirect_q_analytic_within_pair"] = np.nan

    for pair_id, sub_idx in df.groupby("pair_id").groups.items():
        idx = pd.Index(sub_idx)
        valid_idx = idx[df.loc[idx, "valid_for_ranking"].eq(1)]
        if len(valid_idx) > 0:
            df.loc[valid_idx, "indirect_q_analytic_within_pair"] = bh_fdr(
                df.loc[valid_idx, "indirect_p_analytic"]
            )

    return df


def add_pairwise_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rank_cols = [
        "rank_by_indirect_p",
        "rank_by_abs_indirect_logHR",
        "rank_by_abs_PM",
        "percentile_by_indirect_p",
        "percentile_by_abs_indirect_logHR",
        "percentile_by_abs_PM",
        "n_tested_proteins_valid",
        "n_tested_proteins_total",
    ]

    for col in rank_cols:
        df[col] = np.nan

    for pair_id, idx_raw in df.groupby("pair_id").groups.items():
        idx = pd.Index(idx_raw)
        idx_valid = idx[df.loc[idx, "valid_for_ranking"].eq(1)]

        n_total = len(idx)
        n_valid = len(idx_valid)

        df.loc[idx, "n_tested_proteins_total"] = n_total
        df.loc[idx, "n_tested_proteins_valid"] = n_valid

        if n_valid == 0:
            continue

        sub = df.loc[idx_valid].copy()

        rank_p = sub["indirect_p_analytic"].rank(
            method="min",
            ascending=True,
        )

        rank_abs_loghr = sub["abs_indirect_logHR"].rank(
            method="min",
            ascending=False,
        )

        rank_abs_pm = sub["abs_proportion_mediated_logscale"].rank(
            method="min",
            ascending=False,
        )

        df.loc[idx_valid, "rank_by_indirect_p"] = rank_p
        df.loc[idx_valid, "rank_by_abs_indirect_logHR"] = rank_abs_loghr
        df.loc[idx_valid, "rank_by_abs_PM"] = rank_abs_pm

        df.loc[idx_valid, "percentile_by_indirect_p"] = rank_p / n_valid
        df.loc[idx_valid, "percentile_by_abs_indirect_logHR"] = rank_abs_loghr / n_valid
        df.loc[idx_valid, "percentile_by_abs_PM"] = rank_abs_pm / n_valid

    return df


def get_best_nonencoded(df: pd.DataFrame) -> pd.DataFrame:
    valid_nonencoded = df.loc[
        df["valid_for_ranking"].eq(1) & df["is_encoded_protein"].eq(0)
    ].copy()

    if valid_nonencoded.empty:
        return pd.DataFrame()

    valid_nonencoded = valid_nonencoded.sort_values(
        ["pair_id", "rank_by_indirect_p", "rank_by_abs_indirect_logHR"],
        ascending=[True, True, True],
    )

    best = valid_nonencoded.groupby("pair_id", as_index=False).head(1).copy()

    keep = [
        "pair_id",
        "mediator_protein",
        "indirect_logHR",
        "indirect_hr",
        "abs_indirect_logHR",
        "proportion_mediated_logscale",
        "abs_proportion_mediated_logscale",
        "indirect_p_analytic",
        "indirect_q_analytic_global",
        "indirect_q_analytic_within_pair",
        "rank_by_indirect_p",
        "rank_by_abs_indirect_logHR",
        "rank_by_abs_PM",
        "percentile_by_indirect_p",
        "percentile_by_abs_indirect_logHR",
        "percentile_by_abs_PM",
        "model_success",
        "n_complete_case",
        "n_events_complete_case",
        "n_carriers_complete_case",
        "n_carrier_events_complete_case",
    ]

    best = best.loc[:, [c for c in keep if c in best.columns]].copy()

    rename = {
        c: f"best_nonencoded_{c}"
        for c in best.columns
        if c != "pair_id"
    }

    best = best.rename(columns=rename)

    return best


def build_encoded_summary(df: pd.DataFrame, supported_pairs: pd.DataFrame) -> pd.DataFrame:
    encoded = df.loc[df["is_encoded_protein"].eq(1)].copy()

    if encoded.empty:
        raise ValueError("No encoded protein rows found in aggregated Step2 results.")

    keep = [
        "pair_id",
        "gene",
        "icd3",
        "encoded_protein",
        "mediator_protein",
        "tier_label",
        "original_trio_id",
        "valid_for_ranking",
        "model_success",
        "fail_reason",
        "n_tested_proteins_total",
        "n_tested_proteins_valid",
        "n_complete_case",
        "n_events_complete_case",
        "n_carriers_complete_case",
        "n_carrier_events_complete_case",
        "indirect_logHR",
        "indirect_hr",
        "abs_indirect_logHR",
        "proportion_mediated_logscale",
        "abs_proportion_mediated_logscale",
        "indirect_p_analytic",
        "indirect_q_analytic_global",
        "indirect_q_analytic_within_pair",
        "rank_by_indirect_p",
        "rank_by_abs_indirect_logHR",
        "rank_by_abs_PM",
        "percentile_by_indirect_p",
        "percentile_by_abs_indirect_logHR",
        "percentile_by_abs_PM",
    ]

    encoded = encoded.loc[:, [c for c in keep if c in encoded.columns]].copy()

    rename = {
        "mediator_protein": "encoded_mediator_protein",
        "valid_for_ranking": "encoded_valid_for_ranking",
        "model_success": "encoded_model_success",
        "fail_reason": "encoded_fail_reason",
        "n_complete_case": "encoded_n_complete_case",
        "n_events_complete_case": "encoded_n_events_complete_case",
        "n_carriers_complete_case": "encoded_n_carriers_complete_case",
        "n_carrier_events_complete_case": "encoded_n_carrier_events_complete_case",
        "indirect_logHR": "encoded_indirect_logHR",
        "indirect_hr": "encoded_indirect_hr",
        "abs_indirect_logHR": "encoded_abs_indirect_logHR",
        "proportion_mediated_logscale": "encoded_PM",
        "abs_proportion_mediated_logscale": "encoded_abs_PM",
        "indirect_p_analytic": "encoded_indirect_p_analytic",
        "indirect_q_analytic_global": "encoded_indirect_q_analytic_global",
        "indirect_q_analytic_within_pair": "encoded_indirect_q_analytic_within_pair",
        "rank_by_indirect_p": "encoded_rank_by_indirect_p",
        "rank_by_abs_indirect_logHR": "encoded_rank_by_abs_indirect_logHR",
        "rank_by_abs_PM": "encoded_rank_by_abs_PM",
        "percentile_by_indirect_p": "encoded_percentile_by_indirect_p",
        "percentile_by_abs_indirect_logHR": "encoded_percentile_by_abs_indirect_logHR",
        "percentile_by_abs_PM": "encoded_percentile_by_abs_PM",
    }

    encoded = encoded.rename(columns=rename)

    best_nonencoded = get_best_nonencoded(df)

    summary = encoded.merge(best_nonencoded, on="pair_id", how="left")

    # Add missing supported pairs if any encoded row was not generated.
    base_cols = [
        "pair_id",
        "gene",
        "icd3",
        "encoded_protein",
        "tier_label",
        "original_trio_id",
    ]

    supported_base = supported_pairs.loc[
        :, [c for c in base_cols if c in supported_pairs.columns]
    ].drop_duplicates("pair_id")

    summary = supported_base.merge(
        summary,
        on=[c for c in base_cols if c in summary.columns and c in supported_base.columns],
        how="left",
    )

    summary["encoded_is_top1_by_indirect_p"] = (
        summary["encoded_rank_by_indirect_p"].eq(1)
    ).astype("Int64")

    summary["encoded_is_top5_by_indirect_p"] = (
        summary["encoded_rank_by_indirect_p"].le(5)
    ).astype("Int64")

    summary["encoded_is_top10_by_indirect_p"] = (
        summary["encoded_rank_by_indirect_p"].le(10)
    ).astype("Int64")

    summary["encoded_is_top1_by_abs_logHR"] = (
        summary["encoded_rank_by_abs_indirect_logHR"].eq(1)
    ).astype("Int64")

    summary["encoded_is_top5_by_abs_logHR"] = (
        summary["encoded_rank_by_abs_indirect_logHR"].le(5)
    ).astype("Int64")

    summary["encoded_is_top10_by_abs_logHR"] = (
        summary["encoded_rank_by_abs_indirect_logHR"].le(10)
    ).astype("Int64")

    summary["encoded_stronger_than_best_nonencoded_by_abs_logHR"] = (
        summary["encoded_abs_indirect_logHR"]
        >= summary["best_nonencoded_abs_indirect_logHR"]
    ).astype("Int64")

    summary["encoded_stronger_than_best_nonencoded_by_abs_PM"] = (
        summary["encoded_abs_PM"]
        >= summary["best_nonencoded_abs_proportion_mediated_logscale"]
    ).astype("Int64")

    return summary


def build_overall_summary(encoded_summary: pd.DataFrame, df_all: pd.DataFrame) -> pd.DataFrame:
    n_pairs = encoded_summary["pair_id"].nunique()
    encoded_valid = encoded_summary["encoded_valid_for_ranking"].eq(1)

    def count_true(col):
        if col not in encoded_summary.columns:
            return np.nan
        return int(encoded_summary.loc[encoded_valid, col].fillna(0).astype(int).sum())

    def prop_true(col):
        if encoded_valid.sum() == 0:
            return np.nan
        return count_true(col) / int(encoded_valid.sum())

    rows = [
        ("n_gene_disease_pairs", n_pairs),
        ("n_all_rows", int(df_all.shape[0])),
        ("n_valid_rows_for_ranking", int(df_all["valid_for_ranking"].eq(1).sum())),
        ("n_encoded_rows", int(encoded_summary.shape[0])),
        ("n_encoded_valid_for_ranking", int(encoded_valid.sum())),
        ("n_encoded_failed_or_invalid", int((~encoded_valid).sum())),

        ("n_encoded_top1_by_indirect_p", count_true("encoded_is_top1_by_indirect_p")),
        ("n_encoded_top5_by_indirect_p", count_true("encoded_is_top5_by_indirect_p")),
        ("n_encoded_top10_by_indirect_p", count_true("encoded_is_top10_by_indirect_p")),

        ("prop_encoded_top1_by_indirect_p", prop_true("encoded_is_top1_by_indirect_p")),
        ("prop_encoded_top5_by_indirect_p", prop_true("encoded_is_top5_by_indirect_p")),
        ("prop_encoded_top10_by_indirect_p", prop_true("encoded_is_top10_by_indirect_p")),

        ("n_encoded_top1_by_abs_logHR", count_true("encoded_is_top1_by_abs_logHR")),
        ("n_encoded_top5_by_abs_logHR", count_true("encoded_is_top5_by_abs_logHR")),
        ("n_encoded_top10_by_abs_logHR", count_true("encoded_is_top10_by_abs_logHR")),

        ("prop_encoded_top1_by_abs_logHR", prop_true("encoded_is_top1_by_abs_logHR")),
        ("prop_encoded_top5_by_abs_logHR", prop_true("encoded_is_top5_by_abs_logHR")),
        ("prop_encoded_top10_by_abs_logHR", prop_true("encoded_is_top10_by_abs_logHR")),

        (
            "median_encoded_rank_by_indirect_p",
            encoded_summary.loc[encoded_valid, "encoded_rank_by_indirect_p"].median(),
        ),
        (
            "median_encoded_percentile_by_indirect_p",
            encoded_summary.loc[encoded_valid, "encoded_percentile_by_indirect_p"].median(),
        ),
        (
            "median_encoded_abs_indirect_logHR",
            encoded_summary.loc[encoded_valid, "encoded_abs_indirect_logHR"].median(),
        ),
        (
            "median_encoded_abs_PM",
            encoded_summary.loc[encoded_valid, "encoded_abs_PM"].median(),
        ),
        (
            "n_encoded_stronger_than_best_nonencoded_by_abs_logHR",
            count_true("encoded_stronger_than_best_nonencoded_by_abs_logHR"),
        ),
        (
            "prop_encoded_stronger_than_best_nonencoded_by_abs_logHR",
            prop_true("encoded_stronger_than_best_nonencoded_by_abs_logHR"),
        ),
        (
            "n_encoded_stronger_than_best_nonencoded_by_abs_PM",
            count_true("encoded_stronger_than_best_nonencoded_by_abs_PM"),
        ),
        (
            "prop_encoded_stronger_than_best_nonencoded_by_abs_PM",
            prop_true("encoded_stronger_than_best_nonencoded_by_abs_PM"),
        ),
    ]

    return pd.DataFrame(rows, columns=["metric", "value"])


def build_figure_inputs(encoded_summary: pd.DataFrame, df_all: pd.DataFrame, case_gene: str, case_icd3: str):
    fig_rank = encoded_summary.loc[
        :,
        [
            "pair_id",
            "gene",
            "icd3",
            "encoded_protein",
            "tier_label",
            "encoded_valid_for_ranking",
            "n_tested_proteins_valid",
            "encoded_rank_by_indirect_p",
            "encoded_percentile_by_indirect_p",
            "encoded_rank_by_abs_indirect_logHR",
            "encoded_percentile_by_abs_indirect_logHR",
            "encoded_indirect_hr",
            "encoded_PM",
        ],
    ].copy()

    categories = [
        ("top1", "encoded_is_top1_by_indirect_p"),
        ("top5", "encoded_is_top5_by_indirect_p"),
        ("top10", "encoded_is_top10_by_indirect_p"),
    ]

    rows = []
    valid = encoded_summary["encoded_valid_for_ranking"].eq(1)

    for label, col in categories:
        n = int(encoded_summary.loc[valid, col].fillna(0).astype(int).sum())
        denom = int(valid.sum())
        rows.append(
            {
                "category": label,
                "n_pairs": n,
                "n_valid_encoded_pairs": denom,
                "proportion": n / denom if denom > 0 else np.nan,
            }
        )

    fig_top = pd.DataFrame(rows)

    fig_compare = encoded_summary.loc[
        :,
        [
            "pair_id",
            "gene",
            "icd3",
            "encoded_protein",
            "encoded_valid_for_ranking",
            "encoded_abs_indirect_logHR",
            "encoded_abs_PM",
            "encoded_indirect_hr",
            "encoded_PM",
            "best_nonencoded_mediator_protein",
            "best_nonencoded_abs_indirect_logHR",
            "best_nonencoded_abs_proportion_mediated_logscale",
            "best_nonencoded_indirect_hr",
            "best_nonencoded_proportion_mediated_logscale",
            "encoded_stronger_than_best_nonencoded_by_abs_logHR",
            "encoded_stronger_than_best_nonencoded_by_abs_PM",
        ],
    ].copy()

    case_df = df_all.loc[
        (df_all["gene"].astype(str) == str(case_gene))
        & (df_all["icd3"].astype(str) == str(case_icd3))
        & df_all["valid_for_ranking"].eq(1)
    ].copy()

    if not case_df.empty:
        case_df = case_df.sort_values(
            ["rank_by_indirect_p", "rank_by_abs_indirect_logHR"],
            ascending=[True, True],
        )

    return fig_rank, fig_top, fig_compare, case_df


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_all = outdir / f"{args.prefix}_all_protein_mediation_scan.all.tsv.gz"
    out_encoded = outdir / f"{args.prefix}_encoded_protein_ranking_summary.tsv"
    out_overall = outdir / f"{args.prefix}_encoded_protein_ranking_overall_summary.tsv"

    out_fig_rank = outdir / f"{args.prefix}_figure_encoded_rank_distribution.tsv"
    out_fig_top = outdir / f"{args.prefix}_figure_encoded_top_category_summary.tsv"
    out_fig_compare = outdir / f"{args.prefix}_figure_encoded_vs_best_nonencoded.tsv"
    out_fig_case = outdir / (
        f"{args.prefix}_figure_case_{args.case_gene}_{args.case_icd3}_all_protein_ranking.tsv"
    )

    supported_pairs = pd.read_csv(args.supported_pairs, sep="\t", dtype=str)

    df = read_chunks(args.scan_dir, args.chunk_pattern)
    df = convert_numeric(df)

    required_cols = [
        "pair_id",
        "gene",
        "icd3",
        "encoded_protein",
        "mediator_protein",
        "is_encoded_protein",
        "model_success",
        "indirect_p_analytic",
        "indirect_logHR",
        "proportion_mediated_logscale",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns in Step2 outputs: " + ", ".join(missing))

    df = add_validity_and_qvalues(df)
    df = add_pairwise_ranks(df)

    encoded_summary = build_encoded_summary(df, supported_pairs)
    overall_summary = build_overall_summary(encoded_summary, df)

    fig_rank, fig_top, fig_compare, fig_case = build_figure_inputs(
        encoded_summary=encoded_summary,
        df_all=df,
        case_gene=args.case_gene,
        case_icd3=args.case_icd3,
    )

    df.to_csv(out_all, sep="\t", index=False, compression="gzip")
    encoded_summary.to_csv(out_encoded, sep="\t", index=False)
    overall_summary.to_csv(out_overall, sep="\t", index=False)

    fig_rank.to_csv(out_fig_rank, sep="\t", index=False)
    fig_top.to_csv(out_fig_top, sep="\t", index=False)
    fig_compare.to_csv(out_fig_compare, sep="\t", index=False)
    fig_case.to_csv(out_fig_case, sep="\t", index=False)

    print("[Done] Step3 aggregation and ranking completed.")
    print(f"[Output] {out_all}")
    print(f"[Output] {out_encoded}")
    print(f"[Output] {out_overall}")
    print(f"[Output] {out_fig_rank}")
    print(f"[Output] {out_fig_top}")
    print(f"[Output] {out_fig_compare}")
    print(f"[Output] {out_fig_case}")

    print("[QC]")
    for _, r in overall_summary.iterrows():
        print(f"{r['metric']}\t{r['value']}")


if __name__ == "__main__":
    main()
