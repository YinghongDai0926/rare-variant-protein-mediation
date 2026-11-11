#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd


def load_table(path):
    return pd.read_csv(path, sep="\t", compression="infer")


def choose_columns(df):
    preferred = [
        "rank_overall_significant",
        "rank_within_tier",
        "trio_id", "gene", "protein", "icd3", "tier_label",
        "indirect_logHR", "indirect_hr",
        "ci95_l_indirect_logHR", "ci95_u_indirect_logHR",
        "ci95_l_indirect_hr", "ci95_u_indirect_hr",
        "indirect_p_boot", "indirect_q_boot",
        "proportion_mediated_logscale",
        "beta_EY_total", "hr_EY_total",
        "beta_EY_direct", "hr_EY_direct",
        "beta_EM_individual",
        "beta_MY_conditional", "hr_MY_conditional",
        "boot_success_rate", "n_boot_success",
        "n_complete_case", "n_events_complete_case",
        "n_carriers_complete_case", "n_carrier_events_complete_case",
        "model_success", "table_path"
    ]
    cols = [c for c in preferred if c in df.columns]
    return df[cols].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Extract main-text and supplementary result tables from prioritized mediation hits"
    )
    parser.add_argument("--ranked_all_tsv", required=True, help="03_mediation_ranked_all.tsv.gz")
    parser.add_argument("--ranked_sig_tsv", required=True, help="03_mediation_ranked_significant.tsv.gz")
    parser.add_argument("--ranked_tier1_tsv", required=True, help="03_mediation_ranked_Tier1.tsv.gz")
    parser.add_argument("--ranked_tier2_tsv", required=True, help="03_mediation_ranked_Tier2.tsv.gz")
    parser.add_argument("--mediation_results_tsv", required=True, help="02_mediation_results.all.tsv.gz")
    parser.add_argument("--outdir", required=True, help="Output directory")

    parser.add_argument("--top_n_overall", type=int, default=20)
    parser.add_argument("--top_n_tier2", type=int, default=20)
    parser.add_argument("--top_n_tier1", type=int, default=8)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    out_main_overall = os.path.join(args.outdir, "04_main_text_top_overall.tsv")
    out_main_t1 = os.path.join(args.outdir, "04_main_text_tier1.tsv")
    out_main_t2 = os.path.join(args.outdir, "04_main_text_top_tier2.tsv")

    out_supp_prioritized = os.path.join(args.outdir, "04_supplementary_prioritized_all.tsv.gz")
    out_supp_model_success = os.path.join(args.outdir, "04_supplementary_model_success_all.tsv.gz")
    out_supp_model_failed = os.path.join(args.outdir, "04_supplementary_model_failed.tsv.gz")

    out_qc = os.path.join(args.outdir, "04_extract_tables.qc.tsv")
    out_summary = os.path.join(args.outdir, "04_extract_tables.summary.txt")

    ranked_all = load_table(args.ranked_all_tsv)
    ranked_sig = load_table(args.ranked_sig_tsv)
    ranked_t1 = load_table(args.ranked_tier1_tsv)
    ranked_t2 = load_table(args.ranked_tier2_tsv)
    mediation_all = load_table(args.mediation_results_tsv)

    # main-text tables
    top_overall = ranked_sig.head(args.top_n_overall).copy()
    top_t1 = ranked_t1.head(args.top_n_tier1).copy()
    top_t2 = ranked_t2.head(args.top_n_tier2).copy()

    choose_columns(top_overall).to_csv(out_main_overall, sep="\t", index=False)
    choose_columns(top_t1).to_csv(out_main_t1, sep="\t", index=False)
    choose_columns(top_t2).to_csv(out_main_t2, sep="\t", index=False)

    # supplementary tables
    choose_columns(ranked_sig).to_csv(out_supp_prioritized, sep="\t", index=False, compression="gzip")

    model_success_all = mediation_all[pd.to_numeric(mediation_all.get("model_success", 0), errors="coerce") == 1].copy()
    model_failed_all = mediation_all[pd.to_numeric(mediation_all.get("model_success", 0), errors="coerce") != 1].copy()

    choose_columns(model_success_all).to_csv(out_supp_model_success, sep="\t", index=False, compression="gzip")
    mediation_all[pd.to_numeric(mediation_all.get("model_success", 0), errors="coerce") != 1].to_csv(
        out_supp_model_failed, sep="\t", index=False, compression="gzip"
    )

    # QC
    qc_rows = [
        {"metric": "n_ranked_all", "value": ranked_all.shape[0]},
        {"metric": "n_ranked_significant", "value": ranked_sig.shape[0]},
        {"metric": "n_ranked_tier1", "value": ranked_t1.shape[0]},
        {"metric": "n_ranked_tier2", "value": ranked_t2.shape[0]},
        {"metric": "n_model_success_all", "value": model_success_all.shape[0]},
        {"metric": "n_model_failed_all", "value": model_failed_all.shape[0]},
        {"metric": "n_main_text_top_overall", "value": top_overall.shape[0]},
        {"metric": "n_main_text_tier1", "value": top_t1.shape[0]},
        {"metric": "n_main_text_top_tier2", "value": top_t2.shape[0]},
    ]

    if not ranked_sig.empty:
        qc_rows.extend([
            {"metric": "top1_gene", "value": str(ranked_sig.iloc[0]["gene"]) if "gene" in ranked_sig.columns else ""},
            {"metric": "top1_protein", "value": str(ranked_sig.iloc[0]["protein"]) if "protein" in ranked_sig.columns else ""},
            {"metric": "top1_icd3", "value": str(ranked_sig.iloc[0]["icd3"]) if "icd3" in ranked_sig.columns else ""},
            {"metric": "top1_indirect_logHR", "value": float(ranked_sig.iloc[0]["indirect_logHR"]) if "indirect_logHR" in ranked_sig.columns else np.nan},
        ])

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(out_qc, sep="\t", index=False)

    # summary
    lines = []
    lines.append("Main-text and supplementary table extraction summary")
    lines.append("===================================================")
    lines.append("")
    lines.append(f"Prioritized significant trios      : {ranked_sig.shape[0]}")
    lines.append(f"Tier1 prioritized trios            : {ranked_t1.shape[0]}")
    lines.append(f"Tier2 prioritized trios            : {ranked_t2.shape[0]}")
    lines.append(f"All successful mediation models    : {model_success_all.shape[0]}")
    lines.append(f"All failed mediation models        : {model_failed_all.shape[0]}")
    lines.append("")
    lines.append(f"Main-text top overall              : {top_overall.shape[0]}")
    lines.append(f"Main-text Tier1 table              : {top_t1.shape[0]}")
    lines.append(f"Main-text Tier2 top table          : {top_t2.shape[0]}")
    lines.append("")
    lines.append("Interpretation")
    lines.append("--------------")
    lines.append("Main-text tables are ordered by indirect_logHR in descending order among stable and significant trios.")
    lines.append("Supplementary tables retain the full prioritized set and all successful/failed mediation results.")

    with open(out_summary, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[OK] main-text overall table written to: {out_main_overall}")
    print(f"[OK] main-text Tier1 table written to: {out_main_t1}")
    print(f"[OK] main-text Tier2 table written to: {out_main_t2}")
    print(f"[OK] supplementary prioritized table written to: {out_supp_prioritized}")
    print(f"[OK] supplementary successful models table written to: {out_supp_model_success}")
    print(f"[OK] supplementary failed models table written to: {out_supp_model_failed}")
    print(f"[OK] qc written to: {out_qc}")
    print(f"[OK] summary written to: {out_summary}")
    print(f"[INFO] prioritized significant trios={ranked_sig.shape[0]}")


if __name__ == "__main__":
    main()
