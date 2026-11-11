#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Prioritize mediation-supported trios by indirect effect size"
    )
    parser.add_argument("--mediation_results_tsv", required=True, help="02_mediation_results.all.tsv.gz")
    parser.add_argument("--outdir", required=True, help="Output directory")

    # stability + significance thresholds
    parser.add_argument("--min_boot_success_rate", type=float, default=0.80)
    parser.add_argument("--min_n_boot_success", type=int, default=100)
    parser.add_argument("--require_fdr", type=int, default=1, choices=[0, 1],
                        help="1: require indirect_q_boot < 0.05; 0: only require indirect_p_boot < 0.05")
    parser.add_argument("--alpha", type=float, default=0.05)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    out_all_ranked = os.path.join(args.outdir, "03_mediation_ranked_all.tsv.gz")
    out_sig_ranked = os.path.join(args.outdir, "03_mediation_ranked_significant.tsv.gz")
    out_t1_ranked = os.path.join(args.outdir, "03_mediation_ranked_Tier1.tsv.gz")
    out_t2_ranked = os.path.join(args.outdir, "03_mediation_ranked_Tier2.tsv.gz")
    out_qc = os.path.join(args.outdir, "03_mediation_ranked.qc.tsv")
    out_summary = os.path.join(args.outdir, "03_mediation_ranked.summary.txt")

    df = pd.read_csv(
        args.mediation_results_tsv,
        sep="\t",
        compression="infer",
        dtype={"trio_id": str, "gene": str, "protein": str, "icd3": str, "tier_label": str, "table_path": str}
    )

    required_cols = {
        "trio_id", "gene", "protein", "icd3", "tier_label",
        "model_success", "indirect_logHR", "indirect_hr",
        "indirect_p_boot", "indirect_q_boot",
        "boot_success_rate", "n_boot_success",
        "proportion_mediated_logscale"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"mediation_results_tsv missing required columns: {sorted(missing)}")

    # numeric coercion
    num_cols = [
        "model_success", "indirect_logHR", "indirect_hr",
        "indirect_p_boot", "indirect_q_boot",
        "boot_success_rate", "n_boot_success",
        "proportion_mediated_logscale",
        "n_complete_case", "n_events_complete_case",
        "n_carriers_complete_case", "n_carrier_events_complete_case",
        "beta_EY_total", "beta_EY_direct", "beta_EM_individual", "beta_MY_conditional"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # stability + significance flags
    df["pass_model_success"] = (df["model_success"] == 1).astype(int)
    df["pass_boot_stability"] = (
        (df["boot_success_rate"] >= args.min_boot_success_rate) &
        (df["n_boot_success"] >= args.min_n_boot_success)
    ).astype(int)

    if args.require_fdr == 1:
        df["pass_indirect_significance"] = (df["indirect_q_boot"] < args.alpha).astype(int)
    else:
        df["pass_indirect_significance"] = (df["indirect_p_boot"] < args.alpha).astype(int)

    df["pass_priority_filter"] = (
        (df["pass_model_success"] == 1) &
        (df["pass_boot_stability"] == 1) &
        (df["pass_indirect_significance"] == 1)
    ).astype(int)

    # helpful attenuation metrics
    if {"beta_EY_total", "beta_EY_direct"}.issubset(df.columns):
        df["direct_attenuation_logscale"] = df["beta_EY_total"] - df["beta_EY_direct"]
    else:
        df["direct_attenuation_logscale"] = np.nan

    # all ranked (successful models first, then by indirect effect descending)
    df_all = df.copy()
    df_all = df_all.sort_values(
        ["pass_priority_filter", "indirect_logHR", "boot_success_rate", "tier_label", "gene", "protein", "icd3"],
        ascending=[False, False, False, True, True, True, True]
    ).copy()
    df_all["rank_overall_all"] = np.arange(1, df_all.shape[0] + 1)

    # significant + stable ranked set
    sig = df[df["pass_priority_filter"] == 1].copy()
    sig = sig.sort_values(
        ["indirect_logHR", "boot_success_rate", "tier_label", "gene", "protein", "icd3"],
        ascending=[False, False, True, True, True, True]
    ).copy()
    sig["rank_overall_significant"] = np.arange(1, sig.shape[0] + 1)

    # within-tier ranks
    for tier in ["Tier1", "Tier2"]:
        mask = sig["tier_label"] == tier
        sig.loc[mask, "rank_within_tier"] = np.arange(1, mask.sum() + 1)

    # tier-specific outputs
    sig_t1 = sig[sig["tier_label"] == "Tier1"].copy()
    sig_t2 = sig[sig["tier_label"] == "Tier2"].copy()

    # write outputs
    df_all.to_csv(out_all_ranked, sep="\t", index=False, compression="gzip")
    sig.to_csv(out_sig_ranked, sep="\t", index=False, compression="gzip")
    sig_t1.to_csv(out_t1_ranked, sep="\t", index=False, compression="gzip")
    sig_t2.to_csv(out_t2_ranked, sep="\t", index=False, compression="gzip")

    # QC
    qc_rows = [
        {"metric": "n_input_rows", "value": df.shape[0]},
        {"metric": "require_fdr", "value": args.require_fdr},
        {"metric": "min_boot_success_rate", "value": args.min_boot_success_rate},
        {"metric": "min_n_boot_success", "value": args.min_n_boot_success},
        {"metric": "n_pass_model_success", "value": int((df["pass_model_success"] == 1).sum())},
        {"metric": "n_pass_boot_stability", "value": int((df["pass_boot_stability"] == 1).sum())},
        {"metric": "n_pass_indirect_significance", "value": int((df["pass_indirect_significance"] == 1).sum())},
        {"metric": "n_pass_priority_filter", "value": int((df["pass_priority_filter"] == 1).sum())},
        {"metric": "n_ranked_Tier1", "value": sig_t1.shape[0]},
        {"metric": "n_ranked_Tier2", "value": sig_t2.shape[0]},
    ]

    if not sig.empty:
        qc_rows.extend([
            {"metric": "top1_indirect_logHR", "value": float(sig["indirect_logHR"].iloc[0])},
            {"metric": "median_indirect_logHR_ranked", "value": float(sig["indirect_logHR"].median())},
            {"metric": "median_boot_success_rate_ranked", "value": float(sig["boot_success_rate"].median())},
            {"metric": "median_proportion_mediated_ranked", "value": float(pd.to_numeric(sig["proportion_mediated_logscale"], errors="coerce").median())},
        ])

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(out_qc, sep="\t", index=False)

    # summary
    lines = []
    lines.append("Prioritized mediation hits summary")
    lines.append("=================================")
    lines.append("")
    lines.append(f"Input rows                               : {df.shape[0]}")
    lines.append(f"Require FDR                              : {args.require_fdr}")
    lines.append(f"Minimum bootstrap success rate           : {args.min_boot_success_rate}")
    lines.append(f"Minimum successful bootstrap replicates  : {args.min_n_boot_success}")
    lines.append(f"Pass model success                       : {int((df['pass_model_success'] == 1).sum())}")
    lines.append(f"Pass bootstrap stability                 : {int((df['pass_boot_stability'] == 1).sum())}")
    lines.append(f"Pass indirect significance               : {int((df['pass_indirect_significance'] == 1).sum())}")
    lines.append(f"Final prioritized mediation hits         : {int((df['pass_priority_filter'] == 1).sum())}")
    lines.append("")
    lines.append(f"Tier1 prioritized hits                   : {sig_t1.shape[0]}")
    lines.append(f"Tier2 prioritized hits                   : {sig_t2.shape[0]}")
    lines.append("")
    lines.append("Ranking rule")
    lines.append("------------")
    lines.append("Stable and significant trios are ranked by indirect_logHR in descending order.")
    lines.append("This ranking is intended to reflect how much the protein mediator explains the genetic effect under the current model.")

    with open(out_summary, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[OK] ranked all results written to: {out_all_ranked}")
    print(f"[OK] ranked significant results written to: {out_sig_ranked}")
    print(f"[OK] ranked Tier1 results written to: {out_t1_ranked}")
    print(f"[OK] ranked Tier2 results written to: {out_t2_ranked}")
    print(f"[OK] qc written to: {out_qc}")
    print(f"[OK] summary written to: {out_summary}")
    print(f"[INFO] prioritized hits={sig.shape[0]}")


if __name__ == "__main__":
    main()
