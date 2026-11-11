#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from statsmodels.stats.multitest import multipletests


BOOT_STABILITY_THRESHOLD = 0.80


def safe_read_tsv(path, dtype=None):
    """
    Safe reader for .tsv / .tsv.gz
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    if os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t", dtype=dtype, compression="infer")
    except EmptyDataError:
        return pd.DataFrame()


def is_main_result_file(path):
    name = os.path.basename(path)
    return (
        name.startswith("01_mediation.chunk")
        and name.endswith(".tsv.gz")
        and (not name.endswith(".skipped.tsv.gz"))
    )


def add_fdr_column(df, p_col, out_col):
    df[out_col] = np.nan
    if p_col not in df.columns:
        return df

    p = pd.to_numeric(df[p_col], errors="coerce")
    valid = p.notna() & np.isfinite(p) & (p >= 0) & (p <= 1)
    if valid.sum() == 0:
        return df

    q = np.full(df.shape[0], np.nan, dtype=float)
    q[valid.to_numpy()] = multipletests(p.loc[valid].to_numpy(), method="fdr_bh")[1]
    df[out_col] = q
    return df


def supported_mediation_mask(df, boot_threshold=BOOT_STABILITY_THRESHOLD):
    """
    Supported mediation definition:
    model success + bootstrap stability + FDR-significant indirect effect.
    """
    return (
        (df["model_success"] == 1)
        & (pd.to_numeric(df["boot_success_rate"], errors="coerce") >= boot_threshold)
        & (df["indirect_sig_boot_fdr"] == 1)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate chunk-level mediation results and build final mediation result table"
    )
    parser.add_argument("--indir", required=True, help="Step5 mediation directory")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--boot_stability_threshold", type=float, default=0.80)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    out_all = os.path.join(args.outdir, "02_mediation_results.all.tsv.gz")
    out_skipped = os.path.join(args.outdir, "02_mediation_results.skipped.tsv.gz")
    out_qc = os.path.join(args.outdir, "02_mediation_results.qc.tsv")
    out_summary = os.path.join(args.outdir, "02_mediation_results.summary.txt")

    all_tsv = sorted(glob.glob(os.path.join(args.indir, "01_mediation.chunk*.tsv.gz")))
    main_files = [f for f in all_tsv if is_main_result_file(f)]
    skipped_files = sorted(glob.glob(os.path.join(args.indir, "01_mediation.chunk*.skipped.tsv.gz")))

    if not main_files:
        raise RuntimeError(f"No main mediation result files found in {args.indir}")

    # -------------------------
    # 1. merge main result files
    # -------------------------
    res_dfs = []
    for f in main_files:
        df = safe_read_tsv(
            f,
            dtype={"trio_id": str, "gene": str, "protein": str, "icd3": str, "tier_label": str, "table_path": str}
        )
        if not df.empty:
            df["source_result_file"] = os.path.basename(f)
            res_dfs.append(df)

    if not res_dfs:
        raise RuntimeError("All main mediation result files are empty.")

    res = pd.concat(res_dfs, axis=0, ignore_index=True)

    # -------------------------
    # 2. merge skipped files
    # -------------------------
    skip_dfs = []
    for f in skipped_files:
        df = safe_read_tsv(
            f,
            dtype={"trio_id": str, "gene": str, "protein": str, "icd3": str, "tier_label": str, "table_path": str}
        )
        if not df.empty:
            df["source_skipped_file"] = os.path.basename(f)
            skip_dfs.append(df)

    skipped = pd.concat(skip_dfs, axis=0, ignore_index=True) if skip_dfs else pd.DataFrame()

    # -------------------------
    # 3. required columns
    # -------------------------
    required_cols = {
        "trio_id", "gene", "protein", "icd3", "tier_label", "table_path",
        "model_success", "fail_reason"
    }
    miss = required_cols - set(res.columns)
    if miss:
        raise ValueError(f"Merged mediation results missing required columns: {sorted(miss)}")

    # -------------------------
    # 4. deduplicate trio_id
    # -------------------------
    n_rows_before_dedup = res.shape[0]
    dup_mask = res.duplicated(subset=["trio_id"], keep=False)
    dup_rows = res[dup_mask].copy()

    # keep successful row first if duplicate
    if "model_success" in res.columns:
        res["model_success"] = pd.to_numeric(res["model_success"], errors="coerce").fillna(0).astype(int)

    sort_cols = ["trio_id"]
    ascending = [True]
    if "model_success" in res.columns:
        sort_cols.append("model_success")
        ascending.append(False)

    res = (
        res.sort_values(sort_cols, ascending=ascending)
           .drop_duplicates(subset=["trio_id"], keep="first")
           .copy()
    )

    # -------------------------
    # 5. numeric coercion
    # -------------------------
    num_cols = [
        "model_success",
        "n_total_rows", "n_complete_case",
        "n_events_complete_case", "n_carriers_complete_case",
        "n_noncarriers_complete_case", "n_carrier_events_complete_case",
        "n_carrier_nonevents_complete_case", "M_sd_complete_case",
        "beta_EM_individual", "se_EM_individual", "p_EM_individual",
        "beta_EY_total", "se_EY_total", "hr_EY_total",
        "beta_EY_direct", "se_EY_direct", "hr_EY_direct",
        "beta_MY_conditional", "se_MY_conditional", "hr_MY_conditional",
        "indirect_logHR", "indirect_hr",
        "indirect_p_boot", "proportion_mediated_logscale",
        "n_boot_requested", "n_boot_success", "boot_success_rate",
        "indirect_boot_mean", "indirect_boot_median", "indirect_boot_sd"
    ]
    for c in num_cols:
        if c in res.columns:
            res[c] = pd.to_numeric(res[c], errors="coerce")

    res["table_exists"] = res["table_path"].astype(str).apply(os.path.exists).astype(int)

    # -------------------------
    # 6. FDR for indirect_p_boot
    # -------------------------
    res = add_fdr_column(res, "indirect_p_boot", "indirect_q_boot")

    # helpful flags
    if "indirect_p_boot" in res.columns:
        res["indirect_sig_boot_nominal"] = (
            pd.to_numeric(res["indirect_p_boot"], errors="coerce") < args.alpha
        ).astype(int)
    else:
        res["indirect_sig_boot_nominal"] = 0

    if "indirect_q_boot" in res.columns:
        res["indirect_sig_boot_fdr"] = (
            pd.to_numeric(res["indirect_q_boot"], errors="coerce") < args.alpha
        ).astype(int)
    else:
        res["indirect_sig_boot_fdr"] = 0

    # order
    sort_cols = ["tier_label", "gene", "protein", "icd3"]
    sort_cols = [c for c in sort_cols if c in res.columns]
    res = res.sort_values(sort_cols).copy()

    # -------------------------
    # 7. outputs
    # -------------------------
    res.to_csv(out_all, sep="\t", index=False, compression="gzip")

    if not skipped.empty:
        skipped.to_csv(out_skipped, sep="\t", index=False, compression="gzip")
    else:
        pd.DataFrame().to_csv(out_skipped, sep="\t", index=False, compression="gzip")

    # -------------------------
    # 8. QC
    # -------------------------
    qc_rows = [
        {"metric": "n_chunk_result_files", "value": len(main_files)},
        {"metric": "n_chunk_skipped_files", "value": len(skipped_files)},
        {"metric": "n_rows_before_dedup", "value": n_rows_before_dedup},
        {"metric": "n_duplicate_trio_rows", "value": dup_rows.shape[0]},
        {"metric": "n_rows_output_all", "value": res.shape[0]},
        {"metric": "n_rows_with_existing_table", "value": int((res["table_exists"] == 1).sum())},
        {"metric": "n_rows_missing_table_file", "value": int((res["table_exists"] == 0).sum())},
        {"metric": "n_model_success", "value": int((res["model_success"] == 1).sum())},
        {"metric": "n_model_failed", "value": int((res["model_success"] != 1).sum())},
    ]

    if "indirect_sig_boot_nominal" in res.columns:
        qc_rows.append({"metric": "n_indirect_sig_boot_nominal", "value": int((res["indirect_sig_boot_nominal"] == 1).sum())})
    if "indirect_sig_boot_fdr" in res.columns:
        qc_rows.append({"metric": "n_indirect_sig_boot_fdr", "value": int((res["indirect_sig_boot_fdr"] == 1).sum())})
    if "boot_success_rate" in res.columns:
        qc_rows.append({"metric": "median_boot_success_rate", "value": float(pd.to_numeric(res["boot_success_rate"], errors="coerce").median())})

    for tier in ["Tier1", "Tier2", "Tier3"]:
        sub = res[res["tier_label"] == tier]
        qc_rows.extend([
            {"metric": f"n_{tier}", "value": sub.shape[0]},
            {"metric": f"n_{tier}_model_success", "value": int((sub["model_success"] == 1).sum()) if not sub.empty else 0},
            {"metric": f"n_{tier}_model_failed", "value": int((sub["model_success"] != 1).sum()) if not sub.empty else 0},
        ])
        if not sub.empty and "indirect_sig_boot_nominal" in sub.columns:
            qc_rows.append({"metric": f"n_{tier}_indirect_sig_boot_nominal", "value": int((sub["indirect_sig_boot_nominal"] == 1).sum())})
        if not sub.empty and "indirect_sig_boot_fdr" in sub.columns:
            qc_rows.append({"metric": f"n_{tier}_indirect_sig_boot_fdr", "value": int((sub["indirect_sig_boot_fdr"] == 1).sum())})

    fail_counts = res.loc[res["model_success"] != 1, "fail_reason"].astype(str).value_counts()
    for reason, n in fail_counts.items():
        if reason.strip():
            qc_rows.append({"metric": f"fail::{reason}", "value": int(n)})

    if not skipped.empty and "fail_reason" in skipped.columns:
        skip_counts = skipped["fail_reason"].astype(str).value_counts()
        for reason, n in skip_counts.items():
            qc_rows.append({"metric": f"skipped::{reason}", "value": int(n)})

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(out_qc, sep="\t", index=False)

    # -------------------------
    # 9. summary
    # -------------------------
    lines = []
    lines.append("Mediation results summary")
    lines.append("========================")
    lines.append("")
    lines.append(f"Chunk result files merged      : {len(main_files)}")
    lines.append(f"Chunk skipped files merged     : {len(skipped_files)}")
    lines.append(f"Rows before deduplication      : {n_rows_before_dedup}")
    lines.append(f"Rows in final result table     : {res.shape[0]}")
    lines.append(f"Model success                  : {int((res['model_success'] == 1).sum())}")
    lines.append(f"Model failed                   : {int((res['model_success'] != 1).sum())}")
    if "indirect_sig_boot_nominal" in res.columns:
        lines.append(f"Indirect p<{args.alpha} (bootstrap)    : {int((res['indirect_sig_boot_nominal'] == 1).sum())}")
    if "indirect_sig_boot_fdr" in res.columns:
        lines.append(f"Indirect FDR<{args.alpha} (bootstrap)  : {int((res['indirect_sig_boot_fdr'] == 1).sum())}")
    if "boot_success_rate" in res.columns:
        lines.append(f"Median bootstrap success rate  : {float(pd.to_numeric(res['boot_success_rate'], errors='coerce').median()):.4f}")
    lines.append("")

    for tier in ["Tier1", "Tier2", "Tier3"]:
        sub = res[res["tier_label"] == tier]
        supported_sub = sub[supported_mediation_mask(sub, args.boot_stability_threshold)] if not sub.empty else sub

        lines.append(f"{tier}:")
        lines.append(f"  N rows                       : {sub.shape[0]}")
        lines.append(f"  Model success                : {int((sub['model_success'] == 1).sum()) if not sub.empty else 0}")
        if tier in ["Tier1", "Tier2"]:
            lines.append(f"  Candidate proteins           : {int(sub['protein'].dropna().nunique()) if not sub.empty else 0}")
            lines.append(f"  Supported mediation proteins : {int(supported_sub['protein'].dropna().nunique()) if not supported_sub.empty else 0}")
        if not sub.empty and "indirect_sig_boot_nominal" in sub.columns:
            lines.append(f"  Indirect p<{args.alpha}              : {int((sub['indirect_sig_boot_nominal'] == 1).sum())}")
        if not sub.empty and "indirect_sig_boot_fdr" in sub.columns:
            lines.append(f"  Indirect FDR<{args.alpha}            : {int((sub['indirect_sig_boot_fdr'] == 1).sum())}")
        lines.append("")

    with open(out_summary, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[OK] merged mediation results written to: {out_all}")
    print(f"[OK] merged skipped rows written to: {out_skipped}")
    print(f"[OK] qc written to: {out_qc}")
    print(f"[OK] summary written to: {out_summary}")
    print(f"[INFO] final result rows={res.shape[0]}")


if __name__ == "__main__":
    main()