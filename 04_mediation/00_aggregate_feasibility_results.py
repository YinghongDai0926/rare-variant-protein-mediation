#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import pandas as pd
import numpy as np
from pandas.errors import EmptyDataError


def safe_read_tsv(path, dtype=None):
    if not os.path.exists(path):
        return pd.DataFrame()
    if os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t", dtype=dtype)
    except EmptyDataError:
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate chunk-level trio feasibility scan results"
    )
    parser.add_argument("--indir", required=True, help="Step4_feasibility directory")
    parser.add_argument("--outdir", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    out_manifest = os.path.join(args.outdir, "02_feasibility_manifest.tsv")
    out_qc = os.path.join(args.outdir, "02_feasibility_manifest.qc.tsv")
    out_summary = os.path.join(args.outdir, "02_feasibility_summary.txt")
    out_skipped = os.path.join(args.outdir, "02_feasibility_skipped.tsv")

    scan_files = sorted(glob.glob(os.path.join(args.indir, "01_feasibility.chunk[0-9]*.tsv")))
    scan_files = [f for f in scan_files if not f.endswith(".qc.tsv") and not f.endswith(".skipped.tsv")]

    skipped_files = sorted(glob.glob(os.path.join(args.indir, "01_feasibility.chunk[0-9]*.skipped.tsv")))

    if not scan_files:
        raise RuntimeError(f"No chunk scan files found in {args.indir}")

    # 1. merge scan files
    scan_dfs = []
    for f in scan_files:
        df = safe_read_tsv(
            f,
            dtype={"trio_id": str, "gene": str, "protein": str, "icd3": str, "tier_label": str, "table_path": str}
        )
        if not df.empty:
            df["source_scan_file"] = os.path.basename(f)
            scan_dfs.append(df)

    if not scan_dfs:
        raise RuntimeError("All chunk feasibility scan files are empty.")

    manifest = pd.concat(scan_dfs, axis=0, ignore_index=True)

    # 2. merge skipped files
    skipped_dfs = []
    for f in skipped_files:
        df = safe_read_tsv(
            f,
            dtype={"trio_id": str, "gene": str, "protein": str, "icd3": str, "tier_label": str, "table_path": str}
        )
        if not df.empty:
            df["source_skipped_file"] = os.path.basename(f)
            skipped_dfs.append(df)

    skipped = pd.concat(skipped_dfs, axis=0, ignore_index=True) if skipped_dfs else pd.DataFrame()

    # 3. required columns
    required_manifest = {
        "trio_id", "gene", "protein", "icd3", "tier_label", "table_path",
        "n_total_rows", "n_eligible_rows", "n_complete_case",
        "n_events_complete_case", "n_carriers_complete_case", "n_noncarriers_complete_case",
        "n_carrier_events_complete_case", "n_carrier_nonevents_complete_case",
        "n_noncarrier_events_complete_case", "n_noncarrier_nonevents_complete_case",
        "M_sd_complete_case", "M_missing_rate", "cov_missing_rate",
        "pass_feasibility_qc_prelim", "fail_reason_prelim"
    }
    miss = required_manifest - set(manifest.columns)
    if miss:
        raise ValueError(f"Merged feasibility manifest missing required columns: {sorted(miss)}")

    # 4. deduplicate trio_id
    n_rows_before_dedup = manifest.shape[0]
    dup_mask = manifest.duplicated(subset=["trio_id"], keep=False)
    dup_rows = manifest[dup_mask].copy()

    manifest = (
        manifest.sort_values(["trio_id", "tier_label"])
                .drop_duplicates(subset=["trio_id"], keep="first")
                .copy()
    )

    # numeric coercion
    num_cols = [
        "n_total_rows", "n_eligible_rows", "n_complete_case",
        "n_events_complete_case", "n_carriers_complete_case", "n_noncarriers_complete_case",
        "n_carrier_events_complete_case", "n_carrier_nonevents_complete_case",
        "n_noncarrier_events_complete_case", "n_noncarrier_nonevents_complete_case",
        "M_sd_complete_case", "M_missing_rate", "cov_missing_rate",
        "pass_feasibility_qc_prelim"
    ]
    for c in num_cols:
        manifest[c] = pd.to_numeric(manifest[c], errors="coerce")

    manifest["table_exists"] = manifest["table_path"].astype(str).apply(os.path.exists).astype(int)

    manifest = manifest.sort_values(
        ["tier_label", "gene", "protein", "icd3"],
        ascending=[True, True, True, True]
    ).copy()

    manifest.to_csv(out_manifest, sep="\t", index=False)

    if not skipped.empty:
        skipped.to_csv(out_skipped, sep="\t", index=False)
    else:
        pd.DataFrame().to_csv(out_skipped, sep="\t", index=False)

    # 5. QC
    qc_rows = [
        {"metric": "n_chunk_scan_files", "value": len(scan_files)},
        {"metric": "n_chunk_skipped_files", "value": len(skipped_files)},
        {"metric": "n_rows_before_dedup", "value": n_rows_before_dedup},
        {"metric": "n_duplicate_trio_rows", "value": dup_rows.shape[0]},
        {"metric": "n_rows_output_manifest", "value": manifest.shape[0]},
        {"metric": "n_rows_with_existing_table", "value": int((manifest["table_exists"] == 1).sum())},
        {"metric": "n_rows_missing_table_file", "value": int((manifest["table_exists"] == 0).sum())},
        {"metric": "n_rows_pass_feasibility_qc_prelim", "value": int((manifest["pass_feasibility_qc_prelim"] == 1).sum())},
        {"metric": "n_rows_fail_feasibility_qc_prelim", "value": int((manifest["pass_feasibility_qc_prelim"] != 1).sum())},
    ]

    for tier in ["Tier1", "Tier2", "Tier3"]:
        sub = manifest[manifest["tier_label"] == tier]
        qc_rows.extend([
            {"metric": f"n_{tier}", "value": sub.shape[0]},
            {"metric": f"n_{tier}_pass_feasibility_qc_prelim", "value": int((sub["pass_feasibility_qc_prelim"] == 1).sum()) if not sub.empty else 0},
            {"metric": f"n_{tier}_fail_feasibility_qc_prelim", "value": int((sub["pass_feasibility_qc_prelim"] != 1).sum()) if not sub.empty else 0},
            {"metric": f"median_n_complete_case_{tier}", "value": float(sub["n_complete_case"].median()) if not sub.empty else np.nan},
            {"metric": f"median_n_events_complete_case_{tier}", "value": float(sub["n_events_complete_case"].median()) if not sub.empty else np.nan},
            {"metric": f"median_n_carriers_complete_case_{tier}", "value": float(sub["n_carriers_complete_case"].median()) if not sub.empty else np.nan},
            {"metric": f"median_n_carrier_events_complete_case_{tier}", "value": float(sub["n_carrier_events_complete_case"].median()) if not sub.empty else np.nan},
            {"metric": f"median_M_missing_rate_{tier}", "value": float(sub["M_missing_rate"].median()) if not sub.empty else np.nan},
        ])

    fail_counts = manifest.loc[
        manifest["pass_feasibility_qc_prelim"] != 1, "fail_reason_prelim"
    ].astype(str).value_counts()
    for reason, n in fail_counts.items():
        if reason.strip():
            qc_rows.append({"metric": f"fail::{reason}", "value": int(n)})

    if not skipped.empty and "reason" in skipped.columns:
        skip_reason_counts = skipped["reason"].astype(str).value_counts()
        for reason, n in skip_reason_counts.items():
            qc_rows.append({"metric": f"skipped::{reason}", "value": int(n)})

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(out_qc, sep="\t", index=False)

    # 6. summary txt
    lines = []
    lines.append("Feasibility manifest summary")
    lines.append("============================")
    lines.append("")
    lines.append(f"Chunk scan files merged          : {len(scan_files)}")
    lines.append(f"Chunk skipped files merged       : {len(skipped_files)}")
    lines.append(f"Rows before deduplication        : {n_rows_before_dedup}")
    lines.append(f"Rows in final manifest           : {manifest.shape[0]}")
    lines.append(f"Rows passing prelim feasibility  : {int((manifest['pass_feasibility_qc_prelim'] == 1).sum())}")
    lines.append(f"Rows failing prelim feasibility  : {int((manifest['pass_feasibility_qc_prelim'] != 1).sum())}")
    lines.append("")

    for tier in ["Tier1", "Tier2", "Tier3"]:
        sub = manifest[manifest["tier_label"] == tier]
        lines.append(f"{tier}:")
        lines.append(f"  N rows                         : {sub.shape[0]}")
        lines.append(f"  Pass prelim feasibility        : {int((sub['pass_feasibility_qc_prelim'] == 1).sum()) if not sub.empty else 0}")
        lines.append(f"  Median n_complete_case         : {float(sub['n_complete_case'].median()) if not sub.empty else 'NA'}")
        lines.append(f"  Median n_events_complete_case  : {float(sub['n_events_complete_case'].median()) if not sub.empty else 'NA'}")
        lines.append(f"  Median n_carriers_complete_case: {float(sub['n_carriers_complete_case'].median()) if not sub.empty else 'NA'}")
        lines.append(f"  Median n_carrier_events_cc     : {float(sub['n_carrier_events_complete_case'].median()) if not sub.empty else 'NA'}")
        lines.append("")
    
    with open(out_summary, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[OK] feasibility manifest written to: {out_manifest}")
    print(f"[OK] qc written to: {out_qc}")
    print(f"[OK] summary written to: {out_summary}")
    print(f"[OK] skipped summary written to: {out_skipped}")
    print(f"[INFO] final feasibility manifest rows={manifest.shape[0]}")


if __name__ == "__main__":
    main()
