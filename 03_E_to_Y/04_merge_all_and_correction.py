#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

TINY_P = np.nextafter(0, 1)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Robust global merge + multiple testing correction for Step5C chunk outputs."
    )
    ap.add_argument("--mode", required=True, choices=["prevalent", "incident"])
    ap.add_argument("--indir", required=True, help="Directory containing chunk result files")
    ap.add_argument("--outdir", default=None, help="Output directory; default = indir")
    ap.add_argument("--fdr-threshold", type=float, default=0.05)
    return ap.parse_args()


def sanitize_p(x):
    x = pd.to_numeric(x, errors="coerce")
    if pd.isna(x):
        return np.nan
    x = float(x)
    if not np.isfinite(x):
        return np.nan
    if x <= 0:
        return float(TINY_P)
    if x > 1:
        return 1.0
    return x


def find_files(indir: Path, mode: str):
    pat_main = re.compile(rf"step5c\.{mode}\.chunk\d+\.tsv\.gz$")
    pat_skip = re.compile(rf"step5c\.{mode}\.chunk\d+\.skipped\.tsv\.gz$")

    main_files = sorted([p for p in indir.iterdir() if pat_main.fullmatch(p.name)])
    skip_files = sorted([p for p in indir.iterdir() if pat_skip.fullmatch(p.name)])
    return main_files, skip_files


def required_cols_by_mode(mode):
    common = [
        "analysis", "icd3", "gene", "n_used",
        "n_carriers_used", "n_noncarriers_used",
        "beta", "se", "z", "p", "ci95_l", "ci95_u", "model"
    ]
    if mode == "prevalent":
        return common + [
            "n_cases", "n_controls",
            "n_carrier_cases", "n_carrier_controls",
            "or"
        ], "or"
    else:
        return common + [
            "n_events", "n_atrisk",
            "n_carrier_events", "n_carrier_nonevents",
            "hr"
        ], "hr"


def load_and_concat_main(main_files, mode):
    dfs = []
    for f in main_files:
        try:
            df = pd.read_csv(f, sep="\t", compression="infer")
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df["source_file"] = f.name
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)

    # 如果 analysis 列不存在，则补上
    if "analysis" not in df.columns:
        df["analysis"] = mode

    # 保护性过滤：只保留对应 mode
    df = df[df["analysis"] == mode].copy()

    # 如果 chunk 文件中已经有旧的 global 列，统一删掉重算
    for c in ["fdr_q_global", "bonf_p_global"]:
        if c in df.columns:
            df = df.drop(columns=c)

    return df


def convert_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def validate_rows(df, mode, effect_col):
    """
    返回：
      good_df: 可用于多重检验的行
      bad_df: 过滤掉的异常行
    """
    df = df.copy()

    req_cols, _ = required_cols_by_mode(mode)
    missing_required = [c for c in req_cols if c not in df.columns]
    if missing_required:
        raise RuntimeError(f"Missing required columns for mode={mode}: {missing_required}")

    numeric_cols = [
        "n_used", "n_carriers_used", "n_noncarriers_used",
        "beta", "se", "z", "p", "ci95_l", "ci95_u", effect_col
    ]

    if mode == "prevalent":
        numeric_cols += ["n_cases", "n_controls", "n_carrier_cases", "n_carrier_controls"]
    else:
        numeric_cols += ["n_events", "n_atrisk", "n_carrier_events", "n_carrier_nonevents"]

    df = convert_numeric(df, numeric_cols)

    reasons = []

    # 缺失关键字段
    bad_missing = (
        df["icd3"].isna() |
        df["gene"].isna() |
        df["p"].isna() |
        df["beta"].isna() |
        df[effect_col].isna() |
        df["ci95_l"].isna() |
        df["ci95_u"].isna()
    )
    reasons.append(("missing_key_fields", bad_missing))

    # p / effect / CI 非有限
    bad_nonfinite = (
        ~np.isfinite(df["p"]) |
        ~np.isfinite(df["beta"]) |
        ~np.isfinite(df[effect_col]) |
        ~np.isfinite(df["ci95_l"]) |
        ~np.isfinite(df["ci95_u"])
    )
    reasons.append(("nonfinite_numeric", bad_nonfinite))

    # effect/CI 非正
    bad_nonpositive = (
        (df[effect_col] <= 0) |
        (df["ci95_l"] <= 0) |
        (df["ci95_u"] <= 0)
    )
    reasons.append(("nonpositive_effect_or_ci", bad_nonpositive))

    # effect 不在 CI 中
    bad_ci_order = (
        (df["ci95_l"] > df[effect_col]) |
        (df[effect_col] > df["ci95_u"])
    )
    reasons.append(("effect_outside_ci", bad_ci_order))

    # 行为计数不合理
    bad_counts = (
        (df["n_carriers_used"] <= 0) |
        (df["n_noncarriers_used"] <= 0)
    )
    reasons.append(("invalid_carrier_counts", bad_counts))

    if mode == "prevalent":
        bad_counts2 = (
            (df["n_cases"] <= 0) |
            (df["n_controls"] <= 0) |
            (df["n_carrier_cases"] <= 0) |
            (df["n_carrier_controls"] <= 0)
        )
        reasons.append(("invalid_prevalent_cell_counts", bad_counts2))
    else:
        bad_counts2 = (
            (df["n_events"] <= 0) |
            (df["n_atrisk"] <= 0) |
            (df["n_carrier_events"] <= 0) |
            (df["n_carrier_nonevents"] <= 0)
        )
        reasons.append(("invalid_incident_cell_counts", bad_counts2))

    # 汇总 bad mask
    bad_mask = pd.Series(False, index=df.index)
    bad_reason = pd.Series("", index=df.index, dtype="object")

    for reason, mask in reasons:
        bad_reason = np.where((mask) & (~bad_mask), reason, bad_reason)
        bad_mask = bad_mask | mask

    bad_df = df[bad_mask].copy()
    if not bad_df.empty:
        bad_df["filtered_reason"] = bad_reason[bad_mask]

    good_df = df[~bad_mask].copy()

    # p 截断，避免 0.0 进入全局校正
    good_df["p_for_adjust"] = good_df["p"].apply(sanitize_p)
    bad_p2 = good_df["p_for_adjust"].isna()
    if bad_p2.any():
        bad_more = good_df[bad_p2].copy()
        bad_more["filtered_reason"] = "invalid_p_after_sanitize"
        bad_df = pd.concat([bad_df, bad_more], ignore_index=True)
        good_df = good_df[~bad_p2].copy()

    return good_df, bad_df


def deduplicate_rows(df):
    """
    如果目录里混入旧/新 chunk 或重复文件，按 (analysis, icd3, gene) 去重，
    保留 p_for_adjust 最小的一行。
    """
    key = ["analysis", "icd3", "gene"]
    dup_mask = df.duplicated(subset=key, keep=False)

    dup_df = df[dup_mask].copy()
    if not dup_df.empty:
        dup_df = dup_df.sort_values(key + ["p_for_adjust", "source_file"])

    dedup_df = (
        df.sort_values(key + ["p_for_adjust", "source_file"])
          .drop_duplicates(subset=key, keep="first")
          .copy()
    )

    return dedup_df, dup_df


def merge_skipped(skip_files):
    if not skip_files:
        return pd.DataFrame()
    dfs = []
    for f in skip_files:
        try:
            df = pd.read_csv(f, sep="\t", compression="infer")
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        df["source_file"] = f.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=["icd3", "gene", "reason", "detail", "source_file"])


def write_summary(path, mode, n_main_files, n_skip_files,
                  n_rows_raw, n_rows_bad, n_rows_after_dedup,
                  n_fdr, n_bonf):
    lines = [
        f"Mode: {mode}",
        f"Main chunk files merged: {n_main_files}",
        f"Skipped chunk files merged: {n_skip_files}",
        f"Rows before validation: {n_rows_raw}",
        f"Rows filtered before adjustment: {n_rows_bad}",
        f"Rows kept after deduplication: {n_rows_after_dedup}",
        f"Rows with global FDR < 0.05: {n_fdr}",
        f"Rows with global Bonferroni < 0.05: {n_bonf}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    indir = Path(args.indir)
    outdir = Path(args.outdir) if args.outdir else indir
    outdir.mkdir(parents=True, exist_ok=True)

    main_files, skip_files = find_files(indir, args.mode)

    print(f"[INFO] mode={args.mode}")
    print(f"[INFO] main files found: {len(main_files)}")
    print(f"[INFO] skipped files found: {len(skip_files)}")

    if len(main_files) == 0:
        raise RuntimeError(f"No main chunk files found in {indir} for mode={args.mode}")

    raw_df = load_and_concat_main(main_files, args.mode)
    if raw_df.empty:
        raise RuntimeError("Merged main dataframe is empty.")

    _, effect_col = required_cols_by_mode(args.mode)

    good_df, bad_df = validate_rows(raw_df, args.mode, effect_col)
    print(f"[INFO] rows before validation: {len(raw_df)}")
    print(f"[INFO] rows filtered before adjustment: {len(bad_df)}")
    print(f"[INFO] rows remaining after validation: {len(good_df)}")

    if good_df.empty:
        raise RuntimeError("No valid rows remain after validation.")

    dedup_df, dup_df = deduplicate_rows(good_df)
    print(f"[INFO] duplicated rows before dedup: {len(dup_df)}")
    print(f"[INFO] rows kept after dedup: {len(dedup_df)}")

    if dedup_df.empty:
        raise RuntimeError("No rows remain after deduplication.")

    m = len(dedup_df)
    dedup_df["fdr_q_global"] = multipletests(dedup_df["p_for_adjust"].values, method="fdr_bh")[1]
    dedup_df["bonf_p_global"] = np.minimum(dedup_df["p_for_adjust"].values * m, 1.0)

    final_df = dedup_df.drop(columns=["p_for_adjust"]).copy()
    final_df = final_df.sort_values(
        ["fdr_q_global", "p", "icd3", "gene"],
        ascending=[True, True, True, True]
    )

    fdr_df = final_df[final_df["fdr_q_global"] < args.fdr_threshold].copy()
    bonf_df = final_df[final_df["bonf_p_global"] < 0.05].copy()

    prefix = f"step5c.{args.mode}"

    final_df.to_csv(outdir / f"{prefix}.all.global_adjusted.tsv.gz", sep="\t", index=False)
    fdr_df.to_csv(outdir / f"{prefix}.FDR_lt{args.fdr_threshold}.tsv.gz", sep="\t", index=False)
    bonf_df.to_csv(outdir / f"{prefix}.Bonf_lt0.05.tsv.gz", sep="\t", index=False)

    if not bad_df.empty:
        bad_df.to_csv(
            outdir / f"{prefix}.filtered_invalid_before_adjustment.tsv.gz",
            sep="\t", index=False
        )

    if not dup_df.empty:
        dup_df.to_csv(
            outdir / f"{prefix}.duplicated_rows.before_dedup.tsv.gz",
            sep="\t", index=False
        )

    skip_df = merge_skipped(skip_files)
    if not skip_df.empty:
        skip_df.to_csv(
            outdir / f"{prefix}.skipped.merged.tsv.gz",
            sep="\t", index=False
        )

    write_summary(
        outdir / f"{prefix}.merge_summary.txt",
        mode=args.mode,
        n_main_files=len(main_files),
        n_skip_files=len(skip_files),
        n_rows_raw=len(raw_df),
        n_rows_bad=len(bad_df),
        n_rows_after_dedup=len(dedup_df),
        n_fdr=len(fdr_df),
        n_bonf=len(bonf_df),
    )

    print("[DONE]")
    print(f"[DONE] Saved: {outdir / f'{prefix}.all.global_adjusted.tsv.gz'}")
    print(f"[DONE] Saved: {outdir / f'{prefix}.FDR_lt{args.fdr_threshold}.tsv.gz'}")
    print(f"[DONE] Saved: {outdir / f'{prefix}.Bonf_lt0.05.tsv.gz'}")


if __name__ == "__main__":
    main()
