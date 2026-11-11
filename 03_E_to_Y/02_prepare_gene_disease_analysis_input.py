#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step5 input preparation (one script):
  1) Rebuild true icd3 counts from sample-level phenotype files:
       - icd10_first_occurrence.tsv: sample icd3 first_date
       - followup_dates.tsv: sample baseline_date death_date censor_date
     Outputs:
       - inc_icd3_counts.tsv: icd3 n_events n_atrisk
       - prev_icd3_counts.tsv: icd3 n_cases n_controls

  2) Carrier QC for gene-based burden:
       - E_matrix.carrier_counts.tsv (gene n_carriers)
       - optionally: E_matrix.carrier_binary.npz + genes.txt + samples.txt
         -> can recompute n_carriers within valid-followup sample subset
     Outputs:
       - carriers_qc.tsv
       - carriers_qc.flagged.tsv
       - genes.exclude_step5B.txt
       - carriers_qc.summary.txt

Robustness:
  - chunked reading for first_occurrence
  - date parsing with coercion
  - warns if baseline_date looks suspicious (e.g., many 1970-01-01)
"""

import argparse
import os
import sys
import gzip
from collections import Counter
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

try:
    from scipy import sparse
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# -----------------------------
# Utils
# -----------------------------
def die(msg: str, code: int = 1):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(code)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def read_samples_from_tsv(path: str, sample_col: str = "sample") -> List[str]:
    df = pd.read_csv(path, sep="\t", usecols=[sample_col], dtype={sample_col: str})
    # preserve order, drop duplicates
    seen = set()
    out = []
    for s in df[sample_col].astype(str).tolist():
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out

def read_samples_from_list(path: str) -> List[str]:
    out = []
    seen = set()
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s not in seen:
                out.append(s)
                seen.add(s)
    return out

def parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def min_date_two(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    element-wise min for datetime series, ignoring NaT
    """
    out = a.copy()
    mask_a = out.isna()
    out[mask_a] = b[mask_a]
    both = (~a.isna()) & (~b.isna())
    out[both] = np.minimum(a[both].values.astype("datetime64[ns]"),
                           b[both].values.astype("datetime64[ns]"))
    return out

def write_text(path: str, text: str):
    with open(path, "w") as f:
        f.write(text)


# -----------------------------
# A) Build icd3 counts
# -----------------------------
def load_followup(followup_path: str, samples_set: set, log_lines: List[str]) -> pd.DataFrame:
    fu = pd.read_csv(followup_path, sep="\t", dtype={"sample": str})
    if "sample" not in fu.columns:
        die("followup_dates.tsv must contain column: sample")

    need = ["baseline_date", "censor_date"]
    for c in need:
        if c not in fu.columns:
            die(f"followup_dates.tsv missing required column: {c}")

    if "death_date" not in fu.columns:
        fu["death_date"] = ""

    fu = fu[fu["sample"].isin(samples_set)].copy()
    fu["baseline_date"] = parse_date_series(fu["baseline_date"])
    fu["death_date"] = parse_date_series(fu["death_date"])
    fu["censor_date"] = parse_date_series(fu["censor_date"])
    fu["end_date"] = min_date_two(fu["censor_date"], fu["death_date"])

    # basic QC
    n0 = fu.shape[0]
    fu_valid = fu.dropna(subset=["baseline_date", "end_date"]).copy()
    n_valid = fu_valid["sample"].nunique()

    log_lines.append(f"[INFO] Followup rows (in analysis samples): {n0}")
    log_lines.append(f"[INFO] Valid followup (have baseline_date & end_date): {n_valid}")

    # suspicious baseline check (do not drop automatically; just warn)
    if n_valid > 0:
        baseline_counts = fu_valid["baseline_date"].value_counts(dropna=False)
        top_date = baseline_counts.index[0]
        top_prop = baseline_counts.iloc[0] / float(fu_valid.shape[0])
        if str(top_date)[:10] == "1970-01-01" and top_prop > 0.2:
            log_lines.append(
                f"[WARN] baseline_date has high proportion of 1970-01-01: {top_prop:.2%}. "
                f"If unintended (epoch-default), revisit followup_dates.tsv generation."
            )

    # ensure end>=baseline
    bad = (fu_valid["end_date"] < fu_valid["baseline_date"]).sum()
    if bad > 0:
        log_lines.append(f"[WARN] Found {bad} rows with end_date < baseline_date; dropping them.")
        fu_valid = fu_valid[fu_valid["end_date"] >= fu_valid["baseline_date"]].copy()

    fu_valid = fu_valid.drop_duplicates(subset=["sample"], keep="first")
    fu_valid = fu_valid.set_index("sample", drop=True)
    return fu_valid[["baseline_date", "end_date"]]

def load_occurrence_chunked(first_occ_path: str,
                            valid_samples_set: set,
                            chunksize: int,
                            log_lines: List[str]) -> pd.DataFrame:
    # required columns
    # sample icd3 first_date
    # chunk -> groupby min per sample,icd3 -> concat -> groupby min again
    usecols = ["sample", "icd3", "first_date"]
    agg_list = []
    n_read = 0
    n_keep = 0

    it = pd.read_csv(first_occ_path, sep="\t", usecols=usecols, dtype={"sample": str, "icd3": str},
                     chunksize=chunksize)
    for chunk in it:
        n_read += chunk.shape[0]
        chunk = chunk[chunk["sample"].isin(valid_samples_set)].copy()
        if chunk.shape[0] == 0:
            continue
        chunk["first_date"] = parse_date_series(chunk["first_date"])
        chunk = chunk.dropna(subset=["first_date"])
        if chunk.shape[0] == 0:
            continue
        # earliest per sample×icd3 within chunk
        g = chunk.groupby(["sample", "icd3"], as_index=False)["first_date"].min()
        n_keep += g.shape[0]
        agg_list.append(g)

    if not agg_list:
        log_lines.append(f"[INFO] first_occurrence: read={n_read}, kept=0 (no events after filtering).")
        return pd.DataFrame(columns=["sample", "icd3", "first_date"])

    occ = pd.concat(agg_list, axis=0, ignore_index=True)
    # global earliest per sample×icd3
    occ = occ.groupby(["sample", "icd3"], as_index=False)["first_date"].min()
    log_lines.append(f"[INFO] first_occurrence: read={n_read}, kept(after filter+min)={occ.shape[0]}")
    return occ

def build_icd3_counts(samples: List[str],
                      first_occ_path: str,
                      followup_path: str,
                      out_inc: str,
                      out_prev: str,
                      chunksize: int,
                      log_lines: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    samples_set = set(samples)

    fu_valid = load_followup(followup_path, samples_set, log_lines)
    valid_samples = fu_valid.index.astype(str).tolist()
    valid_samples_set = set(valid_samples)
    N_total = len(valid_samples)

    if N_total == 0:
        die("No valid followup samples after filtering. Check followup_dates.tsv baseline/end dates.")

    occ = load_occurrence_chunked(first_occ_path, valid_samples_set, chunksize, log_lines)
    if occ.empty:
        # write empty outputs
        prev_tbl = pd.DataFrame({"icd3": [], "n_cases": [], "n_controls": []})
        inc_tbl = pd.DataFrame({"icd3": [], "n_events": [], "n_atrisk": []})
        prev_tbl.to_csv(out_prev, sep="\t", index=False)
        inc_tbl.to_csv(out_inc, sep="\t", index=False)
        log_lines.append("[WARN] No first-occurrence events after filtering; wrote empty counts.")
        return inc_tbl, prev_tbl, pd.Index([])

    # merge
    df = occ.merge(fu_valid.reset_index(), on="sample", how="left")
    df = df.dropna(subset=["baseline_date", "end_date", "first_date"])

    df["is_prev"] = df["first_date"] <= df["baseline_date"]
    df["is_inc"] = (df["first_date"] > df["baseline_date"]) & (df["first_date"] <= df["end_date"])

    prev_cases = df[df["is_prev"]].groupby("icd3")["sample"].nunique().rename("n_cases")
    inc_events = df[df["is_inc"]].groupby("icd3")["sample"].nunique().rename("n_events")

    icd3_all = pd.Index(df["icd3"].unique(), name="icd3")

    prev_tbl = prev_cases.reindex(icd3_all, fill_value=0).astype(int).reset_index()
    prev_tbl["n_controls"] = int(N_total) - prev_tbl["n_cases"]
    # clamp (shouldn't be negative)
    prev_tbl["n_controls"] = prev_tbl["n_controls"].clip(lower=0)

    inc_tbl = inc_events.reindex(icd3_all, fill_value=0).astype(int).reset_index()
    # at risk excludes prevalent at baseline
    prev_for_atrisk = prev_cases.reindex(icd3_all, fill_value=0).astype(int).reset_index()
    inc_tbl["n_atrisk"] = int(N_total) - prev_for_atrisk["n_cases"].astype(int)
    inc_tbl["n_atrisk"] = inc_tbl["n_atrisk"].clip(lower=0)

    # sort for readability
    prev_tbl = prev_tbl.sort_values(["n_cases", "icd3"], ascending=[False, True])
    inc_tbl = inc_tbl.sort_values(["n_events", "icd3"], ascending=[False, True])

    prev_tbl.to_csv(out_prev, sep="\t", index=False)
    inc_tbl.to_csv(out_inc, sep="\t", index=False)

    # QC summary
    neg_prev = int((prev_tbl["n_controls"] < 0).sum())
    neg_inc = int((inc_tbl["n_atrisk"] < 0).sum())
    log_lines.append(f"[INFO] ICD3 unique observed={len(icd3_all)}")
    log_lines.append(f"[INFO] Total prevalent cases (sum over icd3)={int(prev_tbl['n_cases'].sum())}")
    log_lines.append(f"[INFO] Total incident events (sum over icd3)={int(inc_tbl['n_events'].sum())}")
    log_lines.append(f"[QC] Negative controls rows={neg_prev}, negative at-risk rows={neg_inc}")

    return inc_tbl, prev_tbl, icd3_all


# -----------------------------
# B) Carrier QC
# -----------------------------
def load_carrier_counts(carrier_counts_path: str) -> pd.DataFrame:
    df = pd.read_csv(carrier_counts_path, sep="\t", dtype={"gene": str})
    if "gene" not in df.columns:
        die("carrier_counts TSV must contain column: gene")
    # accept either n_carriers or carrier-like name
    if "n_carriers" not in df.columns:
        # try infer
        cand = [c for c in df.columns if c.lower() in ("n_carriers", "carriers", "ncarrier", "ncarrier_count")]
        if cand:
            df = df.rename(columns={cand[0]: "n_carriers"})
        else:
            die("carrier_counts TSV must contain column: n_carriers")
    df["n_carriers"] = pd.to_numeric(df["n_carriers"], errors="coerce")
    df = df.dropna(subset=["n_carriers"])
    df["n_carriers"] = df["n_carriers"].astype(int)
    df = df.drop_duplicates(subset=["gene"], keep="first")
    return df

def maybe_load_E_matrix(npz_path: str, genes_path: str, samples_path: str, log_lines: List[str]):
    if not HAS_SCIPY:
        die("scipy not available, cannot load .npz matrix. Remove --E_npz or install scipy.")
    X = sparse.load_npz(npz_path).tocsr()
    genes = read_samples_from_list(genes_path)  # one per line
    samples = read_samples_from_list(samples_path)
    if X.shape[0] != len(samples):
        die(f"E_matrix rows ({X.shape[0]}) != len(samples.txt) ({len(samples)})")
    if X.shape[1] != len(genes):
        die(f"E_matrix cols ({X.shape[1]}) != len(genes.txt) ({len(genes)})")
    log_lines.append(f"[INFO] Loaded E_matrix: shape={X.shape} (n_samples x n_genes)")
    return X, genes, samples

def carrier_qc(carrier_counts_path: str,
               analysis_samples: List[str],
               valid_followup_samples_set: set,
               out_prefix: str,
               min_carriers: int,
               min_noncarriers: int,
               alert_carrier_rate: float,
               E_npz: Optional[str],
               E_genes: Optional[str],
               E_samples: Optional[str],
               log_lines: List[str]) -> None:
    ensure_dir(os.path.dirname(out_prefix) or ".")

    N_total = len(analysis_samples)
    N_fu = len(valid_followup_samples_set)

    cc = load_carrier_counts(carrier_counts_path)
    log_lines.append(f"[INFO] Carrier counts rows={cc.shape[0]} (unique genes)")

    # If E_matrix provided: recompute counts in followup-valid subset (preferred denominator for Step5B incident)
    cc_fu = None
    if E_npz and E_genes and E_samples:
        X, genes, samples_E = maybe_load_E_matrix(E_npz, E_genes, E_samples, log_lines)

        # Build mask for followup-valid subset
        idx_map = {s: i for i, s in enumerate(samples_E)}
        idx = [idx_map[s] for s in valid_followup_samples_set if s in idx_map]
        idx = np.array(sorted(idx), dtype=int)
        if idx.size == 0:
            log_lines.append("[WARN] No overlap between followup-valid samples and E_matrix samples; skip subset counts.")
        else:
            Xsub = X[idx, :]
            ncar_sub = np.asarray(Xsub.getnnz(axis=0)).ravel().astype(int)
            cc_fu = pd.DataFrame({"gene": genes, "n_carriers_followup": ncar_sub})
            log_lines.append(f"[INFO] Recomputed n_carriers within followup-valid subset: N={idx.size}")
            # merge into cc
            cc = cc.merge(cc_fu, on="gene", how="left")
    else:
        log_lines.append("[INFO] E_matrix .npz not provided; carrier QC will use n_carriers as-is (no followup-subset recompute).")

    # Denominator choice for QC:
    # - for Step5B incident models: followup-valid subset is the effective analysis set
    denom = N_fu if N_fu > 0 else N_total
    denom_name = "N_followup_valid" if N_fu > 0 else "N_analysis_samples"
    log_lines.append(f"[INFO] Carrier QC denominator: {denom_name}={denom}")

    # Use subset counts if available, else overall
    if "n_carriers_followup" in cc.columns and cc["n_carriers_followup"].notna().any():
        cc["n_carriers_qc"] = cc["n_carriers_followup"].fillna(cc["n_carriers"]).astype(int)
        cc["carrier_count_source"] = np.where(cc["n_carriers_followup"].notna(), "followup_subset", "overall_counts")
    else:
        cc["n_carriers_qc"] = cc["n_carriers"].astype(int)
        cc["carrier_count_source"] = "overall_counts"

    # Compute rates
    cc["n_noncarriers_qc"] = denom - cc["n_carriers_qc"]
    cc["carrier_rate_qc"] = cc["n_carriers_qc"] / float(denom)

    # Hard flags (for Step5B stability)
    cc["flag_low_carriers"] = cc["n_carriers_qc"] < int(min_carriers)
    cc["flag_low_noncarriers"] = cc["n_noncarriers_qc"] < int(min_noncarriers)
    cc["flag_high_rate_alert"] = cc["carrier_rate_qc"] > float(alert_carrier_rate)
    cc["flag_carriers_gt_denom"] = cc["n_carriers_qc"] > int(denom)

    cc["flag_any"] = (
        cc["flag_low_carriers"] |
        cc["flag_low_noncarriers"] |
        cc["flag_high_rate_alert"] |
        cc["flag_carriers_gt_denom"]
    )

    # Sort
    cc = cc.sort_values(["carrier_rate_qc", "n_carriers_qc"], ascending=[False, False])

    out_all = f"{out_prefix}.carriers_qc.tsv"
    out_flag = f"{out_prefix}.carriers_qc.flagged.tsv"
    out_excl = f"{out_prefix}.genes.exclude_step5B.txt"
    out_sum = f"{out_prefix}.carriers_qc.summary.txt"

    cc.to_csv(out_all, sep="\t", index=False)
    cc_flag = cc[cc["flag_any"]].copy()
    cc_flag.to_csv(out_flag, sep="\t", index=False)

    # Exclusion for Step5B main analysis: too few carriers OR too few non-carriers
    # (high_rate_alert is NOT auto-excluded; it is an ALERT to re-check pipeline)
    excl = cc[cc["flag_low_carriers"] | cc["flag_low_noncarriers"]]["gene"].astype(str).tolist()
    excl = sorted(set(excl))
    with open(out_excl, "w") as f:
        for g in excl:
            f.write(g + "\n")

    # Summary
    n_genes = cc.shape[0]
    n_flag = cc_flag.shape[0]
    n_excl = len(excl)
    n_alert = int(cc["flag_high_rate_alert"].sum())

    top_alert = cc[cc["flag_high_rate_alert"]].head(20)[["gene", "n_carriers_qc", "carrier_rate_qc"]]

    lines = []
    lines.append(f"[INFO] Denominator ({denom_name}) = {denom}")
    lines.append(f"[INFO] Genes total = {n_genes}")
    lines.append(f"[INFO] Flagged (any) = {n_flag}")
    lines.append(f"[INFO] Excluded for Step5B (low carriers OR low non-carriers) = {n_excl}")
    lines.append(f"[INFO] High carrier-rate ALERT (> {alert_carrier_rate:.3g}) genes = {n_alert}")
    if n_alert > 0:
        lines.append("[ALERT] Top high-rate genes (first 20):")
        for _, r in top_alert.iterrows():
            lines.append(f"  - {r['gene']}\t{int(r['n_carriers_qc'])}\t{float(r['carrier_rate_qc']):.4f}")

        lines.append("")
        lines.append("[NOTE] If many genes have carrier_rate extremely high under rare LoF/PLP (<0.1%),")
        lines.append("       re-check carrier definition (do NOT treat missing GT './.' as carrier).")

    write_text(out_sum, "\n".join(lines) + "\n")

    log_lines.append(f"[OK] Wrote carrier QC: {out_all}")
    log_lines.append(f"[OK] Wrote flagged   : {out_flag}")
    log_lines.append(f"[OK] Wrote exclude  : {out_excl}")
    log_lines.append(f"[OK] Wrote summary  : {out_sum}")


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--log", default=None, help="Log file path (default: outdir/step5_prepare_inputs.log)")

    # analysis sample definition
    ap.add_argument("--analysis_samples_tsv", default=None,
                    help="wes_olink_covariates_analytic.tsv (contains sample column)")
    ap.add_argument("--sample_col", default="sample", help="Sample column name in analysis_samples_tsv")
    ap.add_argument("--samples_list", default=None, help="samples.txt (one sample per line), alternative to analysis_samples_tsv")

    # phenotype files
    ap.add_argument("--first_occ", required=True, help="icd10_first_occurrence.tsv")
    ap.add_argument("--followup", required=True, help="followup_dates.tsv")
    ap.add_argument("--chunksize", type=int, default=2_000_000, help="Chunksize for first_occ reading (default 2e6)")

    # outputs for counts
    ap.add_argument("--out_inc_counts", default=None, help="Output inc_icd3_counts.tsv (default in outdir)")
    ap.add_argument("--out_prev_counts", default=None, help="Output prev_icd3_counts.tsv (default in outdir)")

    # carrier QC inputs
    ap.add_argument("--carrier_counts_tsv", required=True, help="E_matrix.carrier_counts.tsv (gene n_carriers)")
    ap.add_argument("--min_carriers", type=int, default=10, help="Exclude if n_carriers < this (default 10)")
    ap.add_argument("--min_noncarriers", type=int, default=200, help="Exclude if n_noncarriers < this (default 200)")
    ap.add_argument("--alert_carrier_rate", type=float, default=0.05, help="ALERT if carrier_rate > this (default 0.05)")
    ap.add_argument("--qc_prefix", default=None, help="Prefix for QC outputs (default outdir/step5_inputs)")

    # optional E_matrix to recompute counts in followup-valid subset
    ap.add_argument("--E_npz", default=None, help="E_matrix.carrier_binary.npz (optional)")
    ap.add_argument("--E_genes", default=None, help="E_matrix.genes.txt (optional)")
    ap.add_argument("--E_samples", default=None, help="E_matrix.samples.txt (optional)")

    args = ap.parse_args()

    ensure_dir(args.outdir)
    log_path = args.log or os.path.join(args.outdir, "step5_prepare_inputs.log")
    out_inc = args.out_inc_counts or os.path.join(args.outdir, "inc_icd3_counts.tsv")
    out_prev = args.out_prev_counts or os.path.join(args.outdir, "prev_icd3_counts.tsv")
    qc_prefix = args.qc_prefix or os.path.join(args.outdir, "step5_inputs")

    log_lines = []
    log_lines.append("[INFO] Step5 input preparation started.")
    log_lines.append(f"[INFO] outdir={args.outdir}")

    # Load analysis samples
    if args.analysis_samples_tsv:
        if not os.path.exists(args.analysis_samples_tsv):
            die(f"analysis_samples_tsv not found: {args.analysis_samples_tsv}")
        samples = read_samples_from_tsv(args.analysis_samples_tsv, args.sample_col)
        log_lines.append(f"[INFO] Loaded analysis samples from TSV: n={len(samples)}")
    elif args.samples_list:
        if not os.path.exists(args.samples_list):
            die(f"samples_list not found: {args.samples_list}")
        samples = read_samples_from_list(args.samples_list)
        log_lines.append(f"[INFO] Loaded analysis samples from list: n={len(samples)}")
    else:
        die("Provide either --analysis_samples_tsv or --samples_list")

    if len(samples) == 0:
        die("No analysis samples loaded (empty).")

    # Build counts
    inc_tbl, prev_tbl, icd3_all = build_icd3_counts(
        samples=samples,
        first_occ_path=args.first_occ,
        followup_path=args.followup,
        out_inc=out_inc,
        out_prev=out_prev,
        chunksize=args.chunksize,
        log_lines=log_lines
    )
    log_lines.append(f"[OK] Wrote: {out_inc}")
    log_lines.append(f"[OK] Wrote: {out_prev}")

    # Determine followup-valid sample set for QC denominator & optional subset counts
    # Re-load followup (light) to get valid sample set
    samples_set = set(samples)
    fu_valid = load_followup(args.followup, samples_set, log_lines)
    valid_fu_samples_set = set(fu_valid.index.astype(str).tolist())

    # Carrier QC
    if args.E_npz or args.E_genes or args.E_samples:
        # require all or none
        if not (args.E_npz and args.E_genes and args.E_samples):
            die("If using E_matrix for subset carrier counts, must provide --E_npz --E_genes --E_samples together.")
        for p in [args.E_npz, args.E_genes, args.E_samples]:
            if not os.path.exists(p):
                die(f"E_matrix input not found: {p}")

    carrier_qc(
        carrier_counts_path=args.carrier_counts_tsv,
        analysis_samples=samples,
        valid_followup_samples_set=valid_fu_samples_set,
        out_prefix=qc_prefix,
        min_carriers=args.min_carriers,
        min_noncarriers=args.min_noncarriers,
        alert_carrier_rate=args.alert_carrier_rate,
        E_npz=args.E_npz,
        E_genes=args.E_genes,
        E_samples=args.E_samples,
        log_lines=log_lines
    )

    log_lines.append("[INFO] Step5 input preparation finished.")

    write_text(log_path, "\n".join(log_lines) + "\n")
    print(f"[OK] Log written: {log_path}")


if __name__ == "__main__":
    main()
