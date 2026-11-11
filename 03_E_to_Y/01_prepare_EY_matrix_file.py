#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build unified E→Y gene burden / carrier matrix, but restrict genes to those with measured proteins.

Rationale:
  - We only want E signals that can, in principle, be tested for cis E→M (gene→protein),
    so we restrict E genes to those with measured protein columns in wes_olink_covariates_analytic.tsv.

Inputs
  --carriers_tsv_template : chr{CHR}.gene_carriers.filtered.tsv
      columns: gene, sample, carrier (typically 1)
  --analysis_samples_tsv  : wes_olink_covariates_analytic.tsv
      columns: sample, age, sex, PC1..PC10, <protein columns ...>

Key restriction:
  allowed_genes = all columns in analysis_samples_tsv EXCEPT {sample_col} U {cov_cols}
  (i.e., protein measurement columns)

Outputs (outdir)
  E_matrix.burden_counts.npz         (CSR, int32, counts per sample×gene; duplicates summed)
  E_matrix.carrier_binary.npz        (CSR, int8, 0/1)
  E_matrix.genes.txt
  E_matrix.samples.txt
  E_matrix.carrier_counts.tsv        (gene, n_carriers, carrier_rate, n_noncarriers)
  genes.exclude_step5B.txt           (optional, based on QC thresholds)
  build_E_matrix_for_EY.log

Notes
  - If you still see many genes with carrier_rate ~1 under rare LoF/PLP (<0.1%),
    the carrier-calling logic or AF/source filtering is likely wrong.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz


CHROMS_DEFAULT = [str(i) for i in range(1, 23)] + ["X", "Y"]
COV_COLS_DEFAULT = ["age", "sex"] + [f"PC{i}" for i in range(1, 11)]


def log(msg: str, fp):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fp.write(f"[{ts}] {msg}\n")
    fp.flush()


def read_samples_from_cov(path: str, sample_col: str) -> list[str]:
    df = pd.read_csv(path, sep="\t", usecols=[sample_col], dtype={sample_col: str})
    seen = set()
    out = []
    for s in df[sample_col].astype(str).tolist():
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def read_allowed_genes_from_cov_header(path: str, sample_col: str, cov_cols: list[str]) -> set[str]:
    header = list(pd.read_csv(path, sep="\t", nrows=0).columns)
    colset = set(header)

    required = {sample_col} | set(cov_cols)
    missing = [c for c in required if c not in colset]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    allowed = [c for c in header if c not in required]
    # keep as-is (protein columns may contain underscores/dashes, etc.)
    return set(allowed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carriers_tsv_template", required=True,
                    help="Template: /path/chr{CHR}.gene_carriers.filtered.tsv")
    ap.add_argument("--analysis_samples_tsv", required=True,
                    help="wes_olink_covariates_analytic.tsv (contains sample + covariates + proteins)")
    ap.add_argument("--sample_col", default="sample")
    ap.add_argument("--cov_cols", nargs="+", default=COV_COLS_DEFAULT,
                    help="Covariate columns in analysis_samples_tsv (default: age sex PC1..PC10)")
    ap.add_argument("--chroms", nargs="+", default=CHROMS_DEFAULT,
                    help="Chromosomes to include (default: 1-22 X Y)")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    ap.add_argument("--outdir", required=True)

    # QC thresholds (optional)
    ap.add_argument("--min_carriers", type=int, default=10)
    ap.add_argument("--min_noncarriers", type=int, default=200)
    ap.add_argument("--max_carrier_rate", type=float, default=0.05,
                    help="Exclude if carrier_rate > this (default 0.05). Set 1 to disable.")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    log_path = os.path.join(args.outdir, "build_E_matrix_for_EY.log")
    with open(log_path, "w") as lf:
        log(f"outdir={args.outdir}", lf)
        log(f"analysis_samples_tsv={args.analysis_samples_tsv}", lf)
        log(f"carriers_tsv_template={args.carriers_tsv_template}", lf)
        log(f"chroms={args.chroms}", lf)
        log(f"sample_col={args.sample_col}", lf)
        log(f"cov_cols={args.cov_cols}", lf)

        # 1) samples
        samples = read_samples_from_cov(args.analysis_samples_tsv, args.sample_col)
        n_samples = len(samples)
        if n_samples == 0:
            raise RuntimeError("No samples loaded from analysis_samples_tsv.")
        sample_index = {s: i for i, s in enumerate(samples)}
        log(f"Loaded samples: n={n_samples}", lf)

        # 2) allowed genes = measured proteins columns
        allowed_genes = read_allowed_genes_from_cov_header(
            args.analysis_samples_tsv, args.sample_col, args.cov_cols
        )
        log(f"Allowed genes (measured proteins columns): n={len(allowed_genes)}", lf)
        if len(allowed_genes) == 0:
            raise RuntimeError("No allowed genes (protein columns) found after removing sample/covariates.")

        # 3) build COO pieces (only allowed genes)
        gene_index: dict[str, int] = {}
        genes: list[str] = []
        rows_list = []
        cols_list = []
        data_list = []

        total_rows_read = 0
        total_rows_kept = 0
        total_rows_gene_filtered = 0

        usecols = ["gene", "sample", "carrier"]

        for CHR in args.chroms:
            path = args.carriers_tsv_template.format(CHR=CHR)
            if not os.path.exists(path):
                log(f"[WARN] missing carriers file: {path} (skip)", lf)
                continue

            log(f"Reading {path}", lf)

            for chunk in pd.read_csv(
                path, sep="\t", usecols=usecols,
                dtype={"gene": str, "sample": str},
                chunksize=args.chunksize
            ):
                total_rows_read += chunk.shape[0]
                if chunk.empty:
                    continue

                # keep carrier != 0
                chunk = chunk[chunk["carrier"].astype(str) != "0"]
                if chunk.empty:
                    continue

                # restrict to allowed genes (measured proteins)
                before = chunk.shape[0]
                chunk = chunk[chunk["gene"].isin(allowed_genes)]
                total_rows_gene_filtered += before - chunk.shape[0]
                if chunk.empty:
                    continue

                # map sample to row; drop non-overlap
                r = chunk["sample"].map(sample_index)
                m = r.notna()
                if not m.any():
                    continue
                chunk = chunk.loc[m, :]
                r = r.loc[m].astype(np.int32).to_numpy()

                # assign gene indices
                gvals = chunk["gene"].astype(str).to_numpy()
                c = np.empty(gvals.shape[0], dtype=np.int32)
                for i, g in enumerate(gvals):
                    idx = gene_index.get(g)
                    if idx is None:
                        idx = len(genes)
                        gene_index[g] = idx
                        genes.append(g)
                    c[i] = idx

                rows_list.append(r)
                cols_list.append(c)
                data_list.append(np.ones_like(r, dtype=np.int16))  # burden counts via sum_duplicates
                total_rows_kept += r.shape[0]

        log(f"Total rows read={total_rows_read}", lf)
        log(f"Rows removed by gene restriction (not measured)={total_rows_gene_filtered}", lf)
        log(f"Rows kept(after carrier+gene+sample filter)={total_rows_kept}", lf)
        log(f"Unique genes kept (measured & observed in carriers)={len(genes)}", lf)

        if len(genes) == 0 or total_rows_kept == 0:
            raise RuntimeError(
                "No carrier records kept after restricting to measured proteins. "
                "Check whether gene symbols in carriers match protein column names."
            )

        # 4) build matrices
        rows = np.concatenate(rows_list)
        cols = np.concatenate(cols_list)
        data = np.concatenate(data_list)

        X = sparse.coo_matrix((data, (rows, cols)), shape=(n_samples, len(genes)), dtype=np.int32)
        X.sum_duplicates()
        X_burden = X.tocsr()

        X_bin = X_burden.copy()
        X_bin.data[:] = 1
        X_bin = X_bin.astype(np.int8)

        # 5) carrier counts + QC
        n_carriers = np.asarray(X_bin.getnnz(axis=0)).ravel().astype(int)
        carrier_rate = n_carriers / float(n_samples)
        n_noncarriers = n_samples - n_carriers

        df_cc = pd.DataFrame({
            "gene": genes,
            "n_carriers": n_carriers,
            "carrier_rate": carrier_rate,
            "n_noncarriers": n_noncarriers
        }).sort_values(["carrier_rate", "n_carriers"], ascending=[False, False])

        # alerts
        if args.max_carrier_rate < 1:
            top_alert = df_cc[df_cc["carrier_rate"] > args.max_carrier_rate].head(20)
            if not top_alert.empty:
                log(f"[ALERT] Found genes with carrier_rate > {args.max_carrier_rate}. Top20:", lf)
                for _, r in top_alert.iterrows():
                    log(f"  - {r['gene']}\t{int(r['n_carriers'])}\t{float(r['carrier_rate']):.4f}", lf)
                log("[NOTE] Under rare LoF/PLP (<0.1%), very high carrier_rate usually indicates a carrier-calling bug or AF/source mismatch.", lf)

        excl = df_cc[
            (df_cc["n_carriers"] < args.min_carriers) |
            (df_cc["n_noncarriers"] < args.min_noncarriers) |
            ((df_cc["carrier_rate"] > args.max_carrier_rate) if args.max_carrier_rate < 1 else False)
        ]["gene"].astype(str).tolist()
        excl = sorted(set(excl))

        # 6) save
        out_burden = os.path.join(args.outdir, "E_matrix.burden_counts.npz")
        out_bin = os.path.join(args.outdir, "E_matrix.carrier_binary.npz")
        out_genes = os.path.join(args.outdir, "E_matrix.genes.txt")
        out_samples = os.path.join(args.outdir, "E_matrix.samples.txt")
        out_counts = os.path.join(args.outdir, "E_matrix.carrier_counts.tsv")
        out_excl = os.path.join(args.outdir, "genes.exclude_step5B.txt")

        save_npz(out_burden, X_burden)
        save_npz(out_bin, X_bin)

        with open(out_genes, "w") as f:
            for g in genes:
                f.write(g + "\n")
        with open(out_samples, "w") as f:
            for s in samples:
                f.write(s + "\n")

        df_cc.to_csv(out_counts, sep="\t", index=False)
        with open(out_excl, "w") as f:
            for g in excl:
                f.write(g + "\n")

        log(f"[OK] Saved burden matrix: {out_burden}", lf)
        log(f"[OK] Saved binary  matrix: {out_bin}", lf)
        log(f"[OK] Saved genes/samples: {out_genes}, {out_samples}", lf)
        log(f"[OK] Saved carrier counts: {out_counts}", lf)
        log(f"[OK] Saved exclude list  : {out_excl} (n={len(excl)})", lf)

    print(f"[OK] Done. Log: {log_path}")


if __name__ == "__main__":
    main()

