#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_run_all_protein_mediation_scan.py

Purpose
-------
Run all-protein mediation scan for supported gene-disease pairs:
    E = rare deleterious gene burden
    M = each measured Olink protein
    Y = incident ICD-3 disease

This step fits point-estimate models only. It does not run bootstrap.

Model framework
---------------
1. M ~ E + covariates                          using OLS
2. Surv(time_days, event) ~ E + covariates     using Cox PH
3. Surv(time_days, event) ~ E + M + covariates using Cox PH

Indirect effect:
    indirect_logHR = beta_EM * beta_MY_conditional

Parallel strategy
-----------------
Use modulo chunking over the Step1 runset:
    row_index % n_chunks == chunk_id

For a pair-major runset, this assigns about 1/n_chunks of proteins from every
pair to each task. Each task then loads only its assigned protein columns.
"""

import argparse
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse
from statsmodels.duration.hazard_regression import PHReg


DEFAULT_COVS = ["age", "sex"] + [f"PC{i}" for i in range(1, 11)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all-protein mediation scan for encoded protein ranking."
    )

    parser.add_argument("--runset_tsv", required=True)
    parser.add_argument("--covariate_protein_table", required=True)

    parser.add_argument("--e_samples", required=True)
    parser.add_argument("--e_genes", required=True)
    parser.add_argument("--e_matrix", required=True)

    parser.add_argument("--first_occurrence", required=True)
    parser.add_argument("--followup_dates", required=True)

    parser.add_argument("--outdir", required=True)
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--n_chunks", type=int, default=50)

    parser.add_argument("--covariates", nargs="+", default=DEFAULT_COVS)

    parser.add_argument("--min_complete_case", type=int, default=200)
    parser.add_argument("--min_events_complete_case", type=int, default=20)
    parser.add_argument("--min_carriers_complete_case", type=int, default=10)
    parser.add_argument("--min_noncarriers_complete_case", type=int, default=200)
    parser.add_argument("--min_carrier_events_complete_case", type=int, default=2)
    parser.add_argument("--min_carrier_nonevents_complete_case", type=int, default=5)
    parser.add_argument("--min_M_sd", type=float, default=1e-8)

    return parser.parse_args()


def select_modulo_chunk(df: pd.DataFrame, chunk_id: int, n_chunks: int) -> pd.DataFrame:
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    if not (0 <= chunk_id < n_chunks):
        raise ValueError("chunk_id out of range")

    idx = np.arange(df.shape[0])
    return df.loc[(idx % n_chunks) == chunk_id].copy().reset_index(drop=True)


def read_vector_file(path: str) -> list:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def is_finite_scalar(x) -> bool:
    return pd.notna(x) and np.isfinite(float(x))


def require_finite_scalar(name: str, x) -> float:
    if not is_finite_scalar(x):
        raise ValueError(f"nonfinite_{name}")
    return float(x)


def safe_ci_from_beta_se(beta, se):
    if (not is_finite_scalar(beta)) or (not is_finite_scalar(se)):
        return np.nan, np.nan

    beta = float(beta)
    se = float(se)
    return beta - 1.96 * se, beta + 1.96 * se


def p_from_z(z):
    if not np.isfinite(z):
        return np.nan
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def extract_se_from_cov(fit, idx: int, name: str) -> float:
    cov = np.asarray(fit.cov_params())

    if cov.ndim != 2 or idx >= cov.shape[0] or idx >= cov.shape[1]:
        raise ValueError(f"invalid_cov_matrix_for_{name}")

    v = cov[idx, idx]

    if (not np.isfinite(v)) or v < 0:
        raise ValueError(f"nonfinite_or_negative_variance_for_{name}")

    se = float(np.sqrt(v))

    if not np.isfinite(se):
        raise ValueError(f"nonfinite_se_for_{name}")

    return se


def fit_ols_mediator(dat: pd.DataFrame, covariates: list) -> dict:
    """
    M ~ E + covariates
    """
    cols = ["E"] + covariates
    X = sm.add_constant(dat[cols], has_constant="add")
    y = dat["M"].astype(float)

    fit = sm.OLS(y, X).fit()

    beta = require_finite_scalar("beta_EM", fit.params["E"])
    se = require_finite_scalar("se_EM", fit.bse["E"])
    p = require_finite_scalar("p_EM", fit.pvalues["E"])
    ci_l, ci_u = safe_ci_from_beta_se(beta, se)

    return {
        "mediator_model_success": 1,
        "beta_EM": beta,
        "se_EM": se,
        "p_EM": p,
        "ci95_l_EM": ci_l,
        "ci95_u_EM": ci_u,
        "n_mediator_model": int(dat.shape[0]),
        "r2_mediator_model": float(fit.rsquared) if np.isfinite(fit.rsquared) else np.nan,
    }


def fit_phreg_total(dat: pd.DataFrame, covariates: list) -> dict:
    """
    Surv(time_days, event) ~ E + covariates
    """
    cols = ["E"] + covariates
    X = dat[cols]

    time = dat["time_days"].astype(float).to_numpy()
    event = dat["event"].astype(int).to_numpy()

    model = PHReg(endog=time, exog=X.to_numpy(), status=event, ties="efron")
    fit = model.fit()

    idx_e = cols.index("E")

    beta = require_finite_scalar("beta_EY_total", fit.params[idx_e])
    se = extract_se_from_cov(fit, idx_e, "beta_EY_total")
    ci_l, ci_u = safe_ci_from_beta_se(beta, se)

    return {
        "total_model_success": 1,
        "beta_EY_total": beta,
        "se_EY_total": se,
        "p_EY_total": p_from_z(beta / se),
        "hr_EY_total": float(np.exp(beta)),
        "ci95_l_EY_total": float(np.exp(ci_l)),
        "ci95_u_EY_total": float(np.exp(ci_u)),
        "n_total_model": int(dat.shape[0]),
    }


def fit_phreg_direct(dat: pd.DataFrame, covariates: list) -> dict:
    """
    Surv(time_days, event) ~ E + M + covariates
    """
    cols = ["E", "M"] + covariates
    X = dat[cols]

    time = dat["time_days"].astype(float).to_numpy()
    event = dat["event"].astype(int).to_numpy()

    model = PHReg(endog=time, exog=X.to_numpy(), status=event, ties="efron")
    fit = model.fit()

    idx_e = cols.index("E")
    idx_m = cols.index("M")

    beta_e = require_finite_scalar("beta_EY_direct", fit.params[idx_e])
    se_e = extract_se_from_cov(fit, idx_e, "beta_EY_direct")
    ci_l_e, ci_u_e = safe_ci_from_beta_se(beta_e, se_e)

    beta_m = require_finite_scalar("beta_MY_conditional", fit.params[idx_m])
    se_m = extract_se_from_cov(fit, idx_m, "beta_MY_conditional")
    ci_l_m, ci_u_m = safe_ci_from_beta_se(beta_m, se_m)

    return {
        "direct_model_success": 1,

        "beta_EY_direct": beta_e,
        "se_EY_direct": se_e,
        "p_EY_direct": p_from_z(beta_e / se_e),
        "hr_EY_direct": float(np.exp(beta_e)),
        "ci95_l_EY_direct": float(np.exp(ci_l_e)),
        "ci95_u_EY_direct": float(np.exp(ci_u_e)),

        "beta_MY_conditional": beta_m,
        "se_MY_conditional": se_m,
        "p_MY_conditional": p_from_z(beta_m / se_m),
        "hr_MY_conditional": float(np.exp(beta_m)),
        "ci95_l_MY_conditional": float(np.exp(ci_l_m)),
        "ci95_u_MY_conditional": float(np.exp(ci_u_m)),

        "n_direct_model": int(dat.shape[0]),
    }


def compute_indirect(med_res: dict, direct_res: dict, total_res: dict) -> dict:
    a = med_res["beta_EM"]
    se_a = med_res["se_EM"]

    b = direct_res["beta_MY_conditional"]
    se_b = direct_res["se_MY_conditional"]

    indirect = a * b
    indirect_hr = float(np.exp(indirect))

    # Delta/Sobel approximation for full-panel ranking.
    # Bootstrap validation is handled later for selected mediators.
    var_indirect = (b ** 2) * (se_a ** 2) + (a ** 2) * (se_b ** 2)
    se_indirect = (
        float(np.sqrt(var_indirect))
        if np.isfinite(var_indirect) and var_indirect >= 0
        else np.nan
    )

    z_indirect = (
        indirect / se_indirect
        if np.isfinite(se_indirect) and se_indirect > 0
        else np.nan
    )

    p_indirect = p_from_z(z_indirect)

    beta_total = total_res.get("beta_EY_total", np.nan)

    if np.isfinite(beta_total) and abs(beta_total) > 1e-12:
        pm = indirect / beta_total
    else:
        pm = np.nan

    return {
        "indirect_logHR": float(indirect),
        "indirect_hr": indirect_hr,
        "se_indirect_logHR_analytic": se_indirect,
        "z_indirect_analytic": z_indirect,
        "indirect_p_analytic": p_indirect,
        "proportion_mediated_logscale": float(pm) if np.isfinite(pm) else np.nan,
    }


def load_covariate_protein_table(path: str, covariates: list, protein_cols: list) -> pd.DataFrame:
    usecols = ["sample"] + covariates + protein_cols

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        dtype={"sample": str},
    )

    df["sample"] = df["sample"].astype(str)

    for col in covariates + protein_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.set_index("sample", drop=True)


def load_e_matrix(samples_path: str, genes_path: str, matrix_path: str, needed_genes: list):
    e_samples = read_vector_file(samples_path)
    e_genes = read_vector_file(genes_path)

    gene_to_idx = {g: i for i, g in enumerate(e_genes)}

    missing_genes = sorted(set(needed_genes) - set(gene_to_idx))
    if missing_genes:
        raise ValueError("Genes missing from E matrix: " + ",".join(missing_genes))

    mat = sparse.load_npz(matrix_path).tocsr()

    if mat.shape[0] != len(e_samples) or mat.shape[1] != len(e_genes):
        raise ValueError("E matrix dimensions do not match sample/gene files")

    e_by_gene = {}

    for gene in sorted(set(needed_genes)):
        j = gene_to_idx[gene]
        e_by_gene[gene] = pd.Series(
            np.asarray(mat[:, j].toarray()).ravel(),
            index=pd.Index(e_samples, name="sample"),
            name="E",
        )

    return e_by_gene


def prepare_followup(path: str) -> pd.DataFrame:
    follow = pd.read_csv(path, sep="\t", dtype={"sample": str})

    required = {"sample", "baseline_date", "death_date", "censor_date"}
    missing = required - set(follow.columns)

    if missing:
        raise ValueError("followup_dates missing columns: " + ",".join(sorted(missing)))

    follow["sample"] = follow["sample"].astype(str)
    follow["baseline_date"] = pd.to_datetime(follow["baseline_date"], errors="coerce")
    follow["death_date"] = pd.to_datetime(follow["death_date"], errors="coerce")
    follow["censor_date"] = pd.to_datetime(follow["censor_date"], errors="coerce")

    follow["end_date"] = follow[["death_date", "censor_date"]].min(axis=1, skipna=True)

    follow = follow.set_index("sample", drop=True)

    return follow[["baseline_date", "end_date"]]


def load_first_occurrence(path: str, needed_icd3: list) -> pd.DataFrame:
    occ = pd.read_csv(path, sep="\t", dtype={"sample": str, "icd3": str})

    required = {"sample", "icd3", "first_date"}
    missing = required - set(occ.columns)

    if missing:
        raise ValueError("icd10_first_occurrence missing columns: " + ",".join(sorted(missing)))

    occ = occ.loc[
        occ["icd3"].isin(set(needed_icd3)),
        ["sample", "icd3", "first_date"],
    ].copy()

    occ["sample"] = occ["sample"].astype(str)
    occ["icd3"] = occ["icd3"].astype(str)
    occ["first_date"] = pd.to_datetime(occ["first_date"], errors="coerce")

    occ = occ.dropna(subset=["first_date"])

    # If duplicated records exist, use the earliest occurrence.
    occ = (
        occ.groupby(["icd3", "sample"], as_index=False)["first_date"]
        .min()
    )

    return occ


def build_incident_phenotype(
    icd3: str,
    samples: pd.Index,
    follow: pd.DataFrame,
    occ: pd.DataFrame,
) -> pd.DataFrame:
    base = follow.reindex(samples).copy()

    sub = occ.loc[occ["icd3"] == icd3, ["sample", "first_date"]].copy()

    if not sub.empty:
        first = sub.set_index("sample")["first_date"]
    else:
        first = pd.Series(dtype="datetime64[ns]")

    base["first_date"] = first.reindex(samples)

    baseline = base["baseline_date"]
    end = base["end_date"]
    first_date = base["first_date"]

    eligible = (
        baseline.notna()
        & end.notna()
        & ((first_date.isna()) | (first_date > baseline))
    )

    event = (
        eligible
        & first_date.notna()
        & (first_date > baseline)
        & (first_date <= end)
    )

    time_days = pd.Series(np.nan, index=samples, dtype="float64")

    time_days.loc[event] = (
        first_date.loc[event] - baseline.loc[event]
    ).dt.days.astype(float)

    censored = eligible & (~event)

    time_days.loc[censored] = (
        end.loc[censored] - baseline.loc[censored]
    ).dt.days.astype(float)

    return pd.DataFrame(
        {
            "eligible_for_incident": eligible.astype(int),
            "event": event.astype(int),
            "time_days": time_days,
        },
        index=samples,
    )


def precheck_complete_case(dat: pd.DataFrame, covariates: list, args) -> tuple:
    needed = ["E", "M", "event", "time_days", "eligible_for_incident"] + covariates

    cc_mask = (
        dat["eligible_for_incident"].eq(1)
        & dat["E"].notna()
        & dat["M"].notna()
        & dat["event"].isin([0, 1])
        & dat["time_days"].notna()
        & (dat["time_days"] > 0)
        & dat[covariates].notna().all(axis=1)
    )

    cc = dat.loc[cc_mask, needed].copy()

    n_total_rows = int(dat.shape[0])
    n_complete_case = int(cc.shape[0])

    out = {
        "n_total_rows": n_total_rows,
        "n_complete_case": n_complete_case,
    }

    if n_complete_case == 0:
        out.update(
            {
                "n_events_complete_case": 0,
                "n_carriers_complete_case": 0,
                "n_noncarriers_complete_case": 0,
                "n_carrier_events_complete_case": 0,
                "n_carrier_nonevents_complete_case": 0,
                "M_sd_complete_case": np.nan,
            }
        )
        return cc, out, ["no_complete_case_rows"]

    e = cc["E"].to_numpy()
    y = cc["event"].astype(int).to_numpy()

    carrier_mask = e > 0
    noncarrier_mask = ~carrier_mask
    event_mask = y == 1
    nonevent_mask = y == 0

    n_events = int(event_mask.sum())
    n_carriers = int(carrier_mask.sum())
    n_noncarriers = int(noncarrier_mask.sum())
    n_carrier_events = int((carrier_mask & event_mask).sum())
    n_carrier_nonevents = int((carrier_mask & nonevent_mask).sum())

    m_sd = (
        float(cc["M"].std(skipna=True))
        if cc["M"].notna().sum() > 1
        else np.nan
    )

    out.update(
        {
            "n_events_complete_case": n_events,
            "n_carriers_complete_case": n_carriers,
            "n_noncarriers_complete_case": n_noncarriers,
            "n_carrier_events_complete_case": n_carrier_events,
            "n_carrier_nonevents_complete_case": n_carrier_nonevents,
            "M_sd_complete_case": m_sd,
        }
    )

    fail = []

    if n_complete_case < args.min_complete_case:
        fail.append("low_complete_case")
    if n_events < args.min_events_complete_case:
        fail.append("low_events_complete_case")
    if n_carriers < args.min_carriers_complete_case:
        fail.append("low_carriers_complete_case")
    if n_noncarriers < args.min_noncarriers_complete_case:
        fail.append("low_noncarriers_complete_case")
    if n_carrier_events < args.min_carrier_events_complete_case:
        fail.append("low_carrier_events_complete_case")
    if n_carrier_nonevents < args.min_carrier_nonevents_complete_case:
        fail.append("low_carrier_nonevents_complete_case")
    if (not np.isfinite(m_sd)) or (m_sd <= args.min_M_sd):
        fail.append("low_M_variation_complete_case")
    if np.nanstd(cc["E"].to_numpy()) <= 0:
        fail.append("no_E_variation_complete_case")

    return cc, out, fail


def empty_result_row(row: pd.Series) -> dict:
    return {
        "pair_id": row.get("pair_id", ""),
        "gene": row.get("gene", ""),
        "icd3": row.get("icd3", ""),
        "encoded_protein": row.get("encoded_protein", ""),
        "mediator_protein": row.get("mediator_protein", ""),
        "is_encoded_protein": row.get("is_encoded_protein", np.nan),
        "tier_label": row.get("tier_label", ""),
        "original_trio_id": row.get("original_trio_id", ""),

        "mediator_model_success": 0,
        "total_model_success": 0,
        "direct_model_success": 0,
        "model_success": 0,
        "fail_reason": "",
    }


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_main = outdir / f"02_all_protein_mediation_scan.chunk{args.chunk_id}.tsv.gz"
    out_qc = outdir / f"02_all_protein_mediation_scan.chunk{args.chunk_id}.qc.tsv"

    runset_all = pd.read_csv(
        args.runset_tsv,
        sep="\t",
        compression="infer",
        dtype=str,
    )

    required = {
        "pair_id",
        "gene",
        "icd3",
        "encoded_protein",
        "mediator_protein",
        "is_encoded_protein",
    }

    missing = required - set(runset_all.columns)

    if missing:
        raise ValueError("runset_tsv missing columns: " + ",".join(sorted(missing)))

    runset = select_modulo_chunk(runset_all, args.chunk_id, args.n_chunks)

    if runset.empty:
        pd.DataFrame().to_csv(out_main, sep="\t", index=False, compression="gzip")
        pd.DataFrame(
            [{"metric": "n_rows_assigned", "value": 0}]
        ).to_csv(out_qc, sep="\t", index=False)

        print(f"[INFO] no rows assigned to chunk {args.chunk_id}")
        return

    runset["is_encoded_protein"] = pd.to_numeric(
        runset["is_encoded_protein"],
        errors="coerce",
    )

    needed_genes = sorted(runset["gene"].dropna().astype(str).unique().tolist())
    needed_icd3 = sorted(runset["icd3"].dropna().astype(str).unique().tolist())
    needed_proteins = sorted(runset["mediator_protein"].dropna().astype(str).unique().tolist())

    data = load_covariate_protein_table(
        path=args.covariate_protein_table,
        covariates=args.covariates,
        protein_cols=needed_proteins,
    )

    samples = data.index.astype(str)

    e_by_gene = load_e_matrix(
        args.e_samples,
        args.e_genes,
        args.e_matrix,
        needed_genes,
    )

    follow = prepare_followup(args.followup_dates)
    occ = load_first_occurrence(args.first_occurrence, needed_icd3)

    phenotype_cache = {}
    total_cache = {}
    results = []

    for pair_id, pair_rows in runset.groupby("pair_id", sort=False):
        gene = str(pair_rows["gene"].iloc[0])
        icd3 = str(pair_rows["icd3"].iloc[0])

        e_series = e_by_gene[gene].reindex(samples)

        if icd3 not in phenotype_cache:
            phenotype_cache[icd3] = build_incident_phenotype(
                icd3,
                samples,
                follow,
                occ,
            )

        pheno = phenotype_cache[icd3]

        pair_base = data[args.covariates].copy()
        pair_base["E"] = pd.to_numeric(e_series, errors="coerce")
        pair_base["event"] = pheno["event"]
        pair_base["time_days"] = pheno["time_days"]
        pair_base["eligible_for_incident"] = pheno["eligible_for_incident"]

        total_key = f"{gene}__{icd3}"

        for _, r in pair_rows.iterrows():
            mediator = str(r["mediator_protein"])
            row_out = empty_result_row(r)

            dat = pair_base.copy()
            dat["M"] = data[mediator]

            cc, counts, fail_reasons = precheck_complete_case(
                dat,
                args.covariates,
                args,
            )

            row_out.update(counts)

            if fail_reasons:
                row_out["fail_reason"] = ";".join(fail_reasons)
                results.append(row_out)
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if total_key not in total_cache:
                        try:
                            total_cache[total_key] = fit_phreg_total(
                                cc,
                                args.covariates,
                            )
                        except Exception as e:
                            total_cache[total_key] = {
                                "fail_reason": f"total_model_failed::{str(e)[:200]}"
                            }

                    total_res = total_cache[total_key]

                    if "fail_reason" in total_res:
                        raise ValueError(total_res["fail_reason"])

                    med_res = fit_ols_mediator(cc, args.covariates)
                    direct_res = fit_phreg_direct(cc, args.covariates)
                    indirect_res = compute_indirect(med_res, direct_res, total_res)

                row_out.update(med_res)
                row_out.update(total_res)
                row_out.update(direct_res)
                row_out.update(indirect_res)

                row_out["model_success"] = int(
                    row_out.get("mediator_model_success", 0) == 1
                    and row_out.get("total_model_success", 0) == 1
                    and row_out.get("direct_model_success", 0) == 1
                    and np.isfinite(row_out.get("indirect_logHR", np.nan))
                    and np.isfinite(row_out.get("indirect_hr", np.nan))
                )

                if row_out["model_success"] != 1:
                    row_out["fail_reason"] = "model_incomplete_or_nonfinite_indirect"

            except Exception as e:
                row_out["fail_reason"] = f"model_failed::{str(e)[:300]}"

            results.append(row_out)

        print(
            f"[INFO] chunk {args.chunk_id}: finished pair {pair_id}, rows={pair_rows.shape[0]}",
            flush=True,
        )

    res = pd.DataFrame(results)

    if not res.empty:
        res = res.sort_values(["pair_id", "mediator_protein"]).copy()

    res.to_csv(out_main, sep="\t", index=False, compression="gzip")

    qc_rows = [
        {"metric": "chunk_id", "value": args.chunk_id},
        {"metric": "n_chunks", "value": args.n_chunks},
        {"metric": "n_rows_assigned", "value": int(runset.shape[0])},
        {"metric": "n_unique_pairs", "value": int(runset["pair_id"].nunique())},
        {"metric": "n_unique_genes", "value": int(len(needed_genes))},
        {"metric": "n_unique_icd3", "value": int(len(needed_icd3))},
        {"metric": "n_unique_mediator_proteins", "value": int(len(needed_proteins))},
        {"metric": "n_result_rows", "value": int(res.shape[0])},
        {
            "metric": "n_model_success",
            "value": int((res["model_success"] == 1).sum()) if not res.empty else 0,
        },
        {
            "metric": "n_model_failed",
            "value": int((res["model_success"] != 1).sum()) if not res.empty else 0,
        },
        {
            "metric": "n_encoded_rows",
            "value": int(
                (
                    pd.to_numeric(
                        res.get("is_encoded_protein", pd.Series(dtype=float)),
                        errors="coerce",
                    )
                    == 1
                ).sum()
            )
            if not res.empty
            else 0,
        },
    ]

    if not res.empty and "fail_reason" in res.columns:
        fail_counts = (
            res.loc[res["model_success"] != 1, "fail_reason"]
            .astype(str)
            .value_counts()
        )

        for reason, n in fail_counts.items():
            if reason.strip():
                qc_rows.append({"metric": f"fail::{reason}", "value": int(n)})

    pd.DataFrame(qc_rows).to_csv(out_qc, sep="\t", index=False)

    print(f"[OK] results written to: {out_main}")
    print(f"[OK] qc written to: {out_qc}")
    print(
        f"[INFO] n_rows={res.shape[0]}, "
        f"n_model_success={(res['model_success'] == 1).sum() if not res.empty else 0}"
    )


if __name__ == "__main__":
    main()
