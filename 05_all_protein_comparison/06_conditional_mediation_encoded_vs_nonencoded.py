#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_conditional_mediation_encoded_vs_nonencoded.py

Purpose
-------
Conditional mediation analysis for encoded protein and best non-encoded protein.

For each gene-disease pair:
    E = rare deleterious gene burden
    M1 = encoded protein
    M2 = best non-encoded protein
    Y = incident ICD-3 disease

Models
------
1. M_encoded ~ E + covariates
2. M_nonencoded ~ E + covariates
3. Surv(time_days, event) ~ E + M_encoded + M_nonencoded + covariates

Conditional indirect effects:
    encoded_indirect_conditional = beta_EM_encoded * beta_MY_encoded_conditional
    nonencoded_indirect_conditional = beta_EM_nonencoded * beta_MY_nonencoded_conditional

Inputs
------
1. 05_encoded_nonencoded_correlation.tsv
2. wes_olink_covariates_analytic.tsv
3. E_matrix.samples.txt
4. E_matrix.genes.txt
5. E_matrix.carrier_binary.npz
6. icd10_first_occurrence.tsv
7. followup_dates.tsv

Outputs
-------
1. 06_conditional_mediation_encoded_vs_nonencoded.tsv
2. 06_conditional_mediation_overall_summary.tsv
3. 06_conditional_mediation_by_priority_class.tsv
4. 06_figure_conditional_mediation_class_summary.tsv
"""

import argparse
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse
from statsmodels.duration.hazard_regression import PHReg


DEFAULT_COVARIATES = ["age", "sex"] + [f"PC{i}" for i in range(1, 11)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run conditional mediation analysis for encoded and best non-encoded proteins."
    )

    parser.add_argument("--conditional_runset", required=True)
    parser.add_argument("--covariate_protein_table", required=True)

    parser.add_argument("--e_samples", required=True)
    parser.add_argument("--e_genes", required=True)
    parser.add_argument("--e_matrix", required=True)

    parser.add_argument("--first_occurrence", required=True)
    parser.add_argument("--followup_dates", required=True)

    parser.add_argument("--outdir", required=True)
    parser.add_argument("--prefix", default="06")

    parser.add_argument("--covariates", nargs="+", default=DEFAULT_COVARIATES)

    parser.add_argument("--min_complete_case", type=int, default=200)
    parser.add_argument("--min_events_complete_case", type=int, default=20)
    parser.add_argument("--min_carriers_complete_case", type=int, default=10)
    parser.add_argument("--min_noncarriers_complete_case", type=int, default=200)
    parser.add_argument("--min_carrier_events_complete_case", type=int, default=2)
    parser.add_argument("--min_carrier_nonevents_complete_case", type=int, default=5)
    parser.add_argument("--min_M_sd", type=float, default=1e-8)

    parser.add_argument("--retained_p_cutoff", type=float, default=0.05)

    return parser.parse_args()


def read_vector_file(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def p_from_z(z):
    if not np.isfinite(z):
        return np.nan
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def bh_fdr(pvals):
    p = pd.to_numeric(pvals, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype="float64")

    valid = p.notna() & np.isfinite(p) & (p >= 0) & (p <= 1)

    if valid.sum() == 0:
        return out

    pv = p.loc[valid]
    order = np.argsort(pv.values)
    ranked = pv.values[order]
    n = len(ranked)

    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out.loc[pv.index[order]] = q

    return out


def safe_ci(beta, se):
    if not np.isfinite(beta) or not np.isfinite(se):
        return np.nan, np.nan
    return beta - 1.96 * se, beta + 1.96 * se


def extract_se_from_cov(fit, idx, name):
    cov = np.asarray(fit.cov_params())

    if cov.ndim != 2 or idx >= cov.shape[0] or idx >= cov.shape[1]:
        raise ValueError(f"invalid_cov_matrix_for_{name}")

    v = cov[idx, idx]

    if not np.isfinite(v) or v < 0:
        raise ValueError(f"invalid_variance_for_{name}")

    return float(np.sqrt(v))


def load_runset(path):
    df = pd.read_csv(path, sep="\t", dtype=str)

    required = [
        "pair_id",
        "gene",
        "icd3",
        "encoded_protein",
        "nonencoded_protein",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError("Missing required columns in conditional runset: " + ", ".join(missing))

    df = df.drop_duplicates(
        subset=["gene", "icd3", "encoded_protein", "nonencoded_protein"]
    ).copy()

    return df


def load_covariate_protein_table(path, runset, covariates):
    proteins = sorted(
        set(runset["encoded_protein"].astype(str))
        | set(runset["nonencoded_protein"].astype(str))
    )

    usecols = ["sample"] + covariates + proteins

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        dtype={"sample": str},
    )

    df["sample"] = df["sample"].astype(str)

    for col in covariates + proteins:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.set_index("sample", drop=True)


def load_e_matrix(samples_path, genes_path, matrix_path, needed_genes):
    e_samples = read_vector_file(samples_path)
    e_genes = read_vector_file(genes_path)

    gene_to_idx = {g: i for i, g in enumerate(e_genes)}

    missing = sorted(set(needed_genes) - set(gene_to_idx))
    if missing:
        raise ValueError("Genes missing from E matrix: " + ",".join(missing))

    mat = sparse.load_npz(matrix_path).tocsr()

    if mat.shape[0] != len(e_samples) or mat.shape[1] != len(e_genes):
        raise ValueError("E matrix dimensions do not match sample/gene files.")

    e_by_gene = {}

    for gene in sorted(set(needed_genes)):
        j = gene_to_idx[gene]
        e_by_gene[gene] = pd.Series(
            np.asarray(mat[:, j].toarray()).ravel(),
            index=pd.Index(e_samples, name="sample"),
            name="E",
        )

    return e_by_gene


def prepare_followup(path):
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

    return follow.set_index("sample", drop=True)[["baseline_date", "end_date"]]


def load_first_occurrence(path, needed_icd3):
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

    occ = (
        occ.groupby(["icd3", "sample"], as_index=False)["first_date"]
        .min()
    )

    return occ


def build_incident_phenotype(icd3, samples, follow, occ):
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


def fit_ols(dat, mediator_col, covariates, label):
    cols = ["E"] + covariates
    X = sm.add_constant(dat[cols], has_constant="add")
    y = dat[mediator_col].astype(float)

    fit = sm.OLS(y, X).fit()

    beta = float(fit.params["E"])
    se = float(fit.bse["E"])
    p = float(fit.pvalues["E"])
    ci_l, ci_u = safe_ci(beta, se)

    return {
        f"beta_EM_{label}": beta,
    }


def fit_ols_em(dat, mediator_col, covariates, label):
    cols = ["E"] + covariates
    X = sm.add_constant(dat[cols], has_constant="add")
    y = dat[mediator_col].astype(float)

    fit = sm.OLS(y, X).fit()

    beta = float(fit.params["E"])
    se = float(fit.bse["E"])
    p = float(fit.pvalues["E"])
    ci_l, ci_u = safe_ci(beta, se)

    if not np.isfinite(beta) or not np.isfinite(se):
        raise ValueError(f"nonfinite_EM_{label}")

    return {
        f"beta_EM_{label}": beta,
        f"se_EM_{label}": se,
        f"p_EM_{label}": p,
        f"ci95_l_EM_{label}": ci_l,
        f"ci95_u_EM_{label}": ci_u,
        f"n_mediator_model_{label}": int(dat.shape[0]),
    }


def fit_conditional_cox(dat, covariates):
    cols = ["E", "M_encoded", "M_nonencoded"] + covariates
    X = dat[cols]

    time = dat["time_days"].astype(float).to_numpy()
    event = dat["event"].astype(int).to_numpy()

    model = PHReg(endog=time, exog=X.to_numpy(), status=event, ties="efron")
    fit = model.fit()

    idx_e = cols.index("E")
    idx_encoded = cols.index("M_encoded")
    idx_nonencoded = cols.index("M_nonencoded")

    beta_e = float(fit.params[idx_e])
    se_e = extract_se_from_cov(fit, idx_e, "beta_E_conditional")
    ci_l_e, ci_u_e = safe_ci(beta_e, se_e)

    beta_encoded = float(fit.params[idx_encoded])
    se_encoded = extract_se_from_cov(fit, idx_encoded, "beta_MY_encoded_conditional")
    ci_l_encoded, ci_u_encoded = safe_ci(beta_encoded, se_encoded)

    beta_nonencoded = float(fit.params[idx_nonencoded])
    se_nonencoded = extract_se_from_cov(fit, idx_nonencoded, "beta_MY_nonencoded_conditional")
    ci_l_nonencoded, ci_u_nonencoded = safe_ci(beta_nonencoded, se_nonencoded)

    return {
        "conditional_cox_success": 1,

        "beta_EY_direct_conditional": beta_e,
        "se_EY_direct_conditional": se_e,
        "p_EY_direct_conditional": p_from_z(beta_e / se_e),
        "hr_EY_direct_conditional": float(np.exp(beta_e)),
        "ci95_l_EY_direct_conditional": float(np.exp(ci_l_e)),
        "ci95_u_EY_direct_conditional": float(np.exp(ci_u_e)),

        "beta_MY_encoded_conditional": beta_encoded,
        "se_MY_encoded_conditional": se_encoded,
        "p_MY_encoded_conditional": p_from_z(beta_encoded / se_encoded),
        "hr_MY_encoded_conditional": float(np.exp(beta_encoded)),
        "ci95_l_MY_encoded_conditional": float(np.exp(ci_l_encoded)),
        "ci95_u_MY_encoded_conditional": float(np.exp(ci_u_encoded)),

        "beta_MY_nonencoded_conditional": beta_nonencoded,
        "se_MY_nonencoded_conditional": se_nonencoded,
        "p_MY_nonencoded_conditional": p_from_z(beta_nonencoded / se_nonencoded),
        "hr_MY_nonencoded_conditional": float(np.exp(beta_nonencoded)),
        "ci95_l_MY_nonencoded_conditional": float(np.exp(ci_l_nonencoded)),
        "ci95_u_MY_nonencoded_conditional": float(np.exp(ci_u_nonencoded)),

        "n_conditional_cox_model": int(dat.shape[0]),
    }


def compute_indirect(a, se_a, b, se_b):
    indirect = a * b
    var_indirect = (b ** 2) * (se_a ** 2) + (a ** 2) * (se_b ** 2)

    if np.isfinite(var_indirect) and var_indirect >= 0:
        se_indirect = float(np.sqrt(var_indirect))
    else:
        se_indirect = np.nan

    if np.isfinite(se_indirect) and se_indirect > 0:
        z = indirect / se_indirect
        p = p_from_z(z)
    else:
        z = np.nan
        p = np.nan

    ci_l, ci_u = safe_ci(indirect, se_indirect)

    return {
        "indirect_logHR": float(indirect),
        "indirect_hr": float(np.exp(indirect)) if np.isfinite(indirect) else np.nan,
        "se_indirect_logHR": se_indirect,
        "z_indirect": z,
        "p_indirect": p,
        "ci95_l_indirect_logHR": ci_l,
        "ci95_u_indirect_logHR": ci_u,
        "ci95_l_indirect_hr": float(np.exp(ci_l)) if np.isfinite(ci_l) else np.nan,
        "ci95_u_indirect_hr": float(np.exp(ci_u)) if np.isfinite(ci_u) else np.nan,
    }


def precheck(dat, covariates, args):
    needed = [
        "E",
        "M_encoded",
        "M_nonencoded",
        "event",
        "time_days",
        "eligible_for_incident",
    ] + covariates

    cc_mask = (
        dat["eligible_for_incident"].eq(1)
        & dat["E"].notna()
        & dat["M_encoded"].notna()
        & dat["M_nonencoded"].notna()
        & dat["event"].isin([0, 1])
        & dat["time_days"].notna()
        & (dat["time_days"] > 0)
        & dat[covariates].notna().all(axis=1)
    )

    cc = dat.loc[cc_mask, needed].copy()

    out = {
        "n_total_rows": int(dat.shape[0]),
        "n_complete_case": int(cc.shape[0]),
    }

    if cc.empty:
        out.update(
            {
                "n_events_complete_case": 0,
                "n_carriers_complete_case": 0,
                "n_noncarriers_complete_case": 0,
                "n_carrier_events_complete_case": 0,
                "n_carrier_nonevents_complete_case": 0,
                "M_encoded_sd_complete_case": np.nan,
                "M_nonencoded_sd_complete_case": np.nan,
            }
        )
        return cc, out, ["no_complete_case_rows"]

    e = cc["E"].to_numpy()
    y = cc["event"].astype(int).to_numpy()

    carrier = e > 0
    noncarrier = ~carrier
    event = y == 1
    nonevent = y == 0

    out.update(
        {
            "n_events_complete_case": int(event.sum()),
            "n_carriers_complete_case": int(carrier.sum()),
            "n_noncarriers_complete_case": int(noncarrier.sum()),
            "n_carrier_events_complete_case": int((carrier & event).sum()),
            "n_carrier_nonevents_complete_case": int((carrier & nonevent).sum()),
            "M_encoded_sd_complete_case": float(cc["M_encoded"].std(skipna=True)),
            "M_nonencoded_sd_complete_case": float(cc["M_nonencoded"].std(skipna=True)),
        }
    )

    fail = []

    if out["n_complete_case"] < args.min_complete_case:
        fail.append("low_complete_case")
    if out["n_events_complete_case"] < args.min_events_complete_case:
        fail.append("low_events_complete_case")
    if out["n_carriers_complete_case"] < args.min_carriers_complete_case:
        fail.append("low_carriers_complete_case")
    if out["n_noncarriers_complete_case"] < args.min_noncarriers_complete_case:
        fail.append("low_noncarriers_complete_case")
    if out["n_carrier_events_complete_case"] < args.min_carrier_events_complete_case:
        fail.append("low_carrier_events_complete_case")
    if out["n_carrier_nonevents_complete_case"] < args.min_carrier_nonevents_complete_case:
        fail.append("low_carrier_nonevents_complete_case")
    if not np.isfinite(out["M_encoded_sd_complete_case"]) or out["M_encoded_sd_complete_case"] <= args.min_M_sd:
        fail.append("low_M_encoded_variation")
    if not np.isfinite(out["M_nonencoded_sd_complete_case"]) or out["M_nonencoded_sd_complete_case"] <= args.min_M_sd:
        fail.append("low_M_nonencoded_variation")
    if np.nanstd(cc["E"].to_numpy()) <= 0:
        fail.append("no_E_variation")

    return cc, out, fail


def assign_conditional_class(row, p_cutoff):
    e_sig = row.get("encoded_conditional_indirect_p", np.nan)
    n_sig = row.get("nonencoded_conditional_indirect_p", np.nan)

    encoded_retained = np.isfinite(e_sig) and e_sig < p_cutoff
    nonencoded_retained = np.isfinite(n_sig) and n_sig < p_cutoff

    if encoded_retained and nonencoded_retained:
        return "both_retained"
    if encoded_retained and not nonencoded_retained:
        return "encoded_only_retained"
    if nonencoded_retained and not encoded_retained:
        return "nonencoded_only_retained"
    return "neither_retained"


def summarize_overall(res, p_cutoff=0.05):
    rows = [
        ("n_pairs", res.shape[0]),
        ("n_conditional_success", int(res["conditional_model_success"].eq(1).sum())),
        ("n_conditional_failed", int(res["conditional_model_success"].ne(1).sum())),
    ]

    success = res["conditional_model_success"].eq(1)
    sub = res.loc[success].copy()

    rows.append(
        (
            "n_encoded_conditional_nominal",
            int(sub["encoded_conditional_indirect_p"].lt(p_cutoff).sum()),
        )
    )
    rows.append(
        (
            "prop_encoded_conditional_nominal",
            sub["encoded_conditional_indirect_p"].lt(p_cutoff).mean() if not sub.empty else np.nan,
        )
    )
    rows.append(
        (
            "n_nonencoded_conditional_nominal",
            int(sub["nonencoded_conditional_indirect_p"].lt(p_cutoff).sum()),
        )
    )
    rows.append(
        (
            "prop_nonencoded_conditional_nominal",
            sub["nonencoded_conditional_indirect_p"].lt(p_cutoff).mean() if not sub.empty else np.nan,
        )
    )

    for cls, n in sub["conditional_mediation_class"].value_counts().items():
        rows.append((f"conditional_class::{cls}::n", int(n)))
        rows.append((f"conditional_class::{cls}::proportion", n / sub.shape[0] if sub.shape[0] > 0 else np.nan))

    rows += [
        ("median_encoded_conditional_abs_indirect_logHR", sub["encoded_conditional_abs_indirect_logHR"].median()),
        ("median_nonencoded_conditional_abs_indirect_logHR", sub["nonencoded_conditional_abs_indirect_logHR"].median()),
        ("median_encoded_conditional_indirect_hr", sub["encoded_conditional_indirect_hr"].median()),
        ("median_nonencoded_conditional_indirect_hr", sub["nonencoded_conditional_indirect_hr"].median()),
    ]

    return pd.DataFrame(rows, columns=["metric", "value"])


def summarize_by_priority_class(res, p_cutoff=0.05):
    if "encoded_vs_nonencoded_priority_class" not in res.columns:
        return pd.DataFrame()

    sub = res.loc[res["conditional_model_success"].eq(1)].copy()

    rows = []

    for cls, g in sub.groupby("encoded_vs_nonencoded_priority_class", dropna=False):
        rows.append(
            {
                "encoded_vs_nonencoded_priority_class": cls,
                "n_pairs": g.shape[0],
                "n_encoded_conditional_nominal": int(g["encoded_conditional_indirect_p"].lt(p_cutoff).sum()),
                "prop_encoded_conditional_nominal": g["encoded_conditional_indirect_p"].lt(p_cutoff).mean(),
                "n_nonencoded_conditional_nominal": int(g["nonencoded_conditional_indirect_p"].lt(p_cutoff).sum()),
                "prop_nonencoded_conditional_nominal": g["nonencoded_conditional_indirect_p"].lt(p_cutoff).mean(),
                "median_encoded_conditional_abs_indirect_logHR": g["encoded_conditional_abs_indirect_logHR"].median(),
                "median_nonencoded_conditional_abs_indirect_logHR": g["nonencoded_conditional_abs_indirect_logHR"].median(),
            }
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runset = load_runset(args.conditional_runset)

    protein_data = load_covariate_protein_table(
        args.covariate_protein_table,
        runset,
        args.covariates,
    )

    samples = protein_data.index.astype(str)

    e_by_gene = load_e_matrix(
        args.e_samples,
        args.e_genes,
        args.e_matrix,
        sorted(runset["gene"].astype(str).unique()),
    )

    follow = prepare_followup(args.followup_dates)

    occ = load_first_occurrence(
        args.first_occurrence,
        sorted(runset["icd3"].astype(str).unique()),
    )

    phenotype_cache = {}
    results = []

    for _, row in runset.iterrows():
        gene = str(row["gene"])
        icd3 = str(row["icd3"])
        encoded = str(row["encoded_protein"])
        nonencoded = str(row["nonencoded_protein"])

        out = row.to_dict()
        out.update(
            {
                "conditional_model_success": 0,
                "fail_reason": "",
            }
        )

        try:
            if icd3 not in phenotype_cache:
                phenotype_cache[icd3] = build_incident_phenotype(
                    icd3,
                    samples,
                    follow,
                    occ,
                )

            pheno = phenotype_cache[icd3]
            e = e_by_gene[gene].reindex(samples)

            dat = protein_data[args.covariates].copy()
            dat["E"] = pd.to_numeric(e, errors="coerce")
            dat["M_encoded"] = protein_data[encoded]
            dat["M_nonencoded"] = protein_data[nonencoded]
            dat["event"] = pheno["event"]
            dat["time_days"] = pheno["time_days"]
            dat["eligible_for_incident"] = pheno["eligible_for_incident"]

            cc, counts, fail = precheck(dat, args.covariates, args)
            out.update(counts)

            if fail:
                out["fail_reason"] = ";".join(fail)
                results.append(out)
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                em_encoded = fit_ols_em(
                    cc,
                    mediator_col="M_encoded",
                    covariates=args.covariates,
                    label="encoded",
                )

                em_nonencoded = fit_ols_em(
                    cc,
                    mediator_col="M_nonencoded",
                    covariates=args.covariates,
                    label="nonencoded",
                )

                cox = fit_conditional_cox(cc, args.covariates)

            out.update(em_encoded)
            out.update(em_nonencoded)
            out.update(cox)

            encoded_ind = compute_indirect(
                a=out["beta_EM_encoded"],
                se_a=out["se_EM_encoded"],
                b=out["beta_MY_encoded_conditional"],
                se_b=out["se_MY_encoded_conditional"],
            )

            nonencoded_ind = compute_indirect(
                a=out["beta_EM_nonencoded"],
                se_a=out["se_EM_nonencoded"],
                b=out["beta_MY_nonencoded_conditional"],
                se_b=out["se_MY_nonencoded_conditional"],
            )

            out.update(
                {
                    "encoded_conditional_indirect_logHR": encoded_ind["indirect_logHR"],
                    "encoded_conditional_indirect_hr": encoded_ind["indirect_hr"],
                    "encoded_conditional_abs_indirect_logHR": abs(encoded_ind["indirect_logHR"]),
                    "encoded_conditional_se_indirect_logHR": encoded_ind["se_indirect_logHR"],
                    "encoded_conditional_z_indirect": encoded_ind["z_indirect"],
                    "encoded_conditional_indirect_p": encoded_ind["p_indirect"],
                    "encoded_conditional_ci95_l_indirect_logHR": encoded_ind["ci95_l_indirect_logHR"],
                    "encoded_conditional_ci95_u_indirect_logHR": encoded_ind["ci95_u_indirect_logHR"],
                    "encoded_conditional_ci95_l_indirect_hr": encoded_ind["ci95_l_indirect_hr"],
                    "encoded_conditional_ci95_u_indirect_hr": encoded_ind["ci95_u_indirect_hr"],

                    "nonencoded_conditional_indirect_logHR": nonencoded_ind["indirect_logHR"],
                    "nonencoded_conditional_indirect_hr": nonencoded_ind["indirect_hr"],
                    "nonencoded_conditional_abs_indirect_logHR": abs(nonencoded_ind["indirect_logHR"]),
                    "nonencoded_conditional_se_indirect_logHR": nonencoded_ind["se_indirect_logHR"],
                    "nonencoded_conditional_z_indirect": nonencoded_ind["z_indirect"],
                    "nonencoded_conditional_indirect_p": nonencoded_ind["p_indirect"],
                    "nonencoded_conditional_ci95_l_indirect_logHR": nonencoded_ind["ci95_l_indirect_logHR"],
                    "nonencoded_conditional_ci95_u_indirect_logHR": nonencoded_ind["ci95_u_indirect_logHR"],
                    "nonencoded_conditional_ci95_l_indirect_hr": nonencoded_ind["ci95_l_indirect_hr"],
                    "nonencoded_conditional_ci95_u_indirect_hr": nonencoded_ind["ci95_u_indirect_hr"],
                }
            )

            out["conditional_model_success"] = 1
            out["conditional_mediation_class"] = assign_conditional_class(
                out,
                args.retained_p_cutoff,
            )

        except Exception as e:
            out["fail_reason"] = f"conditional_model_failed::{str(e)[:300]}"

        results.append(out)

    res = pd.DataFrame(results)

    success = res["conditional_model_success"].eq(1)

    res["encoded_conditional_indirect_q"] = np.nan
    res["nonencoded_conditional_indirect_q"] = np.nan

    res.loc[success, "encoded_conditional_indirect_q"] = bh_fdr(
        res.loc[success, "encoded_conditional_indirect_p"]
    )

    res.loc[success, "nonencoded_conditional_indirect_q"] = bh_fdr(
        res.loc[success, "nonencoded_conditional_indirect_p"]
    )

    res["encoded_conditional_fdr_sig"] = (
        res["encoded_conditional_indirect_q"].lt(0.05)
    ).astype("Int64")

    res["nonencoded_conditional_fdr_sig"] = (
        res["nonencoded_conditional_indirect_q"].lt(0.05)
    ).astype("Int64")

    overall = summarize_overall(res, args.retained_p_cutoff)
    by_priority = summarize_by_priority_class(res, args.retained_p_cutoff)

    fig_class = (
        res.loc[res["conditional_model_success"].eq(1), "conditional_mediation_class"]
        .value_counts()
        .reset_index()
    )
    fig_class.columns = ["conditional_mediation_class", "n_pairs"]
    fig_class["proportion"] = fig_class["n_pairs"] / fig_class["n_pairs"].sum()

    out_res = outdir / f"{args.prefix}_conditional_mediation_encoded_vs_nonencoded.tsv"
    out_overall = outdir / f"{args.prefix}_conditional_mediation_overall_summary.tsv"
    out_by_priority = outdir / f"{args.prefix}_conditional_mediation_by_priority_class.tsv"
    out_fig = outdir / f"{args.prefix}_figure_conditional_mediation_class_summary.tsv"

    res.to_csv(out_res, sep="\t", index=False)
    overall.to_csv(out_overall, sep="\t", index=False)
    by_priority.to_csv(out_by_priority, sep="\t", index=False)
    fig_class.to_csv(out_fig, sep="\t", index=False)

    print("[Done] Step6 conditional mediation analysis completed.")
    print(f"[Output] {out_res}")
    print(f"[Output] {out_overall}")
    print(f"[Output] {out_by_priority}")
    print(f"[Output] {out_fig}")

    print("[QC]")
    for _, r in overall.iterrows():
        print(f"{r['metric']}\t{r['value']}")


if __name__ == "__main__":
    main()
