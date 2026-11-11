#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
07_residual_mediation_analysis.py

Purpose
-------
Residual mediation analysis for encoded protein and best non-encoded protein.

For each gene-disease pair, two residual directions are tested:

1. encoded_residual_after_nonencoded
   M_encoded_residual = residual of M_encoded ~ M_nonencoded + covariates

2. nonencoded_residual_after_encoded
   M_nonencoded_residual = residual of M_nonencoded ~ M_encoded + covariates

Then test:
   residual_M ~ E + covariates
   Surv(time_days, event) ~ E + adjusted_protein + residual_M + covariates

Interpretation
--------------
If residual mediation remains significant, the protein provides specific
mediator information beyond the paired protein.

Inputs
------
1. 06_conditional_mediation_encoded_vs_nonencoded.tsv
2. wes_olink_covariates_analytic.tsv
3. E_matrix.samples.txt
4. E_matrix.genes.txt
5. E_matrix.carrier_binary.npz
6. icd10_first_occurrence.tsv
7. followup_dates.tsv

Outputs
-------
1. 07_residual_mediation_long.tsv
2. 07_residual_mediation_pair_summary.tsv
3. 07_residual_mediation_overall_summary.tsv
4. 07_residual_mediation_by_priority_class.tsv
5. 07_figure_residual_specific_component_summary.tsv
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

DIR_ENCODED = "encoded_residual_after_nonencoded"
DIR_NONENCODED = "nonencoded_residual_after_encoded"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run residual mediation analysis for encoded and best non-encoded proteins."
    )

    parser.add_argument("--residual_runset", required=True)
    parser.add_argument("--covariate_protein_table", required=True)

    parser.add_argument("--e_samples", required=True)
    parser.add_argument("--e_genes", required=True)
    parser.add_argument("--e_matrix", required=True)

    parser.add_argument("--first_occurrence", required=True)
    parser.add_argument("--followup_dates", required=True)

    parser.add_argument("--outdir", required=True)
    parser.add_argument("--prefix", default="07")

    parser.add_argument("--covariates", nargs="+", default=DEFAULT_COVARIATES)

    parser.add_argument("--min_complete_case", type=int, default=200)
    parser.add_argument("--min_events_complete_case", type=int, default=20)
    parser.add_argument("--min_carriers_complete_case", type=int, default=10)
    parser.add_argument("--min_noncarriers_complete_case", type=int, default=200)
    parser.add_argument("--min_carrier_events_complete_case", type=int, default=2)
    parser.add_argument("--min_carrier_nonevents_complete_case", type=int, default=5)
    parser.add_argument("--min_residual_sd", type=float, default=1e-8)

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
        raise ValueError("Missing required columns in residual runset: " + ", ".join(missing))

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
    if np.nanstd(cc["E"].to_numpy()) <= 0:
        fail.append("no_E_variation")

    return cc, out, fail


def residualize_target(cc, target_col, adjust_col, covariates):
    """
    Residualize target protein on the paired protein and covariates.

    Scientific logic:
    The residual represents the protein-specific component not explained by
    the paired protein and baseline covariates.
    """
    cols = [adjust_col] + covariates
    X = sm.add_constant(cc[cols], has_constant="add")
    y = cc[target_col].astype(float)

    fit = sm.OLS(y, X).fit()
    resid = y - fit.fittedvalues

    return resid, float(fit.rsquared) if np.isfinite(fit.rsquared) else np.nan


def fit_residual_em(cc, covariates):
    """
    Residual_M ~ E + covariates
    """
    cols = ["E"] + covariates
    X = sm.add_constant(cc[cols], has_constant="add")
    y = cc["M_residual"].astype(float)

    fit = sm.OLS(y, X).fit()

    beta = float(fit.params["E"])
    se = float(fit.bse["E"])
    p = float(fit.pvalues["E"])
    ci_l, ci_u = safe_ci(beta, se)

    if not np.isfinite(beta) or not np.isfinite(se):
        raise ValueError("nonfinite_residual_EM")

    return {
        "beta_E_residual": beta,
        "se_E_residual": se,
        "p_E_residual": p,
        "ci95_l_E_residual": ci_l,
        "ci95_u_E_residual": ci_u,
        "n_residual_mediator_model": int(cc.shape[0]),
    }


def fit_residual_cox(cc, adjust_col, covariates):
    """
    Surv(time_days, event) ~ E + adjusted_protein + residual_M + covariates
    """
    cols = ["E", adjust_col, "M_residual"] + covariates
    X = cc[cols]

    time = cc["time_days"].astype(float).to_numpy()
    event = cc["event"].astype(int).to_numpy()

    model = PHReg(endog=time, exog=X.to_numpy(), status=event, ties="efron")
    fit = model.fit()

    idx_e = cols.index("E")
    idx_adjust = cols.index(adjust_col)
    idx_residual = cols.index("M_residual")

    beta_e = float(fit.params[idx_e])
    se_e = extract_se_from_cov(fit, idx_e, "beta_EY_direct_residual_model")
    ci_l_e, ci_u_e = safe_ci(beta_e, se_e)

    beta_adjust = float(fit.params[idx_adjust])
    se_adjust = extract_se_from_cov(fit, idx_adjust, "beta_MY_adjusted_protein")
    ci_l_adjust, ci_u_adjust = safe_ci(beta_adjust, se_adjust)

    beta_resid = float(fit.params[idx_residual])
    se_resid = extract_se_from_cov(fit, idx_residual, "beta_residual_Y")
    ci_l_resid, ci_u_resid = safe_ci(beta_resid, se_resid)

    return {
        "residual_cox_success": 1,

        "beta_EY_direct_residual_model": beta_e,
        "se_EY_direct_residual_model": se_e,
        "p_EY_direct_residual_model": p_from_z(beta_e / se_e),
        "hr_EY_direct_residual_model": float(np.exp(beta_e)),
        "ci95_l_EY_direct_residual_model": float(np.exp(ci_l_e)),
        "ci95_u_EY_direct_residual_model": float(np.exp(ci_u_e)),

        "beta_MY_adjusted_protein": beta_adjust,
        "se_MY_adjusted_protein": se_adjust,
        "p_MY_adjusted_protein": p_from_z(beta_adjust / se_adjust),
        "hr_MY_adjusted_protein": float(np.exp(beta_adjust)),
        "ci95_l_MY_adjusted_protein": float(np.exp(ci_l_adjust)),
        "ci95_u_MY_adjusted_protein": float(np.exp(ci_u_adjust)),

        "beta_residual_Y": beta_resid,
        "se_residual_Y": se_resid,
        "p_residual_Y": p_from_z(beta_resid / se_resid),
        "hr_residual_Y": float(np.exp(beta_resid)),
        "ci95_l_residual_Y": float(np.exp(ci_l_resid)),
        "ci95_u_residual_Y": float(np.exp(ci_u_resid)),

        "n_residual_cox_model": int(cc.shape[0]),
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
        "residual_indirect_logHR": float(indirect),
        "residual_indirect_hr": float(np.exp(indirect)) if np.isfinite(indirect) else np.nan,
        "residual_abs_indirect_logHR": abs(float(indirect)) if np.isfinite(indirect) else np.nan,
        "residual_se_indirect_logHR": se_indirect,
        "residual_z_indirect": z,
        "residual_indirect_p": p,
        "ci95_l_residual_indirect_logHR": ci_l,
        "ci95_u_residual_indirect_logHR": ci_u,
        "ci95_l_residual_indirect_hr": float(np.exp(ci_l)) if np.isfinite(ci_l) else np.nan,
        "ci95_u_residual_indirect_hr": float(np.exp(ci_u)) if np.isfinite(ci_u) else np.nan,
    }


def run_one_direction(base_out, cc_original, direction, target_col, adjust_col, target_name, adjust_name, covariates, args):
    out = base_out.copy()

    out.update(
        {
            "residual_direction": direction,
            "target_protein": target_name,
            "adjusted_for_protein": adjust_name,
            "residual_model_success": 0,
            "fail_reason": "",
        }
    )

    try:
        cc = cc_original.copy()

        resid, r2 = residualize_target(
            cc=cc,
            target_col=target_col,
            adjust_col=adjust_col,
            covariates=covariates,
        )

        cc["M_residual"] = resid

        residual_sd = float(cc["M_residual"].std(skipna=True))

        out["residualization_r2"] = r2
        out["M_residual_sd"] = residual_sd

        if not np.isfinite(residual_sd) or residual_sd <= args.min_residual_sd:
            out["fail_reason"] = "low_residual_variation"
            return out

        em = fit_residual_em(cc, covariates)
        cox = fit_residual_cox(cc, adjust_col, covariates)

        indirect = compute_indirect(
            a=em["beta_E_residual"],
            se_a=em["se_E_residual"],
            b=cox["beta_residual_Y"],
            se_b=cox["se_residual_Y"],
        )

        out.update(em)
        out.update(cox)
        out.update(indirect)

        out["residual_model_success"] = int(
            np.isfinite(out["residual_indirect_logHR"])
            and np.isfinite(out["residual_indirect_p"])
        )

        if out["residual_model_success"] != 1:
            out["fail_reason"] = "nonfinite_residual_indirect"

    except Exception as e:
        out["fail_reason"] = f"residual_model_failed::{str(e)[:300]}"

    return out


def build_pair_summary(res, p_cutoff):
    encoded = res.loc[res["residual_direction"] == DIR_ENCODED].copy()
    nonencoded = res.loc[res["residual_direction"] == DIR_NONENCODED].copy()

    keep_meta = [
        "pair_id",
        "gene",
        "icd3",
        "encoded_protein",
        "nonencoded_protein",
        "tier_label",
        "encoded_vs_nonencoded_priority_class",
        "correlation_group",
        "pathway_interpretation_class",
    ]

    meta = (
        res.loc[:, [c for c in keep_meta if c in res.columns]]
        .drop_duplicates("pair_id")
        .copy()
    )

    encoded_keep = [
        "pair_id",
        "residual_model_success",
        "residual_indirect_logHR",
        "residual_indirect_hr",
        "residual_abs_indirect_logHR",
        "residual_indirect_p",
        "residual_indirect_q_global",
        "residual_indirect_q_by_direction",
        "residualization_r2",
        "M_residual_sd",
    ]

    nonencoded_keep = encoded_keep.copy()

    encoded = encoded.loc[:, [c for c in encoded_keep if c in encoded.columns]].rename(
        columns={
            "residual_model_success": "encoded_specific_success",
            "residual_indirect_logHR": "encoded_specific_indirect_logHR",
            "residual_indirect_hr": "encoded_specific_indirect_hr",
            "residual_abs_indirect_logHR": "encoded_specific_abs_indirect_logHR",
            "residual_indirect_p": "encoded_specific_indirect_p",
            "residual_indirect_q_global": "encoded_specific_indirect_q_global",
            "residual_indirect_q_by_direction": "encoded_specific_indirect_q_by_direction",
            "residualization_r2": "encoded_residualization_r2_after_nonencoded",
            "M_residual_sd": "encoded_residual_sd",
        }
    )

    nonencoded = nonencoded.loc[:, [c for c in nonencoded_keep if c in nonencoded.columns]].rename(
        columns={
            "residual_model_success": "nonencoded_specific_success",
            "residual_indirect_logHR": "nonencoded_specific_indirect_logHR",
            "residual_indirect_hr": "nonencoded_specific_indirect_hr",
            "residual_abs_indirect_logHR": "nonencoded_specific_abs_indirect_logHR",
            "residual_indirect_p": "nonencoded_specific_indirect_p",
            "residual_indirect_q_global": "nonencoded_specific_indirect_q_global",
            "residual_indirect_q_by_direction": "nonencoded_specific_indirect_q_by_direction",
            "residualization_r2": "nonencoded_residualization_r2_after_encoded",
            "M_residual_sd": "nonencoded_residual_sd",
        }
    )

    pair = meta.merge(encoded, on="pair_id", how="left").merge(nonencoded, on="pair_id", how="left")

    pair["encoded_specific_retained_nominal"] = (
        pair["encoded_specific_indirect_p"].lt(p_cutoff)
    ).astype("Int64")

    pair["nonencoded_specific_retained_nominal"] = (
        pair["nonencoded_specific_indirect_p"].lt(p_cutoff)
    ).astype("Int64")

    def classify(row):
        e = row.get("encoded_specific_retained_nominal", 0) == 1
        n = row.get("nonencoded_specific_retained_nominal", 0) == 1

        if e and n:
            return "both_specific_retained"
        if e and not n:
            return "encoded_specific_only"
        if n and not e:
            return "nonencoded_specific_only"
        return "neither_specific_retained"

    pair["residual_specific_class"] = pair.apply(classify, axis=1)

    return pair


def summarize_overall(res, pair, p_cutoff=0.05):
    rows = [
        ("n_pairs", pair["pair_id"].nunique()),
        ("n_residual_rows", res.shape[0]),
        ("n_residual_success", int(res["residual_model_success"].eq(1).sum())),
        ("n_residual_failed", int(res["residual_model_success"].ne(1).sum())),
    ]

    for direction, g in res.groupby("residual_direction"):
        success = g["residual_model_success"].eq(1)
        gs = g.loc[success]

        rows += [
            (f"{direction}::n_rows", g.shape[0]),
            (f"{direction}::n_success", int(success.sum())),
            (f"{direction}::n_nominal", int(gs["residual_indirect_p"].lt(p_cutoff).sum())),
            (
                f"{direction}::prop_nominal",
                gs["residual_indirect_p"].lt(p_cutoff).mean() if not gs.empty else np.nan,
            ),
            (
                f"{direction}::median_abs_indirect_logHR",
                gs["residual_abs_indirect_logHR"].median(),
            ),
            (
                f"{direction}::median_indirect_hr",
                gs["residual_indirect_hr"].median(),
            ),
            (
                f"{direction}::median_residualization_r2",
                gs["residualization_r2"].median(),
            ),
        ]

    for cls, n in pair["residual_specific_class"].value_counts().items():
        rows.append((f"residual_specific_class::{cls}::n", int(n)))
        rows.append((f"residual_specific_class::{cls}::proportion", n / pair.shape[0]))

    return pd.DataFrame(rows, columns=["metric", "value"])


def summarize_by_priority_class(pair):
    if "encoded_vs_nonencoded_priority_class" not in pair.columns:
        return pd.DataFrame()

    rows = []

    for cls, g in pair.groupby("encoded_vs_nonencoded_priority_class", dropna=False):
        rows.append(
            {
                "encoded_vs_nonencoded_priority_class": cls,
                "n_pairs": g.shape[0],
                "n_both_specific_retained": int((g["residual_specific_class"] == "both_specific_retained").sum()),
                "n_encoded_specific_only": int((g["residual_specific_class"] == "encoded_specific_only").sum()),
                "n_nonencoded_specific_only": int((g["residual_specific_class"] == "nonencoded_specific_only").sum()),
                "n_neither_specific_retained": int((g["residual_specific_class"] == "neither_specific_retained").sum()),
                "prop_both_specific_retained": (g["residual_specific_class"] == "both_specific_retained").mean(),
                "prop_encoded_specific_only": (g["residual_specific_class"] == "encoded_specific_only").mean(),
                "prop_nonencoded_specific_only": (g["residual_specific_class"] == "nonencoded_specific_only").mean(),
                "prop_neither_specific_retained": (g["residual_specific_class"] == "neither_specific_retained").mean(),
                "median_encoded_specific_abs_indirect_logHR": g["encoded_specific_abs_indirect_logHR"].median(),
                "median_nonencoded_specific_abs_indirect_logHR": g["nonencoded_specific_abs_indirect_logHR"].median(),
            }
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runset = load_runset(args.residual_runset)

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

        base_out = row.to_dict()

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

            if fail:
                for direction, target_name, adjust_name in [
                    (DIR_ENCODED, encoded, nonencoded),
                    (DIR_NONENCODED, nonencoded, encoded),
                ]:
                    out = base_out.copy()
                    out.update(counts)
                    out.update(
                        {
                            "residual_direction": direction,
                            "target_protein": target_name,
                            "adjusted_for_protein": adjust_name,
                            "residual_model_success": 0,
                            "fail_reason": ";".join(fail),
                        }
                    )
                    results.append(out)
                continue

            base_out.update(counts)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                res_encoded = run_one_direction(
                    base_out=base_out,
                    cc_original=cc,
                    direction=DIR_ENCODED,
                    target_col="M_encoded",
                    adjust_col="M_nonencoded",
                    target_name=encoded,
                    adjust_name=nonencoded,
                    covariates=args.covariates,
                    args=args,
                )

                res_nonencoded = run_one_direction(
                    base_out=base_out,
                    cc_original=cc,
                    direction=DIR_NONENCODED,
                    target_col="M_nonencoded",
                    adjust_col="M_encoded",
                    target_name=nonencoded,
                    adjust_name=encoded,
                    covariates=args.covariates,
                    args=args,
                )

            results.append(res_encoded)
            results.append(res_nonencoded)

        except Exception as e:
            for direction, target_name, adjust_name in [
                (DIR_ENCODED, encoded, nonencoded),
                (DIR_NONENCODED, nonencoded, encoded),
            ]:
                out = base_out.copy()
                out.update(
                    {
                        "residual_direction": direction,
                        "target_protein": target_name,
                        "adjusted_for_protein": adjust_name,
                        "residual_model_success": 0,
                        "fail_reason": f"residual_pair_failed::{str(e)[:300]}",
                    }
                )
                results.append(out)

    res = pd.DataFrame(results)

    success = res["residual_model_success"].eq(1)

    res["residual_indirect_q_global"] = np.nan
    res.loc[success, "residual_indirect_q_global"] = bh_fdr(
        res.loc[success, "residual_indirect_p"]
    )

    res["residual_indirect_q_by_direction"] = np.nan

    for direction, idx in res.groupby("residual_direction").groups.items():
        idx = pd.Index(idx)
        idx_success = idx[res.loc[idx, "residual_model_success"].eq(1)]
        res.loc[idx_success, "residual_indirect_q_by_direction"] = bh_fdr(
            res.loc[idx_success, "residual_indirect_p"]
        )

    res["residual_nominal_sig"] = (
        res["residual_indirect_p"].lt(args.retained_p_cutoff)
    ).astype("Int64")

    res["residual_fdr_sig_global"] = (
        res["residual_indirect_q_global"].lt(0.05)
    ).astype("Int64")

    res["residual_fdr_sig_by_direction"] = (
        res["residual_indirect_q_by_direction"].lt(0.05)
    ).astype("Int64")

    pair = build_pair_summary(res, args.retained_p_cutoff)
    overall = summarize_overall(res, pair, args.retained_p_cutoff)
    by_priority = summarize_by_priority_class(pair)

    fig_class = (
        pair["residual_specific_class"]
        .value_counts()
        .reset_index()
    )
    fig_class.columns = ["residual_specific_class", "n_pairs"]
    fig_class["proportion"] = fig_class["n_pairs"] / fig_class["n_pairs"].sum()

    out_long = outdir / f"{args.prefix}_residual_mediation_long.tsv"
    out_pair = outdir / f"{args.prefix}_residual_mediation_pair_summary.tsv"
    out_overall = outdir / f"{args.prefix}_residual_mediation_overall_summary.tsv"
    out_by_priority = outdir / f"{args.prefix}_residual_mediation_by_priority_class.tsv"
    out_fig = outdir / f"{args.prefix}_figure_residual_specific_component_summary.tsv"

    res.to_csv(out_long, sep="\t", index=False)
    pair.to_csv(out_pair, sep="\t", index=False)
    overall.to_csv(out_overall, sep="\t", index=False)
    by_priority.to_csv(out_by_priority, sep="\t", index=False)
    fig_class.to_csv(out_fig, sep="\t", index=False)

    print("[Done] Step7 residual mediation analysis completed.")
    print(f"[Output] {out_long}")
    print(f"[Output] {out_pair}")
    print(f"[Output] {out_overall}")
    print(f"[Output] {out_by_priority}")
    print(f"[Output] {out_fig}")

    print("[QC]")
    for _, r in overall.iterrows():
        print(f"{r['metric']}\t{r['value']}")


if __name__ == "__main__":
    main()
