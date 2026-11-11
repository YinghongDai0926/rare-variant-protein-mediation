#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg


DEFAULT_COVS = ["age", "sex"] + [f"PC{i}" for i in range(1, 11)]


def balanced_chunk_indices(n_items, n_chunks, chunk_id):
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    if not (0 <= chunk_id < n_chunks):
        raise ValueError("chunk_id out of range")
    bins = [[] for _ in range(n_chunks)]
    for i in range(n_items):
        bins[i % n_chunks].append(i)
    return bins[chunk_id]


def stable_seed_from_string(s, seed_base=20260416):
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + int(seed_base)) % (2**32 - 1)


def is_finite_scalar(x):
    return pd.notna(x) and np.isfinite(float(x))


def safe_ci_from_beta_se(beta, se):
    if (not is_finite_scalar(beta)) or (not is_finite_scalar(se)):
        return np.nan, np.nan
    beta = float(beta)
    se = float(se)
    l = beta - 1.96 * se
    u = beta + 1.96 * se
    return l, u


def require_finite_scalar(name, x):
    if not is_finite_scalar(x):
        raise ValueError(f"nonfinite_{name}")
    return float(x)


def extract_se_from_cov(fit, idx, name):
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


def fit_ols_mediator(dat, covariates):
    """
    M ~ E + covariates
    """
    cols = ["E"] + covariates
    X = dat[cols].copy()
    X = sm.add_constant(X, has_constant="add")
    y = dat["M"].astype(float)

    fit = sm.OLS(y, X).fit()

    beta_E = require_finite_scalar("beta_EM_individual", fit.params["E"])
    se_E = require_finite_scalar("se_EM_individual", fit.bse["E"])
    p_E = require_finite_scalar("p_EM_individual", fit.pvalues["E"])
    ci_l, ci_u = safe_ci_from_beta_se(beta_E, se_E)

    if (not np.isfinite(ci_l)) or (not np.isfinite(ci_u)):
        raise ValueError("nonfinite_ci_EM_individual")

    r2 = float(fit.rsquared) if hasattr(fit, "rsquared") and np.isfinite(fit.rsquared) else np.nan

    return {
        "mediator_model_success": 1,
        "beta_EM_individual": beta_E,
        "se_EM_individual": se_E,
        "p_EM_individual": p_E,
        "ci95_l_EM_individual": ci_l,
        "ci95_u_EM_individual": ci_u,
        "n_mediator_model": int(dat.shape[0]),
        "r2_mediator_model": r2
    }


def fit_phreg_total(dat, covariates):
    """
    Surv(time_days, event) ~ E + covariates
    """
    cols = ["E"] + covariates
    X = dat[cols].copy()
    time = dat["time_days"].astype(float).to_numpy()
    event = dat["event"].astype(int).to_numpy()

    model = PHReg(endog=time, exog=X.to_numpy(), status=event, ties="efron")
    fit = model.fit()

    idx_E = cols.index("E")
    beta_E = require_finite_scalar("beta_EY_total", fit.params[idx_E])
    se_E = extract_se_from_cov(fit, idx_E, "beta_EY_total")

    ci_l, ci_u = safe_ci_from_beta_se(beta_E, se_E)
    if (not np.isfinite(ci_l)) or (not np.isfinite(ci_u)):
        raise ValueError("nonfinite_ci_EY_total")

    hr = float(np.exp(beta_E))
    hr_l = float(np.exp(ci_l))
    hr_u = float(np.exp(ci_u))

    if (not np.isfinite(hr)) or (not np.isfinite(hr_l)) or (not np.isfinite(hr_u)):
        raise ValueError("nonfinite_hr_EY_total")

    return {
        "total_model_success": 1,
        "beta_EY_total": beta_E,
        "se_EY_total": se_E,
        "hr_EY_total": hr,
        "ci95_l_EY_total": hr_l,
        "ci95_u_EY_total": hr_u,
        "n_total_model": int(dat.shape[0])
    }


def fit_phreg_direct(dat, covariates):
    """
    Surv(time_days, event) ~ E + M + covariates
    """
    cols = ["E", "M"] + covariates
    X = dat[cols].copy()
    time = dat["time_days"].astype(float).to_numpy()
    event = dat["event"].astype(int).to_numpy()

    model = PHReg(endog=time, exog=X.to_numpy(), status=event, ties="efron")
    fit = model.fit()

    idx_E = cols.index("E")
    idx_M = cols.index("M")

    beta_E = require_finite_scalar("beta_EY_direct", fit.params[idx_E])
    se_E = extract_se_from_cov(fit, idx_E, "beta_EY_direct")

    beta_M = require_finite_scalar("beta_MY_conditional", fit.params[idx_M])
    se_M = extract_se_from_cov(fit, idx_M, "beta_MY_conditional")

    ci_l_E, ci_u_E = safe_ci_from_beta_se(beta_E, se_E)
    ci_l_M, ci_u_M = safe_ci_from_beta_se(beta_M, se_M)

    if (not np.isfinite(ci_l_E)) or (not np.isfinite(ci_u_E)):
        raise ValueError("nonfinite_ci_EY_direct")
    if (not np.isfinite(ci_l_M)) or (not np.isfinite(ci_u_M)):
        raise ValueError("nonfinite_ci_MY_conditional")

    hr_E = float(np.exp(beta_E))
    hr_E_l = float(np.exp(ci_l_E))
    hr_E_u = float(np.exp(ci_u_E))

    hr_M = float(np.exp(beta_M))
    hr_M_l = float(np.exp(ci_l_M))
    hr_M_u = float(np.exp(ci_u_M))

    if not all(np.isfinite(x) for x in [hr_E, hr_E_l, hr_E_u, hr_M, hr_M_l, hr_M_u]):
        raise ValueError("nonfinite_hr_direct_model")

    return {
        "direct_model_success": 1,
        "beta_EY_direct": beta_E,
        "se_EY_direct": se_E,
        "hr_EY_direct": hr_E,
        "ci95_l_EY_direct": hr_E_l,
        "ci95_u_EY_direct": hr_E_u,

        "beta_MY_conditional": beta_M,
        "se_MY_conditional": se_M,
        "hr_MY_conditional": hr_M,
        "ci95_l_MY_conditional": hr_M_l,
        "ci95_u_MY_conditional": hr_M_u,

        "n_direct_model": int(dat.shape[0])
    }


def bootstrap_indirect_effect(dat, covariates, n_boot=200, min_boot_success=100, seed=20260416):
    """
    Bootstrap indirect effect for survival mediation:
      indirect_logHR = a * b
    where:
      a = coef(E -> M) from OLS
      b = coef(M -> Y | E) from Cox PH direct model

    Robust version:
    - bootstrap replicate is counted as success only if a, b, and a*b are all finite
    - percentile CI is computed only from finite bootstrap values
    - CI must itself be finite, otherwise the bootstrap test fails
    """
    rng = np.random.default_rng(seed)
    n = dat.shape[0]

    boot_vals = []
    n_fail_precheck = 0
    n_fail_model = 0
    n_fail_nonfinite = 0

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bdat = dat.iloc[idx].copy()

        try:
            e = pd.to_numeric(bdat["E"], errors="coerce").to_numpy()
            m = pd.to_numeric(bdat["M"], errors="coerce").to_numpy()
            y = pd.to_numeric(bdat["event"], errors="coerce").to_numpy()
            t = pd.to_numeric(bdat["time_days"], errors="coerce").to_numpy()

            carrier_mask = e > 0
            noncarrier_mask = e <= 0
            event_mask = y == 1
            nonevent_mask = y == 0

            # stricter pre-checks
            if np.nanstd(e) <= 0:
                n_fail_precheck += 1
                continue
            if np.nanstd(m) <= 0:
                n_fail_precheck += 1
                continue
            if np.sum(event_mask) <= 0:
                n_fail_precheck += 1
                continue
            if np.sum(nonevent_mask) <= 0:
                n_fail_precheck += 1
                continue
            if np.sum(carrier_mask) <= 0:
                n_fail_precheck += 1
                continue
            if np.sum(noncarrier_mask) <= 0:
                n_fail_precheck += 1
                continue
            if np.sum(carrier_mask & event_mask) <= 0:
                n_fail_precheck += 1
                continue
            if np.sum(carrier_mask & nonevent_mask) <= 0:
                n_fail_precheck += 1
                continue
            if np.any(~np.isfinite(t)) or np.any(t <= 0):
                n_fail_precheck += 1
                continue

            med_res = fit_ols_mediator(bdat, covariates)
            dir_res = fit_phreg_direct(bdat, covariates)

            a = med_res["beta_EM_individual"]
            b = dir_res["beta_MY_conditional"]
            ab = a * b

            if (not np.isfinite(a)) or (not np.isfinite(b)) or (not np.isfinite(ab)):
                n_fail_nonfinite += 1
                continue

            boot_vals.append(float(ab))

        except Exception:
            n_fail_model += 1
            continue

    boot_vals = np.asarray(boot_vals, dtype=float)
    boot_vals = boot_vals[np.isfinite(boot_vals)]

    n_success = len(boot_vals)
    success_rate = n_success / n_boot if n_boot > 0 else np.nan

    if n_success < min_boot_success:
        return {
            "mediation_test_success": 0,
            "n_boot_requested": n_boot,
            "n_boot_success": n_success,
            "boot_success_rate": success_rate,
            "n_boot_fail_precheck": n_fail_precheck,
            "n_boot_fail_model": n_fail_model,
            "n_boot_fail_nonfinite": n_fail_nonfinite,
            "indirect_logHR": np.nan,
            "indirect_hr": np.nan,
            "ci95_l_indirect_logHR": np.nan,
            "ci95_u_indirect_logHR": np.nan,
            "ci95_l_indirect_hr": np.nan,
            "ci95_u_indirect_hr": np.nan,
            "indirect_p_boot": np.nan,
            "indirect_boot_mean": np.nan,
            "indirect_boot_median": np.nan,
            "indirect_boot_sd": np.nan,
            "fail_reason_boot": "insufficient_successful_finite_bootstrap_replicates"
        }

    ci_l = float(np.quantile(boot_vals, 0.025))
    ci_u = float(np.quantile(boot_vals, 0.975))

    if (not np.isfinite(ci_l)) or (not np.isfinite(ci_u)):
        return {
            "mediation_test_success": 0,
            "n_boot_requested": n_boot,
            "n_boot_success": n_success,
            "boot_success_rate": success_rate,
            "n_boot_fail_precheck": n_fail_precheck,
            "n_boot_fail_model": n_fail_model,
            "n_boot_fail_nonfinite": n_fail_nonfinite,
            "indirect_logHR": np.nan,
            "indirect_hr": np.nan,
            "ci95_l_indirect_logHR": np.nan,
            "ci95_u_indirect_logHR": np.nan,
            "ci95_l_indirect_hr": np.nan,
            "ci95_u_indirect_hr": np.nan,
            "indirect_p_boot": np.nan,
            "indirect_boot_mean": np.nan,
            "indirect_boot_median": np.nan,
            "indirect_boot_sd": np.nan,
            "fail_reason_boot": "nonfinite_bootstrap_ci"
        }

    # empirical two-sided p-value based on sign
    n_nonpos = int(np.sum(boot_vals <= 0))
    n_nonneg = int(np.sum(boot_vals >= 0))
    p_emp = (2 * min(n_nonpos, n_nonneg) + 1) / (n_success + 1)
    p_emp = min(float(p_emp), 1.0)

    return {
        "mediation_test_success": 1,
        "n_boot_requested": n_boot,
        "n_boot_success": n_success,
        "boot_success_rate": success_rate,
        "n_boot_fail_precheck": n_fail_precheck,
        "n_boot_fail_model": n_fail_model,
        "n_boot_fail_nonfinite": n_fail_nonfinite,
        "ci95_l_indirect_logHR": ci_l,
        "ci95_u_indirect_logHR": ci_u,
        "ci95_l_indirect_hr": float(np.exp(ci_l)),
        "ci95_u_indirect_hr": float(np.exp(ci_u)),
        "indirect_p_boot": p_emp,
        "indirect_boot_mean": float(np.mean(boot_vals)),
        "indirect_boot_median": float(np.median(boot_vals)),
        "indirect_boot_sd": float(np.std(boot_vals, ddof=1)) if n_success > 1 else np.nan,
        "fail_reason_boot": ""
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run bootstrap-based primary mediation analysis for Tier1+Tier2 trio run set"
    )
    parser.add_argument("--runset_tsv", required=True, help="03_runset_primary.tsv")
    parser.add_argument("--outdir", required=True, help="Output directory for primary mediation")
    parser.add_argument("--table_root", default=None, help="Root for relative table_path values; default is runset directory")
    parser.add_argument("--table_path_prefix_from", default=None, help="Optional old absolute path prefix to remap")
    parser.add_argument("--table_path_prefix_to", default=None, help="Replacement absolute path prefix")
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--n_chunks", type=int, default=1)
    parser.add_argument(
        "--covariates",
        nargs="+",
        default=DEFAULT_COVS,
        help="Covariates to adjust"
    )

    # same feasibility thresholds as safety gate
    parser.add_argument("--min_complete_case", type=int, default=200)
    parser.add_argument("--min_events_complete_case", type=int, default=20)
    parser.add_argument("--min_carriers_complete_case", type=int, default=10)
    parser.add_argument("--min_noncarriers_complete_case", type=int, default=200)
    parser.add_argument("--min_carrier_events_complete_case", type=int, default=2)
    parser.add_argument("--min_carrier_nonevents_complete_case", type=int, default=5)
    parser.add_argument("--min_M_sd", type=float, default=1e-8)

    # bootstrap params
    parser.add_argument("--n_boot", type=int, default=200)
    parser.add_argument("--min_boot_success", type=int, default=100)
    parser.add_argument("--seed_base", type=int, default=20260416)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    out_main = os.path.join(args.outdir, f"01_mediation.chunk{args.chunk_id}.tsv.gz")
    out_skip = os.path.join(args.outdir, f"01_mediation.chunk{args.chunk_id}.skipped.tsv.gz")
    out_qc = os.path.join(args.outdir, f"01_mediation.chunk{args.chunk_id}.qc.tsv")

    runset = pd.read_csv(
        args.runset_tsv,
        sep="\t",
        dtype={"trio_id": str, "gene": str, "protein": str, "icd3": str, "tier_label": str, "table_path": str}
    )

    required_runset = {"trio_id", "gene", "protein", "icd3", "tier_label", "table_path"}
    miss = required_runset - set(runset.columns)
    if miss:
        raise ValueError(f"runset_tsv missing required columns: {sorted(miss)}")

    runset = runset.reset_index(drop=True).copy()
    idx_keep = balanced_chunk_indices(len(runset), args.n_chunks, args.chunk_id)
    run_chunk = runset.loc[idx_keep].copy().reset_index(drop=True)

    if run_chunk.empty:
        pd.DataFrame().to_csv(out_main, sep="\t", index=False, compression="gzip")
        pd.DataFrame().to_csv(out_skip, sep="\t", index=False, compression="gzip")
        pd.DataFrame().to_csv(out_qc, sep="\t", index=False)
        print("[INFO] no trio assigned to this chunk")
        return

    results = []
    skipped = []

    for i, row in run_chunk.iterrows():
        trio_id = str(row["trio_id"])
        gene = str(row["gene"])
        protein = str(row["protein"])
        icd3 = str(row["icd3"])
        tier_label = str(row["tier_label"])
        table_path = os.path.expanduser(str(row["table_path"]))
        if args.table_path_prefix_from:
            if not args.table_path_prefix_to:
                raise ValueError("--table_path_prefix_to is required with --table_path_prefix_from")
            old_prefix = os.path.normpath(os.path.expanduser(args.table_path_prefix_from))
            normalized = os.path.normpath(table_path)
            if normalized == old_prefix or normalized.startswith(old_prefix + os.sep):
                table_path = os.path.join(args.table_path_prefix_to, os.path.relpath(normalized, old_prefix))
        elif args.table_path_prefix_to:
            raise ValueError("--table_path_prefix_from is required with --table_path_prefix_to")
        if not os.path.isabs(table_path):
            table_path = os.path.join(args.table_root or os.path.dirname(os.path.abspath(args.runset_tsv)), table_path)
        table_path = os.path.abspath(table_path)

        base_row = {
            "trio_id": trio_id,
            "gene": gene,
            "protein": protein,
            "icd3": icd3,
            "tier_label": tier_label,
            "table_path": table_path,
            "mediator_model_success": 0,
            "total_model_success": 0,
            "direct_model_success": 0,
            "mediation_test_success": 0,
            "model_success": 0,
            "fail_reason": ""
        }

        if not os.path.exists(table_path):
            skip = base_row.copy()
            skip["fail_reason"] = "table_path_not_found"
            skipped.append(skip)
            continue

        try:
            dat = pd.read_csv(table_path, sep="\t", dtype={"sample": str})
        except Exception as e:
            skip = base_row.copy()
            skip["fail_reason"] = f"read_failed::{str(e)[:200]}"
            skipped.append(skip)
            continue

        required_cols = {"E", "M", "event", "time_days", "eligible_for_incident"} | set(args.covariates)
        miss_cols = required_cols - set(dat.columns)
        if miss_cols:
            skip = base_row.copy()
            skip["fail_reason"] = f"missing_columns::{','.join(sorted(miss_cols))}"
            skipped.append(skip)
            continue

        # numeric coercion
        for c in ["E", "M", "event", "time_days", "eligible_for_incident"] + args.covariates:
            dat[c] = pd.to_numeric(dat[c], errors="coerce")

        # modeling complete-case
        dat["complete_case_model"] = (
            dat["eligible_for_incident"].eq(1) &
            dat["E"].notna() &
            dat["M"].notna() &
            dat["event"].isin([0, 1]) &
            dat["time_days"].notna() &
            (dat["time_days"] > 0) &
            dat[args.covariates].notna().all(axis=1)
        ).astype(int)

        cc = dat[dat["complete_case_model"] == 1].copy()

        n_total_rows = int(dat.shape[0])
        n_complete_case = int(cc.shape[0])

        row_out = base_row.copy()
        row_out.update({
            "n_total_rows": n_total_rows,
            "n_complete_case": n_complete_case,
        })

        if n_complete_case == 0:
            row_out["fail_reason"] = "no_complete_case_rows"
            results.append(row_out)
            continue

        e = cc["E"].to_numpy()
        y = cc["event"].astype(int).to_numpy()

        carrier_mask = e > 0
        noncarrier_mask = ~carrier_mask
        event_mask = y == 1
        nonevent_mask = y == 0

        n_events_cc = int(event_mask.sum())
        n_carriers_cc = int(carrier_mask.sum())
        n_noncarriers_cc = int(noncarrier_mask.sum())
        n_carrier_events_cc = int((carrier_mask & event_mask).sum())
        n_carrier_nonevents_cc = int((carrier_mask & nonevent_mask).sum())
        M_sd_cc = float(cc["M"].std(skipna=True)) if cc["M"].notna().sum() > 1 else np.nan

        row_out.update({
            "n_events_complete_case": n_events_cc,
            "n_carriers_complete_case": n_carriers_cc,
            "n_noncarriers_complete_case": n_noncarriers_cc,
            "n_carrier_events_complete_case": n_carrier_events_cc,
            "n_carrier_nonevents_complete_case": n_carrier_nonevents_cc,
            "M_sd_complete_case": M_sd_cc,
        })

        fail_reasons = []
        if n_complete_case < args.min_complete_case:
            fail_reasons.append("low_complete_case")
        if n_events_cc < args.min_events_complete_case:
            fail_reasons.append("low_events_complete_case")
        if n_carriers_cc < args.min_carriers_complete_case:
            fail_reasons.append("low_carriers_complete_case")
        if n_noncarriers_cc < args.min_noncarriers_complete_case:
            fail_reasons.append("low_noncarriers_complete_case")
        if n_carrier_events_cc < args.min_carrier_events_complete_case:
            fail_reasons.append("low_carrier_events_complete_case")
        if n_carrier_nonevents_cc < args.min_carrier_nonevents_complete_case:
            fail_reasons.append("low_carrier_nonevents_complete_case")
        if (not np.isfinite(M_sd_cc)) or (M_sd_cc <= args.min_M_sd):
            fail_reasons.append("low_M_variation_complete_case")
        if np.nanstd(cc["E"].to_numpy()) <= 0:
            fail_reasons.append("no_E_variation_complete_case")

        if fail_reasons:
            row_out["fail_reason"] = ";".join(fail_reasons)
            results.append(row_out)
            continue

        # formal models
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                med_res = fit_ols_mediator(cc, args.covariates)
                total_res = fit_phreg_total(cc, args.covariates)
                direct_res = fit_phreg_direct(cc, args.covariates)

                # full-sample indirect point estimate on log-HR scale
                indirect_logHR = med_res["beta_EM_individual"] * direct_res["beta_MY_conditional"]
                if not np.isfinite(indirect_logHR):
                    raise ValueError("nonfinite_indirect_logHR_full_sample")

                boot_res = bootstrap_indirect_effect(
                    cc,
                    covariates=args.covariates,
                    n_boot=args.n_boot,
                    min_boot_success=args.min_boot_success,
                    seed=stable_seed_from_string(trio_id, args.seed_base)
                )

            row_out.update(med_res)
            row_out.update(total_res)
            row_out.update(direct_res)
            row_out.update(boot_res)

            row_out["indirect_logHR"] = float(indirect_logHR)
            row_out["indirect_hr"] = float(np.exp(indirect_logHR)) if np.isfinite(indirect_logHR) else np.nan

            if not np.isfinite(row_out["indirect_hr"]):
                raise ValueError("nonfinite_indirect_hr_full_sample")

            beta_total = row_out.get("beta_EY_total", np.nan)
            if np.isfinite(beta_total) and abs(beta_total) > 1e-12 and np.isfinite(indirect_logHR):
                row_out["proportion_mediated_logscale"] = indirect_logHR / beta_total
            else:
                row_out["proportion_mediated_logscale"] = np.nan

            row_out["model_success"] = int(
                row_out.get("mediator_model_success", 0) == 1 and
                row_out.get("total_model_success", 0) == 1 and
                row_out.get("direct_model_success", 0) == 1 and
                row_out.get("mediation_test_success", 0) == 1 and
                np.isfinite(row_out.get("indirect_logHR", np.nan)) and
                np.isfinite(row_out.get("indirect_hr", np.nan)) and
                np.isfinite(row_out.get("ci95_l_indirect_logHR", np.nan)) and
                np.isfinite(row_out.get("ci95_u_indirect_logHR", np.nan)) and
                np.isfinite(row_out.get("ci95_l_indirect_hr", np.nan)) and
                np.isfinite(row_out.get("ci95_u_indirect_hr", np.nan))
            )

            if row_out["model_success"] != 1 and row_out.get("fail_reason", "") == "":
                row_out["fail_reason"] = row_out.get("fail_reason_boot", "mediation_test_failed_or_incomplete")

        except Exception as e:
            row_out["fail_reason"] = f"model_failed::{str(e)[:300]}"

        results.append(row_out)

        if (i + 1) % 25 == 0 or (i + 1) == len(run_chunk):
            print(f"[INFO] processed {i+1}/{len(run_chunk)} trios in chunk {args.chunk_id}", flush=True)

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df = res_df.sort_values(["tier_label", "gene", "protein", "icd3"]).copy()
    res_df.to_csv(out_main, sep="\t", index=False, compression="gzip")

    skip_df = pd.DataFrame(skipped)
    if not skip_df.empty:
        skip_df = skip_df.sort_values(["fail_reason", "gene", "protein", "icd3"]).copy()
    skip_df.to_csv(out_skip, sep="\t", index=False, compression="gzip")

    qc_rows = [
        {"metric": "chunk_id", "value": args.chunk_id},
        {"metric": "n_chunks", "value": args.n_chunks},
        {"metric": "n_trios_assigned", "value": run_chunk.shape[0]},
        {"metric": "n_result_rows", "value": res_df.shape[0]},
        {"metric": "n_skipped_rows", "value": skip_df.shape[0]},
    ]

    if not res_df.empty:
        qc_rows.extend([
            {"metric": "n_model_success", "value": int((res_df["model_success"] == 1).sum())},
            {"metric": "n_model_failed", "value": int((res_df["model_success"] != 1).sum())},
            {"metric": "median_boot_success_rate", "value": float(pd.to_numeric(res_df.get("boot_success_rate", np.nan), errors="coerce").median())},
            {"metric": "n_indirect_p_boot_lt_0.05", "value": int((pd.to_numeric(res_df.get("indirect_p_boot", np.nan), errors="coerce") < 0.05).sum())},
            {"metric": "n_rows_with_complete_indirect_hr_ci", "value": int(
                (
                    pd.to_numeric(res_df.get("ci95_l_indirect_hr", np.nan), errors="coerce").notna() &
                    pd.to_numeric(res_df.get("ci95_u_indirect_hr", np.nan), errors="coerce").notna()
                ).sum()
            )},
        ])

        for tier in ["Tier1", "Tier2"]:
            sub = res_df[res_df["tier_label"] == tier]
            qc_rows.append({"metric": f"n_{tier}_rows", "value": sub.shape[0]})
            qc_rows.append({"metric": f"n_{tier}_model_success", "value": int((sub["model_success"] == 1).sum()) if not sub.empty else 0})

        fail_counts = res_df.loc[res_df["model_success"] != 1, "fail_reason"].astype(str).value_counts()
        for reason, n in fail_counts.items():
            if reason.strip():
                qc_rows.append({"metric": f"fail::{reason}", "value": int(n)})

    if not skip_df.empty and "fail_reason" in skip_df.columns:
        skip_counts = skip_df["fail_reason"].astype(str).value_counts()
        for reason, n in skip_counts.items():
            qc_rows.append({"metric": f"skipped::{reason}", "value": int(n)})

    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(out_qc, sep="\t", index=False)

    print(f"[OK] mediation results written to: {out_main}")
    print(f"[OK] skipped rows written to: {out_skip}")
    print(f"[OK] chunk qc written to: {out_qc}")
    print(f"[INFO] result_rows={res_df.shape[0]}, skipped_rows={skip_df.shape[0]}")
    print(f"[INFO] model_success={int((res_df['model_success'] == 1).sum()) if not res_df.empty else 0}")


if __name__ == "__main__":
    main()