#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.tools.sm_exceptions import PerfectSeparationError, ConvergenceWarning


TINY_P = np.nextafter(0, 1)


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def read_lines(path):
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]


def split_covariates(x: str):
    return [i.strip() for i in x.split(",") if i.strip()]


def read_counts_file(path: str, mode: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype={"icd3": str})
    if "icd3" not in df.columns:
        raise ValueError(f"{path} must contain column 'icd3'")

    if mode == "prevalent":
        case_candidates = ["n_cases", "cases", "case_n"]
        ctrl_candidates = ["n_controls", "controls", "control_n", "n_noncases"]
        burden_candidates = case_candidates
    else:
        case_candidates = ["n_events", "events", "event_n"]
        ctrl_candidates = ["n_atrisk", "at_risk", "n_riskset", "riskset_n"]
        burden_candidates = case_candidates

    case_col = next((c for c in case_candidates if c in df.columns), None)
    ctrl_col = next((c for c in ctrl_candidates if c in df.columns), None)
    burden_col = next((c for c in burden_candidates if c in df.columns), None)

    if case_col is None or ctrl_col is None:
        raise ValueError(f"Cannot find required count columns in {path} for mode={mode}")

    out = df[["icd3", case_col, ctrl_col]].copy()
    out.columns = ["icd3", "n_case_like", "n_ctrl_like"]
    out["sort_key"] = df[burden_col].to_numpy()
    out = out.drop_duplicates("icd3")
    return out


def balanced_chunk(items, weights, n_chunks, chunk_id):
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    if not (0 <= chunk_id < n_chunks):
        raise ValueError("chunk_id out of range")

    order = np.argsort(-np.asarray(weights))
    bins = [[] for _ in range(n_chunks)]
    for rank, idx in enumerate(order):
        bins[rank % n_chunks].append(items[idx])
    return bins[chunk_id]


def load_target_icd3(counts_path, mode, n_chunks, chunk_id,
                     min_case_like, min_ctrl_like, icd3_keep_path=None):
    counts = read_counts_file(counts_path, mode)
    counts = counts[
        (counts["n_case_like"] >= min_case_like) &
        (counts["n_ctrl_like"] >= min_ctrl_like)
    ].copy()

    if icd3_keep_path:
        keep = set(read_lines(icd3_keep_path))
        counts = counts[counts["icd3"].isin(keep)].copy()

    counts = counts.sort_values(["sort_key", "icd3"], ascending=[False, True]).reset_index(drop=True)
    items = counts["icd3"].tolist()
    weights = counts["sort_key"].tolist()
    target = balanced_chunk(items, weights, n_chunks, chunk_id)
    return counts, target


def read_icd_subset(icd_path, target_icd3):
    target_icd3 = set(target_icd3)
    dfs = []
    usecols = ["sample", "icd3", "first_date"]

    for chunk in pd.read_csv(
        icd_path,
        sep="\t",
        dtype={"sample": str, "icd3": str},
        usecols=usecols,
        chunksize=1_000_000,
    ):
        sub = chunk[chunk["icd3"].isin(target_icd3)].copy()
        if not sub.empty:
            dfs.append(sub)

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.DataFrame(columns=usecols)

    df["first_date"] = pd.to_datetime(df["first_date"], errors="coerce")
    df = df.dropna(subset=["sample", "icd3"]).copy()
    return df


def prepare_master_dataframe(sample_file, covariate_file, followup_file, covariates):
    samples = read_lines(sample_file)
    master = pd.DataFrame({"sample": samples})

    cov = pd.read_csv(covariate_file, sep="\t", dtype={"sample": str})
    need_cov_cols = ["sample"] + covariates
    missing_cov = [c for c in need_cov_cols if c not in cov.columns]
    if missing_cov:
        raise ValueError(f"Missing covariate columns in {covariate_file}: {missing_cov}")

    cov = cov[need_cov_cols].copy()
    for c in covariates:
        cov[c] = pd.to_numeric(cov[c], errors="coerce")

    fu = pd.read_csv(
        followup_file,
        sep="\t",
        dtype={"sample": str},
        usecols=["sample", "baseline_date", "death_date", "censor_date"],
    )
    for c in ["baseline_date", "death_date", "censor_date"]:
        fu[c] = pd.to_datetime(fu[c], errors="coerce")

    master = master.merge(cov, on="sample", how="left")
    master = master.merge(fu, on="sample", how="left")
    return samples, master


def build_prevalent_arrays(master_df, disease_df):
    first_map = disease_df.drop_duplicates("sample").set_index("sample")["first_date"]
    tmp = master_df[["sample", "baseline_date"]].copy()
    tmp["first_date"] = tmp["sample"].map(first_map)

    eligible = tmp["baseline_date"].notna()
    y = ((tmp["first_date"].notna()) & (tmp["first_date"] <= tmp["baseline_date"])).astype(np.int8)
    return eligible.to_numpy(), y.to_numpy()


def build_incident_arrays(master_df, disease_df):
    first_map = disease_df.drop_duplicates("sample").set_index("sample")["first_date"]
    tmp = master_df[["sample", "baseline_date", "death_date", "censor_date"]].copy()
    tmp["first_date"] = tmp["sample"].map(first_map)

    end_date = tmp[["death_date", "censor_date"]].min(axis=1)
    has_baseline = tmp["baseline_date"].notna()
    has_end = end_date.notna()

    prevalent = (tmp["first_date"].notna()) & (tmp["first_date"] <= tmp["baseline_date"])
    eligible = has_baseline & has_end & (end_date > tmp["baseline_date"]) & (~prevalent)

    event = (
        eligible &
        tmp["first_date"].notna() &
        (tmp["first_date"] > tmp["baseline_date"]) &
        (tmp["first_date"] <= end_date)
    )

    exit_date = np.where(
        event,
        tmp["first_date"].to_numpy(dtype="datetime64[ns]"),
        end_date.to_numpy(dtype="datetime64[ns]")
    )
    baseline = tmp["baseline_date"].to_numpy(dtype="datetime64[ns]")
    time_days = ((exit_date - baseline) / np.timedelta64(1, "D")).astype(float)
    time_days = np.where(eligible.to_numpy(), time_days, np.nan)

    return eligible.to_numpy(), event.astype(np.int8).to_numpy(), time_days


def sanitize_p(p: float) -> float:
    if not np.isfinite(p) or p <= 0:
        return float(TINY_P)
    if p > 1:
        return 1.0
    return float(p)


def safe_exp(x: float) -> float:
    if not np.isfinite(x):
        raise RuntimeError("nonfinite_beta")
    # exp(700) 仍在 double 可表示范围内，以上容易 overflow
    if x > 700:
        raise RuntimeError("beta_too_large_positive_for_exp")
    if x < -700:
        return 0.0
    return float(math.exp(x))


def crude_direction(exposed_event, exposed_nonevent, unexposed_event, unexposed_nonevent):
    ex_total = exposed_event + exposed_nonevent
    un_total = unexposed_event + unexposed_nonevent
    if ex_total <= 0 or un_total <= 0:
        return 0

    ex_rate = exposed_event / ex_total
    un_rate = unexposed_event / un_total

    if ex_rate > un_rate:
        return 1
    elif ex_rate < un_rate:
        return -1
    else:
        return 0


def validate_effect(beta, effect, ci_l, ci_u, p, max_abs_beta):
    vals = [beta, effect, ci_l, ci_u, p]
    if not all(np.isfinite(v) for v in vals):
        raise RuntimeError("nonfinite_estimate")

    if effect <= 0 or ci_l <= 0 or ci_u <= 0:
        raise RuntimeError("nonpositive_effect_or_ci")

    if ci_l > effect or effect > ci_u:
        raise RuntimeError("effect_outside_ci")

    if abs(beta) > max_abs_beta:
        raise RuntimeError(f"abs_beta_exceeds_threshold:{beta:.6f}")


def fit_prevalent_logit(y, x_carrier, cov_df, max_abs_beta):
    X = cov_df.copy()
    X.insert(0, "carrier", x_carrier.astype(float))
    X = sm.add_constant(X, has_constant="add")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            res = sm.Logit(y, X).fit(disp=0, maxiter=200)
        converged = getattr(res, "mle_retvals", {}).get("converged", True)
        if not converged:
            raise RuntimeError("logit_not_converged")

        beta = float(res.params["carrier"])
        se = float(res.bse["carrier"])
        z = float(res.tvalues["carrier"])
        p = sanitize_p(float(res.pvalues["carrier"]))
        model_used = "logit"
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError, RuntimeError) as e:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                res = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=200, disp=0)
            beta = float(res.params["carrier"])
            se = float(res.bse["carrier"])
            z = float(res.tvalues["carrier"])
            p = sanitize_p(float(res.pvalues["carrier"]))
            model_used = "glm_binomial"
        except Exception as e2:
            raise RuntimeError(f"logit_failed: {e}; glm_failed: {e2}")

    or_val = safe_exp(beta)
    ci_l = safe_exp(beta - 1.96 * se)
    ci_u = safe_exp(beta + 1.96 * se)
    validate_effect(beta, or_val, ci_l, ci_u, p, max_abs_beta)
    return beta, se, z, p, or_val, ci_l, ci_u, model_used


def fit_incident_cox(time_days, event, x_carrier, cov_df, max_abs_beta):
    X = cov_df.copy()
    X.insert(0, "carrier", x_carrier.astype(float))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model = PHReg(
            endog=time_days.astype(float),
            exog=X.astype(float),
            status=event.astype(int),
            ties="breslow",
        )
        res = model.fit(disp=0)

    beta = float(res.params[0])
    se = float(res.bse[0])
    z = float(res.tvalues[0])
    p = sanitize_p(float(res.pvalues[0]))

    hr = safe_exp(beta)
    ci_l = safe_exp(beta - 1.96 * se)
    ci_u = safe_exp(beta + 1.96 * se)
    validate_effect(beta, hr, ci_l, ci_u, p, max_abs_beta)
    return beta, se, z, p, hr, ci_l, ci_u, "cox_ph"


def append_skip(skipped, icd3, gene, reason, **kwargs):
    row = {"icd3": icd3, "gene": gene, "reason": reason}
    row.update(kwargs)
    skipped.append(row)



def result_columns(mode):
    common = ["analysis", "icd3", "gene", "n_used", "n_carriers_used", "n_noncarriers_used", "beta", "se", "z", "p", "ci95_l", "ci95_u", "model"]
    extra = ["n_cases", "n_controls", "n_carrier_cases", "n_carrier_controls", "or"] if mode == "prevalent" else ["n_events", "n_atrisk", "n_carrier_events", "n_carrier_nonevents", "hr"]
    return common + extra


def main():
    ap = argparse.ArgumentParser(
        description="Step5C: gene carrier/burden -> disease association (prevalent or incident)"
    )
    ap.add_argument("--mode", required=True, choices=["prevalent", "incident"])
    ap.add_argument("--sample-file", required=True)
    ap.add_argument("--gene-file", required=True)
    ap.add_argument("--carrier-matrix", required=True)
    ap.add_argument("--carrier-counts", required=True)
    ap.add_argument("--covariate-file", required=True)
    ap.add_argument("--followup-file", required=True)
    ap.add_argument("--icd-file", required=True)
    ap.add_argument("--counts-file", required=True, help="Step5A rebuilt prev/inc icd3 counts file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-skip", default=None)
    ap.add_argument(
        "--covariates",
        default="age,sex,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10"
    )
    ap.add_argument("--chunk-id", type=int, default=0)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--icd3-keep", default=None)
    ap.add_argument("--gene-exclude", default=None)

    ap.add_argument("--min-carriers-global", type=int, default=5)
    ap.add_argument("--max-carrier-rate-global", type=float, default=0.2)
    ap.add_argument("--min-case-like", type=int, default=20)
    ap.add_argument("--min-ctrl-like", type=int, default=20)
    ap.add_argument("--min-carriers-used", type=int, default=3)
    ap.add_argument("--min-noncarriers-used", type=int, default=30)

    # 新增：避免 zero-cell / complete separation
    ap.add_argument("--min-carrier-cases", type=int, default=1)
    ap.add_argument("--min-carrier-controls", type=int, default=1)
    ap.add_argument("--min-carrier-events", type=int, default=1)
    ap.add_argument("--min-carrier-nonevents", type=int, default=1)

    # 新增：避免极端不稳定估计
    ap.add_argument("--max-abs-beta", type=float, default=20.0)

    args = ap.parse_args()
    covariates = split_covariates(args.covariates)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    if args.out_skip:
        os.makedirs(Path(args.out_skip).parent, exist_ok=True)

    log("[1/6] load genes and carrier QC info ...")
    genes = read_lines(args.gene_file)
    gene_counts = pd.read_csv(args.carrier_counts, sep="\t", dtype={"gene": str})

    need_cols = {"gene", "n_carriers", "carrier_rate", "n_noncarriers"}
    if not need_cols.issubset(set(gene_counts.columns)):
        raise ValueError(f"{args.carrier_counts} must contain: {sorted(need_cols)}")

    gene_counts = gene_counts.drop_duplicates("gene").set_index("gene")
    gene_exclude = set(read_lines(args.gene_exclude)) if args.gene_exclude else set()

    keep_gene_idx = []
    keep_genes = []
    for j, g in enumerate(genes):
        if g in gene_exclude:
            continue
        if g not in gene_counts.index:
            continue

        row = gene_counts.loc[g]
        if int(row["n_carriers"]) < args.min_carriers_global:
            continue
        if float(row["carrier_rate"]) > args.max_carrier_rate_global:
            continue
        if int(row["n_noncarriers"]) < args.min_noncarriers_used:
            continue

        keep_gene_idx.append(j)
        keep_genes.append(g)

    if not keep_gene_idx:
        raise RuntimeError("No genes passed global QC filters.")

    log(f"  genes kept: {len(keep_genes)} / {len(genes)}")

    log("[2/6] load disease counts and decide chunk ...")
    _, target_icd3 = load_target_icd3(
        counts_path=args.counts_file,
        mode=args.mode,
        n_chunks=args.n_chunks,
        chunk_id=args.chunk_id,
        min_case_like=args.min_case_like,
        min_ctrl_like=args.min_ctrl_like,
        icd3_keep_path=args.icd3_keep,
    )

    if not target_icd3:
        log("No diseases assigned to this chunk after filtering. Writing empty file.")
        pd.DataFrame(columns=result_columns(args.mode)).to_csv(args.out, sep="\t", index=False)
        if args.out_skip:
            pd.DataFrame(columns=["icd3", "gene", "reason", "detail"]).to_csv(args.out_skip, sep="\t", index=False)
        return

    log(f"  diseases in this chunk: {len(target_icd3)}")

    log("[3/6] load matrix / covariates / follow-up ...")
    samples, master = prepare_master_dataframe(
        args.sample_file,
        args.covariate_file,
        args.followup_file,
        covariates
    )

    E = sparse.load_npz(args.carrier_matrix).tocsr()
    if E.shape[0] != len(samples):
        raise ValueError(f"Sample size mismatch: matrix has {E.shape[0]}, sample file has {len(samples)}")
    if E.shape[1] != len(genes):
        raise ValueError(f"Gene size mismatch: matrix has {E.shape[1]}, gene file has {len(genes)}")

    E = E[:, keep_gene_idx].astype(np.int8).toarray()

    log("[4/6] load ICD first-occurrence subset ...")
    icd_df = read_icd_subset(args.icd_file, target_icd3)
    icd_group = {k: v[["sample", "first_date"]].copy() for k, v in icd_df.groupby("icd3", sort=False)}

    results = []
    skipped = []

    log("[5/6] run models ...")
    for d_i, icd3 in enumerate(target_icd3, start=1):
        disease_df = icd_group.get(icd3, pd.DataFrame(columns=["sample", "first_date"]))

        if args.mode == "prevalent":
            eligible, y_all = build_prevalent_arrays(master, disease_df)
            base_mask = eligible.copy()
        else:
            eligible, event_all, time_all = build_incident_arrays(master, disease_df)
            base_mask = eligible & np.isfinite(time_all)

        cov_ok = master[covariates].notna().all(axis=1).to_numpy()
        base_mask = base_mask & cov_ok

        if args.mode == "prevalent":
            n_case_like = int(y_all[base_mask].sum())
            n_ctrl_like = int(base_mask.sum() - n_case_like)
        else:
            n_case_like = int(event_all[base_mask].sum())
            n_ctrl_like = int(base_mask.sum())

        if n_case_like < args.min_case_like or n_ctrl_like < args.min_ctrl_like:
            append_skip(
                skipped, icd3, ".", "disease_failed_final_threshold",
                n_case_like=n_case_like, n_ctrl_like=n_ctrl_like
            )
            continue

        cov_base = master.loc[base_mask, covariates].copy()
        if cov_base.empty:
            append_skip(
                skipped, icd3, ".", "empty_covariate_mask",
                n_case_like=n_case_like, n_ctrl_like=n_ctrl_like
            )
            continue

        if args.mode == "prevalent":
            y = y_all[base_mask]
            total_cases = int(y.sum())
            total_controls = int((y == 0).sum())
        else:
            event = event_all[base_mask]
            time_days = time_all[base_mask]
            total_events = int(event.sum())
            total_nonevents = int((event == 0).sum())

        for j, gene in enumerate(keep_genes):
            x = E[base_mask, j]
            n_carriers_used = int(x.sum())
            n_noncarriers_used = int(len(x) - n_carriers_used)

            if n_carriers_used < args.min_carriers_used:
                append_skip(
                    skipped, icd3, gene, "low_carriers_used",
                    n_case_like=n_case_like, n_ctrl_like=n_ctrl_like,
                    n_carriers_used=n_carriers_used,
                    n_noncarriers_used=n_noncarriers_used
                )
                continue

            if n_noncarriers_used < args.min_noncarriers_used:
                append_skip(
                    skipped, icd3, gene, "low_noncarriers_used",
                    n_case_like=n_case_like, n_ctrl_like=n_ctrl_like,
                    n_carriers_used=n_carriers_used,
                    n_noncarriers_used=n_noncarriers_used
                )
                continue

            try:
                if args.mode == "prevalent":
                    n_carrier_case = int(((x == 1) & (y == 1)).sum())
                    n_carrier_ctrl = int(((x == 1) & (y == 0)).sum())
                    n_noncarrier_case = total_cases - n_carrier_case
                    n_noncarrier_ctrl = total_controls - n_carrier_ctrl

                    # 新增：零格 / 极低格过滤，防止 complete separation
                    if n_carrier_case < args.min_carrier_cases:
                        append_skip(
                            skipped, icd3, gene, "low_carrier_cases",
                            n_cases=total_cases, n_controls=total_controls,
                            n_carriers_used=n_carriers_used,
                            n_noncarriers_used=n_noncarriers_used,
                            n_carrier_cases=n_carrier_case,
                            n_carrier_controls=n_carrier_ctrl
                        )
                        continue

                    if n_carrier_ctrl < args.min_carrier_controls:
                        append_skip(
                            skipped, icd3, gene, "low_carrier_controls",
                            n_cases=total_cases, n_controls=total_controls,
                            n_carriers_used=n_carriers_used,
                            n_noncarriers_used=n_noncarriers_used,
                            n_carrier_cases=n_carrier_case,
                            n_carrier_controls=n_carrier_ctrl
                        )
                        continue

                    if n_noncarrier_case <= 0 or n_noncarrier_ctrl <= 0:
                        append_skip(
                            skipped, icd3, gene, "zero_cell_in_noncarriers_prevalent",
                            n_cases=total_cases, n_controls=total_controls,
                            n_carriers_used=n_carriers_used,
                            n_noncarriers_used=n_noncarriers_used,
                            n_carrier_cases=n_carrier_case,
                            n_carrier_controls=n_carrier_ctrl
                        )
                        continue

                    beta, se, z, p, eff, ci_l, ci_u, model_used = fit_prevalent_logit(
                        y, x, cov_base, args.max_abs_beta
                    )

                    exp_sign = crude_direction(
                        n_carrier_case, n_carrier_ctrl,
                        n_noncarrier_case, n_noncarrier_ctrl
                    )
                    if exp_sign != 0 and beta * exp_sign <= 0:
                        append_skip(
                            skipped, icd3, gene, "model_sign_inconsistent_with_crude_prevalent",
                            n_cases=total_cases, n_controls=total_controls,
                            n_carriers_used=n_carriers_used,
                            n_noncarriers_used=n_noncarriers_used,
                            n_carrier_cases=n_carrier_case,
                            n_carrier_controls=n_carrier_ctrl,
                            beta=beta, or_est=eff
                        )
                        continue

                    results.append({
                        "analysis": "prevalent",
                        "icd3": icd3,
                        "gene": gene,
                        "n_used": int(base_mask.sum()),
                        "n_cases": total_cases,
                        "n_controls": total_controls,
                        "n_carriers_used": n_carriers_used,
                        "n_noncarriers_used": n_noncarriers_used,
                        "n_carrier_cases": n_carrier_case,
                        "n_carrier_controls": n_carrier_ctrl,
                        "beta": beta,
                        "se": se,
                        "z": z,
                        "p": p,
                        "or": eff,
                        "ci95_l": ci_l,
                        "ci95_u": ci_u,
                        "model": model_used,
                    })

                else:
                    n_carrier_event = int(((x == 1) & (event == 1)).sum())
                    n_carrier_nonevent = int(((x == 1) & (event == 0)).sum())
                    n_noncarrier_event = total_events - n_carrier_event
                    n_noncarrier_nonevent = total_nonevents - n_carrier_nonevent

                    if n_carrier_event < args.min_carrier_events:
                        append_skip(
                            skipped, icd3, gene, "low_carrier_events",
                            n_events=total_events, n_atrisk=int(base_mask.sum()),
                            n_carriers_used=n_carriers_used,
                            n_noncarriers_used=n_noncarriers_used,
                            n_carrier_events=n_carrier_event,
                            n_carrier_nonevents=n_carrier_nonevent
                        )
                        continue

                    if n_carrier_nonevent < args.min_carrier_nonevents:
                        append_skip(
                            skipped, icd3, gene, "low_carrier_nonevents",
                            n_events=total_events, n_atrisk=int(base_mask.sum()),
                            n_carriers_used=n_carriers_used,
                            n_noncarriers_used=n_noncarriers_used,
                            n_carrier_events=n_carrier_event,
                            n_carrier_nonevents=n_carrier_nonevent
                        )
                        continue

                    if n_noncarrier_event <= 0 or n_noncarrier_nonevent <= 0:
                        append_skip(
                            skipped, icd3, gene, "zero_cell_in_noncarriers_incident",
                            n_events=total_events, n_atrisk=int(base_mask.sum()),
                            n_carriers_used=n_carriers_used,
                            n_noncarriers_used=n_noncarriers_used,
                            n_carrier_events=n_carrier_event,
                            n_carrier_nonevents=n_carrier_nonevent
                        )
                        continue

                    beta, se, z, p, eff, ci_l, ci_u, model_used = fit_incident_cox(
                        time_days, event, x, cov_base, args.max_abs_beta
                    )

                    exp_sign = crude_direction(
                        n_carrier_event, n_carrier_nonevent,
                        n_noncarrier_event, n_noncarrier_nonevent
                    )
                    if exp_sign != 0 and beta * exp_sign <= 0:
                        append_skip(
                            skipped, icd3, gene, "model_sign_inconsistent_with_crude_incident",
                            n_events=total_events, n_atrisk=int(base_mask.sum()),
                            n_carriers_used=n_carriers_used,
                            n_noncarriers_used=n_noncarriers_used,
                            n_carrier_events=n_carrier_event,
                            n_carrier_nonevents=n_carrier_nonevent,
                            beta=beta, hr_est=eff
                        )
                        continue

                    results.append({
                        "analysis": "incident",
                        "icd3": icd3,
                        "gene": gene,
                        "n_used": int(base_mask.sum()),
                        "n_events": total_events,
                        "n_atrisk": int(base_mask.sum()),
                        "n_carriers_used": n_carriers_used,
                        "n_noncarriers_used": n_noncarriers_used,
                        "n_carrier_events": n_carrier_event,
                        "n_carrier_nonevents": n_carrier_nonevent,
                        "beta": beta,
                        "se": se,
                        "z": z,
                        "p": p,
                        "hr": eff,
                        "ci95_l": ci_l,
                        "ci95_u": ci_u,
                        "model": model_used,
                    })

            except Exception as e:
                append_skip(
                    skipped, icd3, gene, f"model_failed: {type(e).__name__}",
                    detail=str(e),
                    n_case_like=n_case_like,
                    n_ctrl_like=n_ctrl_like,
                    n_carriers_used=n_carriers_used,
                    n_noncarriers_used=n_noncarriers_used
                )

        if d_i % 10 == 0 or d_i == len(target_icd3):
            log(f"  processed diseases: {d_i}/{len(target_icd3)} ; results={len(results)} ; skipped={len(skipped)}")

    log("[6/6] write outputs ...")
    res_df = pd.DataFrame(results) if results else pd.DataFrame(columns=result_columns(args.mode))
    if not res_df.empty:
        res_df = res_df.sort_values(["p", "icd3", "gene"], ascending=[True, True, True])
    res_df.to_csv(args.out, sep="\t", index=False, compression="infer")

    if args.out_skip:
        skip_df = pd.DataFrame(skipped) if skipped else pd.DataFrame(columns=["icd3", "gene", "reason", "detail"])
        if not skip_df.empty:
            skip_df = skip_df.sort_values(["icd3", "gene", "reason"], ascending=[True, True, True])
        skip_df.to_csv(args.out_skip, sep="\t", index=False, compression="infer")

    log("Done.")


if __name__ == "__main__":
    main()