#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
prefix="${MY_OUTDIR}/inc_chunks/inc_proteome_phenome"
run_python "${SCRIPT_DIR}/05_clean_inc_fdr_results.py" --all_tsv "${prefix}.all.tsv" \
  --orig_fdr_tsv "${prefix}.fdr_q_lt${ALPHA}.tsv" \
  --out_cleaned_tsv "${prefix}.fdr_q_lt${ALPHA}_cleaned.tsv" \
  --out_anomaly_tsv "${prefix}.fdr_q_lt${ALPHA}_cleaned.anomalies.tsv" \
  --out_qc_txt "${prefix}.fdr_q_lt${ALPHA}_cleaned.qc.txt" \
  --beta_abs_hard_cut "${QC_BETA_ABS}" --se_hard_cut "${QC_SE}" \
  --tol_log_hr "${QC_LOG_TOL}" --tol_ci "${QC_CI_TOL}" --z_underflow_min "${QC_UNDERFLOW_Z}" "$@"
