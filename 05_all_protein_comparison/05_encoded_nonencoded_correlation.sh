#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
run_python "${SCRIPT_DIR}/05_encoded_nonencoded_correlation.py" \
  --correlation_runset "${ALL_PROTEIN_OUTDIR}/04_encoded_vs_best_nonencoded/04_step5_correlation_runset_encoded_best_nonencoded.tsv" \
  --covariate_protein_table "${ANALYTIC_TSV}" --outdir "${ALL_PROTEIN_OUTDIR}/05_encoded_nonencoded_correlation" \
  --prefix 05 --covariates "${COVARIATES[@]}" --min_complete_case "${MED_MIN_CC}" \
  --high_corr_cutoff "${HIGH_CORR_CUTOFF}" --moderate_corr_cutoff "${MODERATE_CORR_CUTOFF}" "$@"
