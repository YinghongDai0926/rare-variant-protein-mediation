#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
run_python "${SCRIPT_DIR}/01_prepare_EY_matrix_file.py" \
  --carriers_tsv_template "${CARRIERS_TSV_TEMPLATE}" --analysis_samples_tsv "${ANALYTIC_TSV}" \
  --cov_cols "${COVARIATES[@]}" --chroms "${CHROMOSOMES[@]}" \
  --min_carriers "${EY_MATRIX_MIN_CARRIERS}" --min_noncarriers "${EY_MATRIX_MIN_NONCARRIERS}" \
  --max_carrier_rate "${EY_MATRIX_MAX_CARRIER_RATE}" --outdir "${EY_INPUTDIR}" "$@"
