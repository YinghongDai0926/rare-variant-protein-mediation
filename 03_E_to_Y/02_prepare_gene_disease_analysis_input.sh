#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
run_python "${SCRIPT_DIR}/02_prepare_gene_disease_analysis_input.py" \
  --outdir "${EY_INPUTDIR}" --analysis_samples_tsv "${ANALYTIC_TSV}" --sample_col sample \
  --first_occ "${ICD_FIRST_TSV}" --followup "${FOLLOWUP_TSV}" \
  --carrier_counts_tsv "${EY_INPUTDIR}/E_matrix.carrier_counts.tsv" \
  --E_npz "${EY_INPUTDIR}/E_matrix.carrier_binary.npz" --E_genes "${EY_INPUTDIR}/E_matrix.genes.txt" \
  --E_samples "${EY_INPUTDIR}/E_matrix.samples.txt" --min_carriers "${EY_MATRIX_MIN_CARRIERS}" \
  --min_noncarriers "${EY_MATRIX_MIN_NONCARRIERS}" --alert_carrier_rate "${EY_MATRIX_MAX_CARRIER_RATE}" "$@"
