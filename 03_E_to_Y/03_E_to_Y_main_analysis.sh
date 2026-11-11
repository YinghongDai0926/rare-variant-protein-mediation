#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
validate_chunk_settings
for mode in ${MODE:-prevalent incident}; do
  case "${mode}" in prevalent) tag=prev;; incident) tag=inc;; *) printf 'Unknown MODE: %s\n' "${mode}" >&2; exit 2;; esac
  mkdir -p "${EY_OUTDIR}/${mode}"
  run_python "${SCRIPT_DIR}/03_E_to_Y_main_analysis.py" --mode "${mode}" \
    --sample-file "${EY_INPUTDIR}/E_matrix.samples.txt" --gene-file "${EY_INPUTDIR}/E_matrix.genes.txt" \
    --carrier-matrix "${EY_INPUTDIR}/E_matrix.carrier_binary.npz" --carrier-counts "${EY_INPUTDIR}/E_matrix.carrier_counts.tsv" \
    --covariate-file "${ANALYTIC_TSV}" --followup-file "${FOLLOWUP_TSV}" --icd-file "${ICD_FIRST_TSV}" \
    --counts-file "${EY_INPUTDIR}/${tag}_icd3_counts.tsv" --covariates "${COVARIATE_CSV}" \
    --chunk-id "${CHUNK_ID}" --n-chunks "${N_CHUNKS}" \
    --min-carriers-global "${EY_MIN_CARRIERS_GLOBAL}" --max-carrier-rate-global "${EY_MAX_CARRIER_RATE_GLOBAL}" \
    --min-case-like "${EY_MIN_CASE_LIKE}" --min-ctrl-like "${EY_MIN_CTRL_LIKE}" \
    --min-carriers-used "${EY_MIN_CARRIERS_USED}" --min-noncarriers-used "${EY_MIN_NONCARRIERS_USED}" \
    --min-carrier-cases "${EY_MIN_CARRIER_CASES}" --min-carrier-controls "${EY_MIN_CARRIER_CONTROLS}" \
    --min-carrier-events "${EY_MIN_CARRIER_EVENTS}" --min-carrier-nonevents "${EY_MIN_CARRIER_NONEVENTS}" \
    --max-abs-beta "${EY_MAX_ABS_BETA}" --out "${EY_OUTDIR}/${mode}/step5c.${mode}.chunk${CHUNK_ID}.tsv.gz" \
    --out-skip "${EY_OUTDIR}/${mode}/step5c.${mode}.chunk${CHUNK_ID}.skipped.tsv.gz" "$@"
done
