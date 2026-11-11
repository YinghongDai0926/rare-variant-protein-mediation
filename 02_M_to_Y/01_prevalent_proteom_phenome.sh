#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
require_files "${ANALYTIC_TSV}" "${ICD_FIRST_TSV}" "${FOLLOWUP_TSV}"
protein_slice
mkdir -p "${MY_OUTDIR}/prev_chunks"
run_python "${SCRIPT_DIR}/01_prevalent_proteom_phenome.py" \
  --analytic_tsv "${ANALYTIC_TSV}" --icd_first_tsv "${ICD_FIRST_TSV}" --followup_tsv "${FOLLOWUP_TSV}" \
  --protein_start "${PROTEIN_START}" --protein_end "${PROTEIN_END}" --min_cases "${MY_MIN_CASES}" --min_controls "${MY_MIN_CONTROLS}" \
  --out_tsv "${MY_OUTDIR}/prev_chunks/prev_chunk_${CHUNK_ID}.tsv" "$@"
