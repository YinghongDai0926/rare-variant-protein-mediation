#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
validate_chunk_settings
mediation_gate_args
all_protein_input_args
run_python "${SCRIPT_DIR}/02_run_all_protein_mediation_scan.py" \
  --runset_tsv "${ALL_PROTEIN_OUTDIR}/01_all_protein_mediation_runset.tsv.gz" "${ALL_INPUT_ARGS[@]}" \
  --outdir "${ALL_PROTEIN_OUTDIR}/02_all_protein_mediation_scan" --chunk_id "${CHUNK_ID}" --n_chunks "${N_CHUNKS}" \
  "${MED_GATE_ARGS[@]}" --min_M_sd "${MED_MIN_SD}" "$@"
