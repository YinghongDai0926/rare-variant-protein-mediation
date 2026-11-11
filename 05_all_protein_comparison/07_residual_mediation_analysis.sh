#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
mediation_gate_args
all_protein_input_args
run_python "${SCRIPT_DIR}/07_residual_mediation_analysis.py" --residual_runset "${ALL_PROTEIN_OUTDIR}/06_conditional_mediation_encoded_vs_nonencoded/06_conditional_mediation_encoded_vs_nonencoded.tsv" \
  "${ALL_INPUT_ARGS[@]}" --outdir "${ALL_PROTEIN_OUTDIR}/07_residual_mediation_analysis" --prefix 07 \
  "${MED_GATE_ARGS[@]}" --min_residual_sd "${MED_MIN_SD}" --retained_p_cutoff "${RETAINED_P_CUTOFF}" "$@"
