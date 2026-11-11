#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
run_python "${SCRIPT_DIR}/08_final_pathway_classification.py" \
  --step4_comparison "${ALL_PROTEIN_OUTDIR}/04_encoded_vs_best_nonencoded/04_encoded_vs_best_nonencoded_comparison.tsv" \
  --step5_correlation "${ALL_PROTEIN_OUTDIR}/05_encoded_nonencoded_correlation/05_encoded_nonencoded_correlation.tsv" \
  --step6_conditional "${ALL_PROTEIN_OUTDIR}/06_conditional_mediation_encoded_vs_nonencoded/06_conditional_mediation_encoded_vs_nonencoded.tsv" \
  --step7_residual_pair "${ALL_PROTEIN_OUTDIR}/07_residual_mediation_analysis/07_residual_mediation_pair_summary.tsv" \
  --outdir "${ALL_PROTEIN_OUTDIR}/08_final_pathway_classification" --prefix 08 "$@"
