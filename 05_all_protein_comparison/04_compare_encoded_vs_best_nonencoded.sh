#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
run_python "${SCRIPT_DIR}/04_compare_encoded_vs_best_nonencoded.py" \
  --encoded_summary "${ALL_PROTEIN_OUTDIR}/03_encoded_protein_ranking/03_encoded_protein_ranking_summary.tsv" \
  --outdir "${ALL_PROTEIN_OUTDIR}/04_encoded_vs_best_nonencoded" --prefix 04 "$@"
