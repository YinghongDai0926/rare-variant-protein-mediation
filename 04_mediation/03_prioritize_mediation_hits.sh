#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
run_python "${SCRIPT_DIR}/03_prioritize_mediation_hits.py" \
  --mediation_results_tsv "${MEDIATION_OUTDIR}/02_mediation_results.all.tsv.gz" --outdir "${MEDIATION_OUTDIR}" \
  --min_boot_success_rate "${BOOT_STABILITY}" --min_n_boot_success "${MIN_BOOT_SUCCESS}" --require_fdr 1 --alpha "${ALPHA}" "$@"
