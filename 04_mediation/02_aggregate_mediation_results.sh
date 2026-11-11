#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
require_chunks "${MEDIATION_OUTDIR}" "${N_CHUNKS}" '01_mediation.chunk' '.tsv.gz'
run_python "${SCRIPT_DIR}/02_aggregate_mediation_results.py" --indir "${MEDIATION_OUTDIR}" \
  --outdir "${MEDIATION_OUTDIR}" --alpha "${ALPHA}" --boot_stability_threshold "${BOOT_STABILITY}" "$@"
