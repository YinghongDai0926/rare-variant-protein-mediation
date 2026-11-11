#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
# Optional existing aggregator. It does not create candidate triplets, feasibility scans or runsets.
require_chunks "${FEASIBILITY_CHUNKS_DIR}" "${N_CHUNKS}" '01_feasibility.chunk' '.tsv'
run_python "${SCRIPT_DIR}/00_aggregate_feasibility_results.py" --indir "${FEASIBILITY_CHUNKS_DIR}" \
  --outdir "${MEDIATION_OUTDIR}/feasibility_summary" "$@"
