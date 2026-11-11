#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
run_python "${SCRIPT_DIR}/01_build_all_protein_mediation_runset.py" --master-table "${SUPPORTED_MEDIATION_TSV}" \
  --covariate-protein-table "${ANALYTIC_TSV}" --outdir "${ALL_PROTEIN_OUTDIR}" --prefix 01 --strict-encoded-protein-check "$@"
