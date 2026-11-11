#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
run_python "${SCRIPT_DIR}/01_prepare_protein_sets.py" \
  --tier1_tsv "${MEDIATION_OUTDIR}/03_mediation_ranked_Tier1.tsv.gz" \
  --tier2_tsv "${MEDIATION_OUTDIR}/03_mediation_ranked_Tier2.tsv.gz" --outdir "${ENRICHMENT_OUTDIR}/protein_sets" "$@"
