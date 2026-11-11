#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
for chromosome in "${CHROMOSOMES[@]}"; do
  require_files "${EM_OUTDIR}/chr${chromosome}.combined.cis_pqtl.per_protein.tsv.gz"
done
run_python "${SCRIPT_DIR}/02_merge_and_complete_FDR_correct.py" --indir "${EM_OUTDIR}" \
  --out "${EM_OUTDIR}/cis_pqtl.per_protein.ALL.global_corrected.tsv" "$@"
