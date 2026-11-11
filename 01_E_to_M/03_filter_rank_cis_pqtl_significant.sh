#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
run_python "${SCRIPT_DIR}/03_filter_rank_cis_pqtl_significant.py" \
  --in_tsv "${EM_OUTDIR}/cis_pqtl.per_protein.ALL.global_corrected.tsv" \
  --out_prefix "${EM_OUTDIR}/cis_pqtl.per_protein" --fdr_thr "${ALPHA}" --bonf_thr "${ALPHA}" --write_full "$@"
