#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
for mode in ${MODE:-prevalent incident}; do
  case "${mode}" in prevalent|incident) ;; *) printf 'Unknown MODE: %s\n' "${mode}" >&2; exit 2;; esac
  require_chunks "${EY_OUTDIR}/${mode}" "${N_CHUNKS}" "step5c.${mode}.chunk" '.tsv.gz'
  run_python "${SCRIPT_DIR}/04_merge_all_and_correction.py" --mode "${mode}" \
    --indir "${EY_OUTDIR}/${mode}" --outdir "${EY_OUTDIR}/${mode}" --fdr-threshold "${ALPHA}" "$@"
done
