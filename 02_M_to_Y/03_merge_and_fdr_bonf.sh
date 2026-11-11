#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
for mode in ${MODE:-prevalent incident}; do
  case "${mode}" in prevalent) tag=prev;; incident) tag=inc;; *) printf 'Unknown MODE: %s\n' "${mode}" >&2; exit 2;; esac
  require_chunks "${MY_OUTDIR}/${tag}_chunks" "${N_CHUNKS}" "${tag}_chunk_" '.tsv'
  run_python "${SCRIPT_DIR}/03_merge_and_fdr_bonf.py" \
    --chunk_glob "${MY_OUTDIR}/${tag}_chunks/${tag}_chunk_*.tsv" \
    --out_prefix "${MY_OUTDIR}/${tag}_chunks/${tag}_proteome_phenome" --alpha "${ALPHA}" "$@"
done
