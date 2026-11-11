#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
require_files "${REACTOME_GMT}" "${GO_GMT}"
for set_name in ${ENRICHMENT_SETS}; do
  case "${set_name}" in combined) list=tier1_2_combined; label=Tier1_2_Combined;; tier1) list=tier1; label=Tier1;; tier2) list=tier2; label=Tier2;; *) printf 'Unknown enrichment set: %s\n' "${set_name}" >&2; exit 2;; esac
  for database in reactome go; do
    if [[ "${database}" == reactome ]]; then gmt="${REACTOME_GMT}"; db_label=Reactome; else gmt="${GO_GMT}"; db_label=GO; fi
    run_python "${SCRIPT_DIR}/02_run_enrichment_ora.py" \
      --input_list "${ENRICHMENT_OUTDIR}/protein_sets/${list}_protein_list.txt" --gmt "${gmt}" \
      --set_name "${label}" --database_label "${db_label}" \
      --out_tsv "${ENRICHMENT_OUTDIR}/${set_name}.${database}.ora.tsv" --out_qc "${ENRICHMENT_OUTDIR}/${set_name}.${database}.ora.qc.tsv" \
      --min_term_size "${ENRICHMENT_MIN_TERM_SIZE}" --max_term_size "${ENRICHMENT_MAX_TERM_SIZE}" "$@"
  done
done
