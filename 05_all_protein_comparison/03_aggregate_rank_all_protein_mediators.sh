#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
require_chunks "${ALL_PROTEIN_OUTDIR}/02_all_protein_mediation_scan" "${N_CHUNKS}" '02_all_protein_mediation_scan.chunk' '.tsv.gz'
run_python "${SCRIPT_DIR}/03_aggregate_rank_all_protein_mediators.py" \
  --scan_dir "${ALL_PROTEIN_OUTDIR}/02_all_protein_mediation_scan" --chunk_pattern '02_all_protein_mediation_scan.chunk*.tsv.gz' \
  --supported_pairs "${ALL_PROTEIN_OUTDIR}/01_supported_gene_disease_pairs.tsv" \
  --outdir "${ALL_PROTEIN_OUTDIR}/03_encoded_protein_ranking" --prefix 03 --case_gene EGFR --case_icd3 N17 "$@"
