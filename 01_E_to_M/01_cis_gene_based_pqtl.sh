#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
mkdir -p "${EM_OUTDIR}"
if [[ -n "${CHR:-}" ]]; then
  selected=("${CHR}")
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if (( SLURM_ARRAY_TASK_ID < 1 || SLURM_ARRAY_TASK_ID > ${#CHROMOSOMES[@]} )); then
    printf 'Chromosome array IDs must be 1..%s\n' "${#CHROMOSOMES[@]}" >&2; exit 2
  fi
  selected=("${CHROMOSOMES[$((SLURM_ARRAY_TASK_ID-1))]}")
else
  selected=("${CHROMOSOMES[@]}")
fi
for chromosome in "${selected[@]}"; do
  run_python "${SCRIPT_DIR}/01_cis_gene_based_pqtl.py" --chr "${chromosome}" \
    --carriers_tsv_template "${CARRIERS_TSV_TEMPLATE}" --covar_protein_tsv "${ANALYTIC_TSV}" \
    --cov_cols "${COVARIATES[@]}" --min_carriers_used "${EM_MIN_CARRIERS}" \
    --min_samples_used "${EM_MIN_SAMPLES}" --outdir "${EM_OUTDIR}" "$@"
done
