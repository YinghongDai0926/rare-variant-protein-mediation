#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
run_python "${SCRIPT_DIR}/04_extract_main_and_supp_tables.py" \
  --ranked_all_tsv "${MEDIATION_OUTDIR}/03_mediation_ranked_all.tsv.gz" \
  --ranked_sig_tsv "${MEDIATION_OUTDIR}/03_mediation_ranked_significant.tsv.gz" \
  --ranked_tier1_tsv "${MEDIATION_OUTDIR}/03_mediation_ranked_Tier1.tsv.gz" \
  --ranked_tier2_tsv "${MEDIATION_OUTDIR}/03_mediation_ranked_Tier2.tsv.gz" \
  --mediation_results_tsv "${MEDIATION_OUTDIR}/02_mediation_results.all.tsv.gz" --outdir "${MEDIATION_OUTDIR}" \
  --top_n_overall 20 --top_n_tier1 8 --top_n_tier2 20 "$@"
