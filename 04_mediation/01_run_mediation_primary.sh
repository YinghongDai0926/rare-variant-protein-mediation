#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"
validate_chunk_settings
mediation_gate_args
path_args=(--table_root "${MEDIATION_TABLE_ROOT}")
if [[ -n "${MEDIATION_PATH_PREFIX_FROM}" || -n "${MEDIATION_PATH_PREFIX_TO}" ]]; then
  if [[ -z "${MEDIATION_PATH_PREFIX_FROM}" || -z "${MEDIATION_PATH_PREFIX_TO}" ]]; then
    printf 'Set both MEDIATION_PATH_PREFIX_FROM and MEDIATION_PATH_PREFIX_TO\n' >&2; exit 2
  fi
  path_args+=(--table_path_prefix_from "${MEDIATION_PATH_PREFIX_FROM}" --table_path_prefix_to "${MEDIATION_PATH_PREFIX_TO}")
fi
run_python "${SCRIPT_DIR}/01_run_mediation_primary.py" --runset_tsv "${MEDIATION_RUNSET_TSV}" \
  --outdir "${MEDIATION_OUTDIR}" --chunk_id "${CHUNK_ID}" --n_chunks "${N_CHUNKS}" \
  "${path_args[@]}" "${MED_GATE_ARGS[@]}" --min_M_sd "${MED_MIN_SD}" \
  --n_boot "${N_BOOT}" --min_boot_success "${MIN_BOOT_SUCCESS}" --seed_base "${SEED_BASE}" "$@"
