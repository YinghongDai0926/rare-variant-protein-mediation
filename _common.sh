#!/usr/bin/env bash
# Shared launcher support; no statistical models are implemented here.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/config.sh}"
if [[ ! -f "${CONFIG_FILE}" && "${CONFIG_FILE}" == "${REPO_ROOT}/config.sh" ]]; then
  CONFIG_FILE="${REPO_ROOT}/config.example.sh"
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
  printf 'Configuration file not found: %s\n' "${CONFIG_FILE}" >&2
  exit 2
fi
source "${CONFIG_FILE}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "${PYTHON_BIN}" >&2
  exit 2
fi
mkdir -p "${LOG_ROOT}"
CHUNK_ID="${CHUNK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
N_CHUNKS="${N_CHUNKS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
validate_chunk_settings() {
  if [[ ! "${CHUNK_ID}" =~ ^[0-9]+$ || ! "${N_CHUNKS}" =~ ^[0-9]+$ ]] || (( N_CHUNKS < 1 || CHUNK_ID >= N_CHUNKS )); then
    printf 'Require zero-based 0 <= CHUNK_ID < N_CHUNKS, got %s/%s\n' "${CHUNK_ID}" "${N_CHUNKS}" >&2
    return 2
  fi
}
COVARIATE_CSV="$(printf '%s,' "${COVARIATES[@]}")"
COVARIATE_CSV="${COVARIATE_CSV%,}"

require_files() {
  local file
  for file in "$@"; do
    if [[ ! -f "${file}" ]]; then
      printf 'Required input file not found: %s\n' "${file}" >&2
      return 2
    fi
  done
}

require_chunks() {
  # Directory, expected count, filename prefix, filename suffix.
  "${PYTHON_BIN}" - "$1" "$2" "$3" "$4" <<'PY'
from pathlib import Path
import re, sys
directory, count, prefix, suffix = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
expected = {prefix + str(i) + suffix for i in range(count)}
pattern = re.compile(re.escape(prefix) + r'[0-9]+' + re.escape(suffix))
actual = {p.name for p in directory.glob('*') if p.is_file() and pattern.fullmatch(p.name)}
missing, extra = sorted(expected - actual), sorted(actual - expected)
if missing or extra:
    raise SystemExit(f'Incomplete or mixed chunk batch in {directory}: missing={missing}, unexpected={extra}. Set N_CHUNKS to the actual submitted count.')
PY
}

protein_slice() {
  # Single-task execution covers all proteins. Array tasks retain the original 60-column slice.
  local slice
  validate_chunk_settings
  slice="$("${PYTHON_BIN}" - "${ANALYTIC_TSV}" "${CHUNK_ID}" "${N_CHUNKS}" "${PROTEIN_CHUNK_SIZE}" <<'PY'
import csv, math, sys
path, chunk, chunks, width = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
with open(path) as f:
    header = next(csv.reader(f, delimiter='\t'))
covariates = {'sample', 'age', 'sex'} | {f'PC{i}' for i in range(1, 11)}
n = sum(c not in covariates for c in header)
if width < 1:
    raise SystemExit('PROTEIN_CHUNK_SIZE must be positive')
if chunks == 1:
    start, end = 0, n
else:
    if chunks < math.ceil(n / width):
        raise SystemExit(f'{chunks} tasks x {width} proteins do not cover {n} proteins')
    start, end = chunk * width, min((chunk + 1) * width, n)
print(start, end)
PY
)" || return $?
  read -r PROTEIN_START PROTEIN_END <<< "${slice}"
}

mediation_gate_args() {
  MED_GATE_ARGS=(
    --covariates "${COVARIATES[@]}"
    --min_complete_case "${MED_MIN_CC}"
    --min_events_complete_case "${MED_MIN_EVENTS}"
    --min_carriers_complete_case "${MED_MIN_CARRIERS}"
    --min_noncarriers_complete_case "${MED_MIN_NONCARRIERS}"
    --min_carrier_events_complete_case "${MED_MIN_CARRIER_EVENTS}"
    --min_carrier_nonevents_complete_case "${MED_MIN_CARRIER_NONEVENTS}"
  )
}

all_protein_input_args() {
  ALL_INPUT_ARGS=(
    --covariate_protein_table "${ANALYTIC_TSV}"
    --e_samples "${EY_INPUTDIR}/E_matrix.samples.txt"
    --e_genes "${EY_INPUTDIR}/E_matrix.genes.txt"
    --e_matrix "${EY_INPUTDIR}/E_matrix.carrier_binary.npz"
    --first_occurrence "${ICD_FIRST_TSV}"
    --followup_dates "${FOLLOWUP_TSV}"
  )
}

run_python() {
  local metadata rc=0
  metadata="$(mktemp "${LOG_ROOT}/$(basename "$1" .py).chunk${CHUNK_ID}.XXXXXX")"
  mv -- "${metadata}" "${metadata}.json"
  metadata="${metadata}.json"
  "${PYTHON_BIN}" - "${metadata}" "${CONFIG_FILE}" "$@" <<'PY'
from pathlib import Path
import datetime, hashlib, importlib.metadata as metadata, json, platform, sys
output, config, *argv = sys.argv[1:]
paths = []
for value in argv:
    p = Path(value)
    if p.is_file():
        stat = p.stat()
        row = {'path': str(p.resolve()), 'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if stat.st_size <= 20 * 1024 * 1024:
            row['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
        paths.append(row)
versions = {}
for package in ('numpy', 'pandas', 'scipy', 'statsmodels'):
    try:
        versions[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        versions[package] = None
record = {'status': 'started', 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'argv': argv, 'python_executable': sys.executable, 'python_version': platform.python_version(),
          'author_reported_runtime': {'python': '3.12.6', 'R': '4.5.1'}, 'package_versions': versions,
          'config_path': str(Path(config).resolve()), 'config_sha256': hashlib.sha256(Path(config).read_bytes()).hexdigest(),
          'input_file_metadata': paths}
Path(output).write_text(json.dumps(record, indent=2) + '\n')
PY
  "${PYTHON_BIN}" "$@" || rc=$?
  "${PYTHON_BIN}" - "${metadata}" "${rc}" <<'PY'
from pathlib import Path
import datetime, json, sys
p = Path(sys.argv[1]); record = json.loads(p.read_text()); code = int(sys.argv[2])
record.update(status='succeeded' if code == 0 else 'failed', exit_code=code,
              finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
p.write_text(json.dumps(record, indent=2) + '\n')
PY
  return "${rc}"
}
