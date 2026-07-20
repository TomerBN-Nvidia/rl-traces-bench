#!/usr/bin/env bash
# Replay a Mooncake trace as a STATIC BATCH: concurrency == #sessions, exact OSL.
#
# RUNS ON HSG (against a live `vllm serve` on localhost:8000, inside the
# same container/allocation as the server — see run/hsg_static_batch.sbatch),
# NOT locally — aiperf is not installed in this environment.
set -euo pipefail
cd "$(dirname "$0")/.."
TRACE="${1:?usage: run_batch.sh <trace.jsonl> <B> <out_dir>}"; B="${2:?}"; OUT="${3:?}"
aiperf profile --model "${AIPERF_MODEL:-super}" --endpoint-type chat \
  --endpoint /v1/chat/completions --url localhost:8000 --streaming \
  --custom-dataset-type mooncake_trace --input-file "$TRACE" \
  --concurrency "$B" --export-level records --output-artifact-dir "$OUT" \
  --extra-inputs ignore_eos:true --extra-inputs min_tokens:1
# aiperf may nest the export in a run subdir; locate it rather than assume the path.
EXPORT="$(find "$OUT" -name 'profile_export.jsonl' | head -1)"
[ -n "$EXPORT" ] || { echo "no profile_export.jsonl under $OUT"; find "$OUT" -type f; exit 1; }
PYTHONPATH=. python3 scripts/analyze.py --export "$EXPORT" \
  --out-html "$OUT/report.html" --out-json "$OUT/report.json"
