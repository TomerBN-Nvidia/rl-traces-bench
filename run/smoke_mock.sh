#!/usr/bin/env bash
# Self-contained GPU-free smoke of the whole pipeline: starts a tiny mock chat
# endpoint (run/mock_chat_server.py), generates a small mooncake trace, replays it
# with aiperf, and analyzes the result. RUNS ON HSG (needs aiperf; no GPU).
# Validates trace -> aiperf -> analyze end to end and surfaces aiperf's real
# per-request profile_export field names (update analyze.py's alias map if needed).
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8000}"
OUT="${OUT:-/tmp/smoke_out}"
TRACE="${TRACE:-/tmp/smoke_trace.jsonl}"
rm -rf "$OUT"; mkdir -p "$OUT"

python3 run/mock_chat_server.py "$PORT" & MOCK=$!
trap 'kill $MOCK 2>/dev/null || true' EXIT

# wait for the mock endpoint (stdlib; no curl assumption)
for i in $(seq 1 40); do
  if python3 -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:${PORT}/v1/models',timeout=1)" 2>/dev/null; then
    echo "mock up on :${PORT}"; break
  fi
  sleep 0.5
  [ "$i" = "40" ] && { echo "mock server never came up"; exit 1; }
done

PYTHONPATH=. python3 scripts/gen_trace.py --num-rollouts 8 --osl-level per_turn --seed 0 --out "$TRACE"

aiperf profile --model mock-model --endpoint-type chat --endpoint /v1/chat/completions \
  --url "localhost:${PORT}" --custom-dataset-type mooncake_trace --input-file "$TRACE" \
  --concurrency 8 --streaming --export-level records --output-artifact-dir "$OUT"

# aiperf's export path can vary by version; find the per-request records file.
EXPORT="$(find "$OUT" -name 'profile_export.jsonl' | head -1)"
[ -n "$EXPORT" ] || { echo "no profile_export*.jsonl under $OUT"; find "$OUT" -type f; exit 1; }
echo "profile export: $EXPORT"

PYTHONPATH=. python3 scripts/analyze.py --export "$EXPORT" \
  --out-html "$OUT/report.html" --out-json "$OUT/report.json"
echo "SMOKE OK"
