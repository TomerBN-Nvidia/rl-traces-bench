#!/usr/bin/env bash
# GPU-free end-to-end smoke: mock endpoint -> gen-trace -> run -> report. No cluster.
set -euo pipefail
PORT="${PORT:-8000}"; OUT="${OUT:-/tmp/rl_traces_smoke}"; TRACE="${TRACE:-/tmp/rl_traces_smoke.jsonl}"
rm -rf "$OUT"; mkdir -p "$OUT"
python examples/mock_server.py "$PORT" & MOCK=$!
trap 'kill $MOCK 2>/dev/null || true' EXIT
for i in $(seq 1 40); do
  python -c "import urllib.request;urllib.request.urlopen('http://localhost:${PORT}/v1/models',timeout=1)" 2>/dev/null && break
  sleep 0.5; [ "$i" = 40 ] && { echo "mock never came up"; exit 1; }
done
rl-traces gen-trace --num-rollouts 8 --seed 0 --out "$TRACE"
rl-traces run --url "localhost:${PORT}" --trace "$TRACE" --concurrency 8 --tokenizer gpt2 --out "$OUT"
echo "SMOKE OK"
