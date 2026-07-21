#!/usr/bin/env bash
# GPU-free A/B compare: replay ONE trace against two mock configs that differ only
# in modelled decode speed, then render the interactive compare report. No cluster.
#
# The two configs stand in for "baseline (no speculative decoding)" vs "MTP with a
# forced/synthetic acceptance rate": MOCK_ACCEPT_LEN is the mean accepted
# tokens-per-step, so config B decodes proportionally faster over the same trace.
# On a real GPU this is the vLLM knob
#   --speculative-config '{"method":"mtp","num_speculative_tokens":3,
#                          "rejection_sample_method":"synthetic","synthetic_acceptance_rate":0.8}'
# (see examples/compare_mtp.sh). Here it is a latency MODEL, not a real engine.
set -euo pipefail
PORT="${PORT:-8137}"
OUT="${OUT:-/tmp/rl_traces_ab}"
TRACE="${TRACE:-$OUT/trace.jsonl}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-32}"
MS_PER_TOKEN="${MOCK_MS_PER_TOKEN:-0.1}"   # base decode latency per output token
ACCEPT_A="${ACCEPT_A:-1.0}"                 # baseline: 1 accepted token/step
ACCEPT_B="${ACCEPT_B:-2.6}"                 # "MTP α≈0.8, k=3": ~2.6 tokens/step
HERE="$(cd "$(dirname "$0")" && pwd)"
rm -rf "$OUT"; mkdir -p "$OUT"

wait_up(){ for i in $(seq 1 60); do
  python3 -c "import urllib.request;urllib.request.urlopen('http://localhost:${PORT}/v1/models',timeout=1)" 2>/dev/null && return 0
  sleep 0.5; done; echo "mock never came up"; exit 1; }

run_cfg(){  # name accept_len
  local name="$1" accept="$2"
  MOCK_MS_PER_TOKEN="$MS_PER_TOKEN" MOCK_ACCEPT_LEN="$accept" \
    python3 "$HERE/mock_server.py" "$PORT" >"$OUT/mock_$name.log" 2>&1 &
  local pid=$!
  wait_up
  rl-traces run --url "localhost:${PORT}" --trace "$TRACE" --concurrency "$NUM_ROLLOUTS" \
    --tokenizer gpt2 --out "$OUT/$name" >"$OUT/run_$name.log" 2>&1
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

echo "[1/3] gen-trace ($NUM_ROLLOUTS rollouts)"
rl-traces gen-trace --num-rollouts "$NUM_ROLLOUTS" --seed 0 --out "$TRACE" >/dev/null

echo "[2/3] replay baseline (accept=$ACCEPT_A) then mtp (accept=$ACCEPT_B)"
run_cfg baseline "$ACCEPT_A"
run_cfg mtp      "$ACCEPT_B"

echo "[3/3] compare -> $OUT/compare.html"
rl-traces compare \
  "baseline (no spec)=$OUT/baseline/report.json" \
  "mtp accept=$ACCEPT_B=$OUT/mtp/report.json" \
  --out-html "$OUT/compare.html"

echo "AB OK"
echo "  baseline : $OUT/baseline/report.html"
echo "  mtp      : $OUT/mtp/report.html"
echo "  compare  : $OUT/compare.html"
