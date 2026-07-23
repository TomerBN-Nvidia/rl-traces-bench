#!/usr/bin/env bash
# Real-vLLM A/B: baseline (no speculative decoding) vs MTP with a FORCED
# (synthetic) acceptance rate, over ONE shared trace — then render the compare
# report. Requires a GPU + a vLLM 0.24+ install and a model whose checkpoint has
# an MTP head (config.json: num_nextn_predict_layers >= 1).
#
# Why synthetic acceptance: real MTP drafts are verified against the target
# model, so on the synthetic trace prompts (random text, forced OSL) acceptance
# is ~0 and you'd measure no speedup. vLLM's synthetic rejection sampler
# BYPASSES verification and accepts a target fraction, isolating the serving-path
# effect of "MTP at acceptance = X" independent of draft quality:
#   --speculative-config '{"method":"mtp","num_speculative_tokens":3,
#                          "rejection_sample_method":"synthetic",
#                          "synthetic_acceptance_rate":0.8}'
#
# This script assumes you host the model yourself (two serves on the same GPU,
# one after the other). If you already have two endpoints, skip serving and just
# point `rl-traces run --url` at each, then compare.
set -euo pipefail
MODEL="${MODEL:?set MODEL=<hf-id-or-checkpoint-path with an MTP head>}"
TOKENIZER="${TOKENIZER:-$MODEL}"     # aiperf needs an HF repo id (namespace/name)
TP="${TP:-4}"
B="${B:-64}"                          # rollouts == static-batch concurrency
ACCEPT="${ACCEPT:-0.8}"               # forced MTP acceptance rate in [0,1]
K="${K:-3}"                           # num_speculative_tokens
OUT="${OUT:-runs/mtp_ab}"
PORT="${PORT:-8000}"
mkdir -p "$OUT"

# One shared trace (deterministic) drives both configs, so any timing delta is
# attributable to the serving config, not the workload.
rl-traces gen-trace --num-rollouts "$B" --seed 0 --out "$OUT/trace.jsonl"

serve_and_run(){  # name  extra-serve-args...
  local name="$1"; shift
  vllm serve "$MODEL" --tensor-parallel-size "$TP" --port "$PORT" \
    --gpu-memory-utilization 0.9 --max-model-len 131072 \
    --enable-prefix-caching --enable-prompt-tokens-details \
    --served-model-name m "$@" > "$OUT/$name.serve.log" 2>&1 &
  local spid=$!
  echo "[$name] waiting for health..."
  until python3 -c "import urllib.request;urllib.request.urlopen('http://localhost:$PORT/health',timeout=5)" 2>/dev/null; do
    kill -0 "$spid" 2>/dev/null || { echo "[$name] serve died"; tail -30 "$OUT/$name.serve.log"; exit 1; }
    sleep 10
  done
  rl-traces run --url "localhost:$PORT" --trace "$OUT/trace.jsonl" --concurrency "$B" \
    --tokenizer "$TOKENIZER" --model m --out "$OUT/$name"
  kill "$spid" 2>/dev/null || true
  sleep 5
}

# A: baseline — no speculative decoding
serve_and_run baseline

# B: MTP with a forced acceptance rate
serve_and_run mtp \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$K,\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_rate\":$ACCEPT}"

rl-traces compare \
  "baseline (no spec)=$OUT/baseline/report.json" \
  "mtp accept=$ACCEPT=$OUT/mtp/report.json" \
  --out-html "$OUT/compare.html"

echo "compare -> $OUT/compare.html"
