#!/usr/bin/env bash
# Replay a Mooncake trace as a STATIC BATCH: concurrency == #sessions, exact OSL.
#
# RUNS ON HSG (against a live `vllm serve` on localhost:8000, inside the
# same container/allocation as the server — see run/hsg_static_batch.sbatch),
# NOT locally — aiperf is not installed in this environment.
set -euo pipefail
cd "$(dirname "$0")/.."
TRACE="${1:?usage: run_batch.sh <trace.jsonl> <B> <out_dir>}"; B="${2:?}"; OUT="${3:?}"
# aiperf needs a tokenizer (to synthesize prompts to the trace input_length) and
# resolves it via HuggingFace from_pretrained, which requires an HF REPO ID in
# namespace/name form (an absolute local path is rejected). Pass the model HF id,
# e.g. nvidia/NVIDIA-Nemotron-3-Super-120B-BF16-BF16KV-012726, and point HF_HOME
# at a warm cache to load it offline. Required.
TOKENIZER="${TOKENIZER:?set TOKENIZER to the served model HF repo id (namespace/name)}"
# Endpoint: default /v1/chat/completions. Chat is REQUIRED for multi-turn — aiperf
# accumulates the conversation across turns (growing prefix -> prefix caching) and
# generates every turn. (The completions endpoint only generates the first turn of a
# mooncake conversation, collapsing the rollout tail — validated on HSG.) Exact
# per-turn OSL is enforced on chat by ignore_eos + the CORRECT tokenizer +
# --use-server-token-count (no min_tokens needed) — also validated on HSG.
ENDPOINT_TYPE="${ENDPOINT_TYPE:-chat}"
ENDPOINT="${ENDPOINT:-/v1/chat/completions}"
# Per NVIDIA btk-recipes bench-clients-aiperf guide:
#  --tokenizer-trust-remote-code : required for custom tokenizers (Nemotron etc.)
#  --use-server-token-count      : take OSL from the server's usage.completion_tokens
#                                  instead of re-detokenizing client-side (avoids the
#                                  spurious OSL-mismatch from a client tokenizer roundtrip).
aiperf profile --model "${AIPERF_MODEL:-super}" --tokenizer "$TOKENIZER" \
  --tokenizer-trust-remote-code --use-server-token-count \
  --endpoint-type "$ENDPOINT_TYPE" --endpoint "$ENDPOINT" --url localhost:8000 --streaming \
  --custom-dataset-type mooncake_trace --input-file "$TRACE" \
  --concurrency "$B" --export-level records --output-artifact-dir "$OUT" \
  ${SYNTH_MAX_OSL:+--synthesis-max-osl $SYNTH_MAX_OSL} \
  --extra-inputs ignore_eos:true
# aiperf may nest the export in a run subdir; locate it rather than assume the path.
EXPORT="$(find "$OUT" -name 'profile_export.jsonl' | head -1)"
[ -n "$EXPORT" ] || { echo "no profile_export.jsonl under $OUT"; find "$OUT" -type f; exit 1; }
PYTHONPATH=. python3 scripts/analyze.py --export "$EXPORT" \
  --out-html "$OUT/report.html" --out-json "$OUT/report.json"
