#!/usr/bin/env bash
# Single-node Super BF16 vLLM serve on HSG (GB200). Sourced env picks the config.
#
# RUNS ON HSG (inside the vllm/vllm-openai container, via srun/sbatch), NOT
# locally — there is no GPU / vLLM install in this environment. Intended to be
# invoked from run/hsg_static_batch.sbatch (or manually via `srun ... bash
# serve/serve_super_bf16_hsg.sh configs/baseline.env` for a standalone smoke).
#
# Team-shareable: no hardcoded checkpoint path — MODEL/TP/EXTRA_SERVE_ARGS
# come entirely from the sourced config (configs/*.env). Fill
# <SUPER_BF16_CKPT_PATH> in the config with the real checkpoint path
# (design §10 Q4) before the first real run.
#
# EXTRA_SERVE_ARGS must be a bash ARRAY in the config (not a string) and is
# expanded here as "${EXTRA_SERVE_ARGS[@]}" (quoted) so multi-word tokens
# with embedded quotes — e.g. mtp.env's --speculative-config JSON — survive
# as a single argv element instead of being word-split/quote-mangled.
set -euo pipefail
CFG="${1:?usage: serve_super_bf16_hsg.sh <configs/xxx.env>}"; source "$CFG"

if [ -z "${MODEL:-}" ] || [ "$MODEL" = "<SUPER_BF16_CKPT_PATH>" ]; then
  echo "ERROR: MODEL is unset or still the placeholder <SUPER_BF16_CKPT_PATH> in $CFG" >&2
  echo "       Fill in the real Super BF16 checkpoint path (design §10 Q4) before serving." >&2
  exit 1
fi

vllm serve "$MODEL" --tensor-parallel-size "${TP}" --port 8000 \
  --gpu-memory-utilization 0.9 --enable-prompt-tokens-details "${EXTRA_SERVE_ARGS[@]}"
# NOTE: CUDA graphs remain ON. Do not add --enforce-eager.
# --enable-prompt-tokens-details makes vLLM report usage.prompt_tokens_details.cached_tokens
# so aiperf/analyze can populate the PREFIX-CACHE HIT RATE (otherwise the cache metrics
# are silently empty even though prefix caching is active).
