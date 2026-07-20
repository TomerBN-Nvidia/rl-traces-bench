#!/usr/bin/env bash
#SBATCH --job-name=rl-longtail-static-batch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=rl-longtail-static-batch-%j.log
#
# Combined single-allocation orchestrator for the RL long-tail reproducer:
#   serve -> wait-for-health -> gen_trace (B rollouts) -> run_batch (aiperf) -> analyze
# One sbatch, one allocation (avoid separate build/serve/bench queue waits).
#
# RUNS ON HSG (single-node GB200, aarch64), NOT locally. Submit from an HSG
# login node with `sbatch`, from inside reproducer/ or with a path to this file.
#
# Team-shareable script: NO hardcoded --account/--partition/--qos and NO
# personal/user directory defaults. Pass account/partition on the sbatch
# command line; pass CONTAINER/OUT_DIR/MOUNTS as env vars. Missing required
# values fail fast with a usage error instead of silently defaulting to a
# user's home/lustre dir.
#
# Usage (example — HSG batch partition needs --gpus-per-node=4 even here):
#   MODEL=<SUPER_BF16_CKPT_PATH> \
#   CONTAINER=<VLLM_CONTAINER_IMAGE_OR_SQSH_PATH> \
#   OUT_DIR=<RESULTS_DIR> \
#   sbatch --account=<ACCOUNT> --partition=batch --qos=<QOS> \
#     run/hsg_static_batch.sbatch configs/baseline.env
#
# Submit under every Slurm account you have access to (race for allocation);
# cancel the loser once one starts running.
set -euo pipefail

CFG="${1:?usage: sbatch [--account=... --partition=...] run/hsg_static_batch.sbatch <configs/xxx.env>}"

: "${CONTAINER:?ERROR: set CONTAINER=<vllm/vllm-openai:TAG or /path/to/vllm.sqsh> — no default, this script is team-shareable}"
: "${OUT_DIR:?ERROR: set OUT_DIR=<results dir> — no default, do not write into a personal directory}"
: "${MOUNTS:?ERROR: set MOUNTS=<host:container mounts, e.g. /lustre:/lustre,/home/\${USER}:/home/\${USER}> — no default}"
: "${B:=512}"                    # static-batch size (design §8 default; sweep upward for prod-like runs)
: "${AIPERF_MODEL:=super}"
: "${HEALTH_TIMEOUT_S:=1800}"    # generous — Super BF16 TP4 load can take a while
: "${AIPERF_INSTALL_CMD:=pip install -q aiperf}"
: "${CONTAINER_NAME:=rl-lt-${SLURM_JOB_ID:-local}}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$OUT_DIR"

echo "=========================================="
echo "job=${SLURM_JOB_ID:-local} cfg=$CFG container=$CONTAINER out=$OUT_DIR B=$B"
echo "=========================================="

# --- 1) Serve (background, inside the container, held open across steps via
#         --container-name so later srun steps share its filesystem/overlay
#         and can reach it on localhost) -----------------------------------
srun --container-image="$CONTAINER" --container-mounts="$MOUNTS" \
     --container-name="$CONTAINER_NAME" \
     bash "$REPO_DIR/serve/serve_super_bf16_hsg.sh" "$CFG" \
     > "$OUT_DIR/serve.log" 2>&1 &
SERVE_PID=$!

# --- 2) Wait for health (python3 stdlib only — don't assume curl exists in
#         the vllm-openai image) ------------------------------------------
echo "waiting for server health (timeout ${HEALTH_TIMEOUT_S}s)..."
elapsed=0
until srun --container-name="$CONTAINER_NAME" --overlap \
        python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" \
        >/dev/null 2>&1; do
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "ERROR: serve process exited before becoming healthy — see $OUT_DIR/serve.log" >&2
    exit 1
  fi
  if [ "$elapsed" -ge "$HEALTH_TIMEOUT_S" ]; then
    echo "ERROR: server did not become healthy within ${HEALTH_TIMEOUT_S}s — see $OUT_DIR/serve.log" >&2
    kill "$SERVE_PID" 2>/dev/null || true
    exit 1
  fi
  sleep 10
  elapsed=$((elapsed + 10))
done
echo "server healthy after ${elapsed}s"

# --- 3) Generate the B-rollout trace (pure python, no aiperf needed) -------
TRACE="$OUT_DIR/trace_b${B}.jsonl"
srun --container-name="$CONTAINER_NAME" --overlap \
     bash -c "cd '$REPO_DIR' && PYTHONPATH=. python3 scripts/gen_trace.py --num-rollouts $B --osl-level per_turn --out '$TRACE'"

# --- 4) aiperf static-batch replay + analyzer ------------------------------
# aiperf is not baked into the stock vllm/vllm-openai image; install into the
# same named container instance (persists for the life of this job).
srun --container-name="$CONTAINER_NAME" --overlap \
     bash -c "$AIPERF_INSTALL_CMD"
srun --container-name="$CONTAINER_NAME" --overlap \
     bash -c "cd '$REPO_DIR' && AIPERF_MODEL='$AIPERF_MODEL' bash run/run_batch.sh '$TRACE' '$B' '$OUT_DIR/run_b${B}'"

# --- 5) Stop the server, we're done ----------------------------------------
kill "$SERVE_PID" 2>/dev/null || true

echo "=========================================="
echo "DONE — report: $OUT_DIR/run_b${B}/report.html"
echo "Checkpoint: validate_token.passed and validate_time ratios in that report."
echo "=========================================="
