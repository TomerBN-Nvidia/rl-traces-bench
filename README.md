# RL Long-Tail Serving Reproducer

## What / why

At Ultra scale, RL rollout generation has an extreme, bimodal long tail: a fixed
rollout batch's completion time is gated by its slowest rollout, so a handful of
stragglers dominate wall time and starve the trainer (p99/p50 token ratio ≈ 87×,
real batch p50≈84s vs p99≈909s vs max≈1,669s). This tool is a **trace-driven,
static-(fixed-)batch serving benchmark**: it synthesizes a multi-turn Mooncake
trace calibrated to the measured long tail, replays it against a real vLLM
server with `aiperf`, and reports the long-tail / goodput metric set so you can
**A/B different serving configs** (baseline vs MTP vs chunked-prefill, etc.) to
see which shrinks the tail the most. Phase 1 only — no RL-loop orchestration
yet (see "Future work").

Full design/rationale: [`specs/2026-07-19-rl-long-tail-reproducer-design.md`](specs/2026-07-19-rl-long-tail-reproducer-design.md).
Implementation plan + task-by-task history: [`specs/2026-07-19-rl-long-tail-reproducer-plan.md`](specs/2026-07-19-rl-long-tail-reproducer-plan.md).
Full session task log (investigation, bugs, fixes, rationale): [`docs/2026-07-20-task-log.md`](docs/2026-07-20-task-log.md).

## Repository structure

```
reproducer/
├── scripts/                     # pure-Python, stdlib-only, laptop-safe
│   ├── distributions.py         # OSL quantile sampler (calibration core)
│   ├── turn_structure.py        # turns-per-rollout sampler + OSL-split helper
│   ├── prompt_model.py          # per-turn ISL growth + cumulative prefix hash_ids
│   ├── gen_trace.py             # ASSEMBLES the Mooncake trace (the main entry point)
│   ├── metrics.py               # makespan / tail-bubble / goodput / percentiles
│   ├── analyze.py               # ingest aiperf export -> report + dual validation gate
│   ├── compare.py               # rank A/B configs by tail bubble
│   ├── tau2_estimator.py        # (optional) fit duration<->(ISL,OSL) from Tau2 data
│   └── probe_traces.py          # one-off parser for the source HTML rollout traces
├── data/
│   └── turn_counts.json         # REAL per-rollout turn counts (see below)
├── configs/                     # A/B serving-knob variants (sourced by the serve script)
│   ├── baseline.env             # prefix-caching on, no chunked prefill
│   ├── mtp.env                  # + MTP speculative decoding
│   └── chunked_prefill.env      # + chunked prefill
├── serve/
│   └── serve_super_bf16_hsg.sh  # (HSG) vllm serve for one config
├── run/
│   ├── run_batch.sh             # (HSG) aiperf static-batch replay -> analyze
│   ├── smoke_mock.sh            # (HSG) GPU-free harness smoke via aiperf mock server
│   └── hsg_static_batch.sbatch  # (HSG) one-allocation serve->health->gen->bench->analyze
├── tests/                       # pytest, stdlib-only (one test module per script)
│   └── fixtures/profile_export_sample.jsonl   # tiny synthetic aiperf export for tests
└── specs/                       # design spec + implementation plan
```

## How the pieces fit (data flow)

```
distributions.py ─┐
turn_structure.py ─┼─► gen_trace.py ──► trace.jsonl  ──(aiperf, HSG)──►  profile_export.jsonl
prompt_model.py  ─┘        │            (+ .stats.json)                         │
   (turn_counts.json) ─────┘                                                    ▼
                                                                     analyze.py ──► report.html
                                                                        │           report.json
                                                                        ▼               │
                                                              dual validation gate      ▼
                                                                                  compare.py (A/B rank)
```

`gen_trace.py` is the thing you run to **create a trace**. It draws output
lengths from `distributions.py`, a turn count from `turn_structure.py` (seeded by
the real `turn_counts.json`), and computes per-turn input lengths + prefix
`hash_ids` from `prompt_model.py`. The resulting `trace.jsonl` is what `aiperf`
replays on HSG; `analyze.py` then turns aiperf's per-request export into the
long-tail report, and `compare.py` ranks multiple reports.

## `data/turn_counts.json` — what it is

A JSON **array of 1,024 integers** — the observed number of turns for each of the
1,024 rollouts in the real coding-agent (OpenHands) rollout trace we were given
(`all_rollouts_timeline_first_rollout_zero.html`, "File-1"). Example:

```json
[15, 21, 25, 26, 27, 29, 29, 29, 29, 30, ...]   // len 1024, min 15, max 30
```

- **Where it came from:** extracted once from File-1's Plotly `model_call` bars
  by `scripts/probe_traces.py` (max `Turn` index seen per `Rollout Id`). It is
  the only *real* structural signal we have from production rollouts — the HTMLs
  contained per-turn **durations** but **no token counts**, so tokens are
  synthesized and only the turn *shape* is real.
- **What it's for:** `gen_trace.py` samples a turn count from this array for each
  synthetic rollout (via `turn_structure.sample_turn_count`), so synthetic
  rollouts have a realistic multi-turn shape (mostly 30-turn agent runs, some
  shorter) instead of one giant single-turn request. This is what makes the
  benchmark **multi-turn** and lets prefix caching matter.
- **To regenerate it** (only needed if the source trace changes), see
  `probe_traces.py` and Task 2 in the plan.

## Creating a trace (the important part)

`gen_trace.py` writes a **Mooncake-format** JSONL trace — one JSON object per
request (= one turn), which is exactly what aiperf's
`--custom-dataset-type mooncake_trace` consumes.

```bash
cd reproducer
PYTHONPATH=. python3 scripts/gen_trace.py \
  --num-rollouts 512 \          # B: number of rollouts (= sessions = static-batch size)
  --osl-level per_turn \        # per_turn (default) or per_rollout — see below
  --seed 0 \                    # deterministic: same seed -> identical trace
  --block-size 512 \            # prefix-cache block size (match aiperf --isl-block-size)
  --system-tokens 300 \         # shared system-prompt length (modeled)
  --user-turn-tokens 200 \      # per-turn incoming user/tool-result length (modeled)
  --shared-blocks 1 \           # how many leading blocks are globally shared (system)
  --turn-counts data/turn_counts.json \
  --out mytrace.jsonl
# -> writes mytrace.jsonl  AND  mytrace.jsonl.stats.json
```

### Trace record format

Each line is one turn:

```json
{"session_id": "0", "turn_idx": 0, "timestamp": 0, "input_length": 500,  "output_length": 6823, "hash_ids": [0]}
{"session_id": "0", "turn_idx": 1, "timestamp": 0, "input_length": 7523, "output_length": 234,  "hash_ids": [0, 1000001, ..., 1000014]}
{"session_id": "0", "turn_idx": 2, "timestamp": 0, "input_length": 7957, "output_length": 29,   "hash_ids": [0, 1000001, ..., 1000015]}
```

- `session_id` — one per rollout; all turns of a rollout share it.
- `turn_idx` — 0-based turn within the rollout.
- `timestamp` — `0` for all: it's a **static batch** (every rollout available at
  t=0; aiperf sequences a session's turns as dependent requests).
- `input_length` (ISL) — grows monotonically: `system + Σ(prior user msgs +
  prior assistant outputs) + this turn's user msg`. In the example above:
  `500 = 300+200`, then `7523 = 500 + 200 + 6823` (folds in turn-0's output),
  then `7957 = 7523 + 200 + 234`.
- `output_length` (OSL) — drawn from the calibrated long-tail distribution (turn
  0 here drew a 6,823-token straggler; later turns are short).
- `hash_ids` — block-aligned prefix-cache identifiers, `ceil(ISL/block_size)` of
  them. The first `shared_blocks` are **global** (id `0` = the shared system
  prompt, cached across the whole batch); the rest are **per-rollout**
  (`rollout_base + block_index`, where `rollout_base = (session_id+1)*1_000_000`).
  Each turn's block set is a **superset** of the previous turn's, so within a
  session the growing prefix produces genuine prefix-cache hits; different
  rollouts use disjoint id ranges so there's no false cross-rollout sharing.

### `--osl-level`: per_turn vs per_rollout

- `per_turn` (default) — **each turn** draws an OSL from the published
  distribution. Consistent with the 654-token median being a plausible per-turn
  length, and it's corroborated by File-1's per-turn duration tail (p50≈3.6s →
  max≈93.5s). This creates per-turn stragglers, which is the serving-side tail we
  benchmark.
- `per_rollout` — draw one OSL **total** per rollout from the distribution, then
  split it across turns (`turn_structure.split_osl`). Use this if you interpret
  the published "per sample" figure as a per-rollout total.

Which interpretation is "right" is an open question (design §10 Q2); the default
is `per_turn`. The `.stats.json` sidecar reports realized OSL percentiles **at
the configured level** so you can check calibration.

### The `.stats.json` sidecar

```json
{"num_rollouts": 512, "num_records": 14983, "osl_level": "per_turn",
 "osl_p50": 654, "osl_p90": 22000, "osl_p95": 33212, "osl_p99": 57067, "osl_max": 65489}
```

These realized percentiles should track the published anchors (p50=654,
p95=33,212, p99=57,067) — that's the token-domain half of the validation gate.
Percentiles converge with `--num-rollouts`; small traces (e.g. B=4) are noisy.

## Execution locality — read this first

- **Laptop-safe:** every `scripts/*.py` module is stdlib-only; unit tests and
  **trace generation** run fine on a Mac (no GPU/aiperf).
- **HSG-only:** `aiperf` is not installed locally and nothing that talks to a
  served model runs on the laptop. `serve/serve_super_bf16_hsg.sh`,
  `run/run_batch.sh`, `run/smoke_mock.sh` (the GPU-free mock smoke still needs
  aiperf, so it runs on HSG), and `run/hsg_static_batch.sbatch` all run on HSG.

## Full pipeline (HSG)

Steps marked **(HSG)** run on an HSG node/container.

```bash
cd reproducer

# 1. Create the trace (laptop-safe) — see "Creating a trace" above
PYTHONPATH=. python3 scripts/gen_trace.py --num-rollouts 512 --osl-level per_turn \
  --seed 0 --out <TRACE_OUT>/trace_b512.jsonl

# 2. (HSG) recommended first: GPU-free harness smoke via aiperf's mock server —
#    validates trace -> aiperf -> analyze end to end AND confirms aiperf's real
#    per-request field names (update analyze.py's alias map if they differ).
bash run/smoke_mock.sh

# 3. (HSG) serve the target model for a config
bash serve/serve_super_bf16_hsg.sh configs/baseline.env   # or mtp / chunked_prefill

# 4. (HSG) replay the trace as a static batch (concurrency == B; exact OSL via ignore_eos/min_tokens)
bash run/run_batch.sh <TRACE_OUT>/trace_b512.jsonl 512 <RUN_OUT_DIR>

# 5. analyze -> report.html + report.json + metrics + dual validation gate
PYTHONPATH=. python3 scripts/analyze.py --export <RUN_OUT_DIR>/profile_export.jsonl \
  --out-html <RUN_OUT_DIR>/report.html --out-json <RUN_OUT_DIR>/report.json

# 6. compare configs once you have >=2 report.json files (ranks by tail bubble)
PYTHONPATH=. python3 scripts/compare.py \
  baseline=<RUN_OUT_BASELINE>/report.json mtp=<RUN_OUT_MTP>/report.json
```

Single-allocation orchestrator (serve → wait-for-health → gen_trace → run_batch →
analyze in one sbatch, avoiding separate queue waits):

```bash
MODEL=<SUPER_BF16_CKPT_PATH> CONTAINER=<VLLM_CONTAINER_OR_SQSH> \
OUT_DIR=<RESULTS_DIR> MOUNTS=<HOST>:<CONTAINER> \
sbatch --account=<ACCOUNT> --partition=batch --qos=<QOS> \
  run/hsg_static_batch.sbatch configs/baseline.env
```

If you submit under both Slurm accounts to reduce queue wait, submit duplicates
under both accounts; never cancel a running job.

**First real HSG run** fills the placeholders (`<SUPER_BF16_CKPT_PATH>`, and the
MTP `"method"` in `configs/mtp.env`) and confirms two things the analyzer is
built to absorb: aiperf's real per-request field names (alias map in
`analyze.load_profile_export`) and how aiperf sequences dependent multi-turn
requests.

## Metric definitions

Computed per static-batch run by `metrics.py` / `analyze.py`:

- **Makespan** — wall time of the slowest rollout (the batch isn't done until
  every rollout finishes).
- **Per-rollout completion p50 / p90 / p99** — distribution of per-session
  completion times within the batch.
- **Tail bubble** — `makespan − p90`; the extra wall time spent waiting on the
  tail beyond the 90th-percentile rollout.
- **Goodput proxy** — `mean_completion / makespan`; fraction of batch wall time
  that is "useful" (→1.0 is better; low = straggler-dominated).
- **Prefix-cache hit rate** — capture manually from the served vLLM's `/metrics`
  during the run (automated scrape is future work; not in the aiperf export).
- **TTFT / ITL** — from the aiperf per-request export.
- 

## Calibration inputs

Three grounded sources feed the synthesis:

- **Published OSL percentiles** (Sean Choi, per-sample generated tokens) — the
  authoritative long tail: p50=654, p80<10,000, p95=33,212, p99=57,067,
  max≈65,489 (p90≈22,000 is estimated). `distributions.py` builds a quantile
  function from these anchors (log-linear interpolation; no assumed parametric
  family, since the real distribution is bimodal).
- **Real coding-agent turn skeleton** (`data/turn_counts.json`) — see above.
- **Optional Tau2 token↔time estimator** (`tau2_estimator.py`) — fits
  `duration ≈ ttft_a + ttft_b·ISL + itl·OSL` from a prior Tau2 run's per-turn
  `usage.prompt_tokens` / `usage.completion_tokens` / `generation_time_seconds`,
  then inverts durations into estimated ISL/OSL as a cross-check. Optional /
  non-blocking; fitting real HSG Tau2 data is a later step.

## Dual validation gate

`analyze.py` runs two independent checks; Phase 1 is validated only if both pass:

- **Token domain** (`validate_token_domain`) — realized OSL percentiles reproduce
  the published distribution within tolerance (p50=654 / p95=33,212 / p99=57,067).
- **Time domain** (`validate_time_domain`) — the *served* per-rollout completion
  distribution's SHAPE (p99/p50 and max/p50 ratios) matches the real
  `nemorl_trace` rollout-time shape (p50≈84s, p99≈909s, max≈1,669s). A ratio/shape
  match, not absolute times (which are model/config-dependent).

## Running the tests

Stdlib-only; only external dep is `pytest`. Python 3.9+.

```bash
cd reproducer
PYTHONPATH=. python3 -m pytest -v      # expect 23 passing
```

## Future work

- **HTML visualization / dashboard for analyzer + compare.** Today `analyze.py`
  emits only a plain `report.html` table and `compare.py` prints to stdout. Build
  a self-contained interactive HTML report (e.g. Plotly, or the team's
  `generate_traces.py` idiom) that renders: the per-rollout **completion-time CDF
  + histogram** (the tail), stat tiles for makespan / tail-bubble / goodput /
  throughput, the **OSL fidelity** + ISL/OSL distributions, prefix-cache hit rate,
  and an optional per-turn **timeline (Gantt)** of one batch. Plus an **A/B compare
  view**: configs ranked by tail bubble, goodput/throughput bars, and overlaid
  completion CDFs — so two coherent runs can be eyeballed side by side. Should
  consume the existing `report.json` (+ aiperf's native summary) so no re-run is
  needed.
- **Ingest aiperf's native summary** (`profile_export_aiperf.json`) into
  `report.json` instead of recomputing throughput/latency — one report with
  aiperf's authoritative perf metrics + our rollout-level tail metrics.
- **Phase 2 — RL-loop orchestration.** Layer the RL loop on top of this
  static-batch harness: lag-1 admission control and a refit-bubble model, measured
  against the real goodput anchors (buffer-starvation ≈49%, refit bubble ≈120s) —
  see design spec §3 and §11. The Phase-1 static batch maps directly onto a
  lag-1-gated fixed batch.
```
