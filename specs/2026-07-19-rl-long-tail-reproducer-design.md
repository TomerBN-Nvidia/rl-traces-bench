# 2026-07-19 — RL long-tail serving reproducer (design spec)

**Status:** APPROVED (framing) — pre-implementation.
**Parent task:** `vllm_rl_long_tail_task.md` (team handoff, measurements, acceptance criteria) — internal doc, not included in this standalone repo.
**Branch:** `tbn/rl-long-tail-reproducer` → PR.

## 1. Problem & motivation

At Ultra scale, RL rollout generation has an extreme, bimodal long tail. A fixed rollout
batch's completion time is gated by its slowest rollout, so a few stragglers dominate wall
time and starve the trainer. Measured evidence:

- **Token tail** (Sean Choi, per-sample generated tokens): p50=654, mean=5,355, p95=33,212,
  p99=57,067, max≈65,489; 80% of rollouts < 10k tokens. p99/p50 ≈ 87×.
- **Time tail** (`nemorl_trace.html`, real 2026-03-16 run, `timing/rollout/total`, n=541):
  p50=84s → p90=591s → p99=909s → max=1,669s (≈28 min); `await_results` (straggler wait)
  p90=166s → max=1,664s.
- **Goodput breakdown** (same trace `SUMMARY`): `exposed_generation` ≈50% of step time,
  `idle/buffer_starvation` ≈49%, `idle/refit_bubble` ≈120s each (MXFP8), `weight_sync` mirrors it.

We need a repeatable reproducer to (a) mimic the long tail against a real vLLM server,
(b) measure where we stand, and (c) A/B serving approaches to see which shrinks the tail most.

## 2. Goal & Definition of Done

A trace-driven, **static-(fixed-)batch** serving benchmark that reproduces the RL long tail
against a real vLLM server, reports the long-tail / goodput metric set, and compares serving
configurations. DoD: a working way to mimic the long tail, measure our performance, and
validate different approaches — without needing the full RL loop yet.

## 3. Scope & phasing

- **Phase 1 (this build) — serving-level, synthetic trace.** aiperf replays a synthetic
  Mooncake trace at a served model in static-batch mode. Measure makespan, tail bubble,
  goodput proxy, cache hit rate; A/B serving knobs.
- **Phase 2 (later) — RL-loop orchestration.** Layer fixed-batch + lag-1 admission +
  simulated async-refit pauses on top; measure the admission bubble against the real goodput
  anchors (§7). Out of scope for Phase 1 but the Phase-1 static batch maps directly onto it.
- **Trace source:** synthetic first (calibrated to measured distributions), then swap in a
  recorded real trace to confirm the synthetic matches. Generator is built swappable so a
  recorded Mooncake trace drops in with no harness change.

## 4. Data sources (what each provides)

| Source | Provides | Used for |
| --- | --- | --- |
| Sean Choi published percentiles | per-sample OSL distribution | **calibrate synthetic OSL** |
| `all_rollouts_timeline_first_rollout_zero` (File 1) | per-turn (Rollout Id, Turn, Start, Duration); turns/rollout 15–30, capped 30; coding-agent (OpenHands) | **turn-structure skeleton** |
| `nemorl_trace.html` (File 2) | rollout-time distribution + goodput phase breakdown (no tokens; `meta` null) | **time-domain validation anchor + Phase-2 goodput targets** |
| Tau2 / agentic per-request runs (local candidate + HSG results) | per-turn `usage.prompt_tokens` (ISL), `usage.completion_tokens` (OSL), `generation_time_seconds`, `turn_idx`, `timestamp` | **token↔time estimator (planned, §6.1)** |

Neither HTML contains ISL/OSL token counts or prefix-cache signal — confirmed by extraction.
That is why Phase 1 synthesizes rather than replays literal recorded tokens. A prior Tau2 run
*does* carry per-turn tokens+time and can ground the synthesis (§6.1).

## 5. Architecture / components

All under `vllm-rl-long-tail-goodput/reproducer/`:

1. **`scripts/gen_trace.py`** — synthesizes a Mooncake JSONL trace of `B` rollouts (= sessions),
   each a multi-turn conversation. Per turn emits `input_length`, `output_length`, `hash_ids`
   (cumulative → prefix-cache reuse), `timestamp`. Deterministic (`--seed`). Emits a sidecar
   `trace_stats.json` with the realized OSL percentiles for the §6 self-check.
2. **`scripts/run_batch.sh`** — launches aiperf against a served model in static-batch mode:
   `--custom-dataset-type mooncake_trace --input-file <trace> --concurrency B`, `ignore_eos`
   for exact OSL. One serving config per run; writes `profile_export.jsonl`.
3. **`scripts/analyze.py`** — ingests aiperf `profile_export.jsonl` + server Prometheus,
   computes the §6 metric set, and emits a compact HTML/table (reuse the team's
   `generate_traces.py` HTML idiom). Also runs the dual validation (§6).
4. **`configs/`** — serving-knob variants for A/B: `baseline`, `mtp`/dynamic-draft,
   `chunked-prefill`, `dp-routing`.
5. **`serve/`** — vLLM serve launcher for the target model (§8), plus an aiperf mock-server
   smoke path for harness validation with no GPU.

## 6. Synthesis model & validation (the technical core)

- **OSL** sampled per-rollout from a distribution calibrated to the measured percentiles
  (p50=654, p95=33,212, p99=57,067, max≈65,489, 80%<10k) — lognormal or two-component mixture,
  fit and frozen. This is the authoritative long-tail source.
- **Turn structure** (turns/rollout, per-turn OSL split) drawn from File-1's empirical skeleton
  (15–30 turns, capped 30); longer total OSL biases toward more/longer turns.
- **ISL per turn** = accumulated conversation context (system + prior turns) → monotonically
  growing prompt → **cumulative `hash_ids`** so prefix caching is exercised turn-over-turn.
  Block size aligned to `--isl-block-size` (default 512).
- **Arrival:** all `B` sessions available at t=0 (static batch); intra-session turn N+1 arrives
  after turn N completes (dependent turns) — closed-loop within a session.

### 6.1 Planned refinement — Tau2-grounded token↔time estimator

A prior Tau2/agentic run logged per-turn `usage.prompt_tokens` (ISL),
`usage.completion_tokens` (OSL), and `generation_time_seconds`. **As a planned task (not
Phase-1-blocking):** fit `duration ≈ TTFT(ISL) + OSL·ITL` from that data, then invert File-1's
per-turn *durations* into estimated ISL/OSL. Uses:
- A **third calibration input** — the estimator's per-turn OSL/ISL and turn-growth pattern,
  cross-checked against the published percentiles (they should agree on tail shape).
- An optional **alternative synthesis path** — reconstruct a near-real trace from File-1's
  actual turn skeleton + durations, instead of pure distribution sampling.

Caveats: File-1 is a coding agent, Tau2 is a different workload; absolute times are
model/config-dependent, so transfer the *relationship*, not the raw times. Source data:
locate the local Tau2 per-turn artifact (candidate spotted) and/or pull fuller per-request
results from HSG. Sequenced in the implementation plan.

**Dual validation gate (Phase 1 success = the "does synthetic match real" check):**
1. **Token domain:** generated trace OSL percentiles reproduce the published distribution
   within tolerance.
2. **Time domain:** the *served* per-rollout completion distribution matches the real
   `timing/rollout/total` shape (p50≈84s, p99≈909s, max≈1,669s) — shape/ratio match, since
   absolute times depend on the served model/config.

## 7. Metrics (the "measure ourselves" DoD)

Per static-batch run: **makespan** (= slowest rollout completion); per-rollout completion
p50/p90/p99/max; **tail bubble** = `makespan − p90`; **goodput proxy** = `mean_completion /
makespan`; TTFT / ITL / request-latency percentiles; **prefix-cache hit rate** (server
metrics, cross-checked). Per-run JSONL retained for cross-config comparison. Phase-2 goodput
anchors from §1 (refit bubble ≈120s, buffer-starvation ≈49%, await-results tail).

## 8. Defaults / environment

- **Target:** Super BF16, single node on HSG (consistent with `nemorl-perf-harness` and
  `dynamic-draft-length-tau2`). aiperf mock-server for harness smoke; real Super for measurement.
- **Batch size:** `B=512` default (March config: 32 prompts × 16 gen), parameterized (the
  thread called 512 "too small" — sweep upward for production-like runs).
- **Tooling:** aiperf Mooncake trace replay (`--fixed-schedule` auto-on with timestamps;
  closed-loop concurrency for the static batch). Stock latest vLLM, no fork.

## 9. A/B knobs to validate approaches

MTP / dynamic draft length (ties to `dynamic-draft-length-tau2`); chunked prefill /
`max-num-seqs`; DP + length-aware routing vs round-robin (needs multiple engines);
heterogeneous parallelism for predicted-long requests (Phase 2-leaning).

## 10. Open questions (settle during planning / early runs)

1. Confirm aiperf's `hash_id ⇒ identical tokens ⇒ real cache hit` guarantee against source
   before trusting exact hit ratios (flagged in the aiperf feasibility research).
2. Is the published OSL "per sample" a per-rollout total or per-generation? Affects the
   per-turn split; default = per-rollout total distributed across turns.
3. How aiperf sequences dependent multi-turn requests vs fixed-schedule timestamps for a
   session (docs ambiguous) — validate with a 2-turn smoke.
4. Confirm the exact Super BF16 single-node config + checkpoint path on HSG.

## 11. Non-goals (Phase 1)

Modeling lag-1 admission, async refit, or the trainer loop (Phase 2). Real agentic harness
(Tau2/TerminalBench) capture (Phase-1.5 / real-trace swap). Any vLLM code change.
