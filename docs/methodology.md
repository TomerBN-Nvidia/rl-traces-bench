# Methodology

How `rl-traces-bench` turns a long-tail output-length profile into a replayable
trace, and how it checks the replay actually reproduced that profile.

## The problem with parametric distributions here

Long-tail generation-length data from real multi-turn rollouts is typically
**bimodal**: a large mass of short turns (tool calls, short replies) plus a
much smaller mass of very long ones (long-form generation, extended
reasoning). No standard single-family distribution (log-normal, Pareto, ...)
fits that shape well. Rather than pick one and accept the mismatch, this tool
builds an **inverse-CDF (quantile) sampler directly from measured percentile
anchors** and interpolates between them — so the realized distribution
matches by construction, not by assumption.

## Output-length (OSL) sampling

A distribution input is a JSON object with an `osl_anchors` list:

```json
{"osl_anchors": [[0.0, 1], [0.5, 654], [0.8, 10000], [0.9, 22000],
                 [0.95, 33212], [0.99, 57067], [1.0, 65489]]}
```

Each entry is `[cumulative_probability, output_tokens]` — "at the Pth
percentile, output length is this many tokens." To sample: draw
`u ~ Uniform(0, 1)`, find the bracketing anchor pair, and interpolate
**log-linearly** between the two token values (log space, because token
counts span multiple orders of magnitude and linear interpolation between a
handful of anchors would badly misrepresent the middle of each bracket).

The tool ships one example long-tail profile as the packaged default
distribution, at `src/rl_traces_bench/data/example_longtail.json` (mirrored
for easy reference/editing at `examples/distributions/example_longtail.json`).
Swap in your own with `--distribution <path>` — same schema, any anchors you
want, from any workload you're trying to reproduce.

### Per-turn vs per-rollout

`--osl-level per_turn` (default) draws an independent OSL for **each turn**
in a rollout — appropriate when the anchors describe a per-turn/per-generation
length. `--osl-level per_rollout` instead draws one OSL for the **whole
rollout** and splits it across that rollout's turns — appropriate when the
anchors describe a per-rollout total. Which interpretation matches your data
depends on how the source percentiles were measured; the `.stats.json`
sidecar written alongside every trace reports realized OSL percentiles **at
the configured level**, so you can check which one you meant to use.

## Multi-turn skeleton

Real multi-turn workloads (agent loops, RL rollouts) don't all have the same
number of turns. A distribution's `turn_counts` field is an array of
per-rollout turn counts — one integer per example rollout — sampled (with
replacement) to give each synthetic rollout in the trace a turn count drawn
from a realistic shape (e.g., "most rollouts run to a turn budget, some end
early") rather than a fixed constant. This is what makes the trace
genuinely multi-turn: each session's growing prefix produces real
prefix-cache hits across turns, not one giant single-turn request per
rollout.

## Input-length growth and prefix caching

Per-turn input length grows monotonically within a rollout: it's the shared
system-prompt tokens, plus every prior turn's user/tool input and the
model's own prior outputs, plus the current turn's new input. Each turn's
trace record carries block-aligned prefix-cache `hash_ids` — a superset of
the previous turn's — so replaying the trace against a prefix-caching-aware
server produces genuine cache hits on the growing shared prefix, and no
false hits across unrelated rollouts.

## Dual validation gate

A trace can be generated correctly but still fail to reproduce the intended
*served* behavior — e.g. if a chat template truncates generation early, or if
`aiperf`'s replay doesn't actually enforce the sampled output lengths. `run`
(via `analyze`) checks two independent things, and Phase 1 is only
considered validated if **both** pass:

- **Token-domain check** — realized OSL percentiles from the actual served
  responses (p50/p95/p99) reproduce the distribution's anchors within
  tolerance. This confirms the trace was replayed faithfully — the server
  actually generated approximately the lengths the trace asked for.
- **Time-domain check** — the *shape* of the per-rollout completion-time
  distribution (the p99/p50 and max/p50 ratios) matches a reference shape,
  not absolute times (which are model- and hardware-dependent and expected
  to vary run to run). This confirms the long tail actually shows up as a
  serving-time long tail, not just a token-count long tail — the thing the
  benchmark exists to measure.

Both checks are printed in `report.json` under `validate_token` and
`validate_time`.

## See also

- [`docs/metrics.md`](metrics.md) — the metrics computed from a replay.
- [`docs/quickstart.md`](quickstart.md) — running the pipeline end to end.
