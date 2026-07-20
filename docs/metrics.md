# Metrics

Definitions for every field `rl-traces analyze` (and therefore `rl-traces
run`) writes into `report.json`. All are computed over one static-batch
replay — a fixed set of rollouts (sessions) dispatched together, where the
batch isn't "done" until every rollout finishes.

## Core report fields

- **`makespan_s`** — wall-clock time from the first request in the batch to
  the last one finishing. The batch's total completion time; dominated by
  whichever rollout finishes last.
- **`completion_p50_s` / `completion_p90_s` / `completion_p99_s`** — the
  50th/90th/99th percentile of **per-rollout completion time** (each
  rollout's completion time = the end time of its last turn). Comparing
  p50 to p99 is how you see the long tail: a batch can have a fast p50 and
  a p99 many times larger.
- **`tail_bubble_s`** — `makespan − completion_p90_s`. The extra wall time
  spent waiting on the slowest ~10% of rollouts, past the point where 90% of
  the batch has already finished. This is the core "how much is the tail
  costing you" number — the metric `compare` sorts by.
- **`goodput_proxy`** — `mean(completion times) / makespan`. The fraction of
  total batch wall time that was, on average, "useful" work rather than idle
  waiting on stragglers. Close to `1.0` means most rollouts finish near the
  makespan (little bubble); a low value means the batch is straggler-
  dominated — most rollouts finished long before the batch as a whole did.
- **`output_tok_throughput`** — total generated output tokens across every
  request in the batch, divided by `makespan_s` (tokens/sec). A batch-level
  aggregate throughput number, derived from our own per-request records
  (`sum(osl) / makespan`) — independent of, and a useful cross-check against,
  `aiperf.output_token_throughput_tok_s` below.
- **`request_throughput`** — total number of requests (turns) in the batch
  divided by `makespan_s` (req/sec). Same cross-check relationship to
  `aiperf.request_throughput_req_s`.

## Validation gate fields

- **`validate_token`** — token-domain check: do realized **per-rollout**
  output-length percentiles match the *active distribution's* anchors within
  tolerance? `{"passed": bool, "realized": {...}, "checks": {...}}`.
  "Per-rollout" means each session's OSL is **summed across its turns**
  before taking percentiles — the published anchors describe the per-sample
  (whole-rollout) OSL distribution, so comparing against a per-turn OSL would
  be an apples-to-oranges comparison regardless of whether the trace was
  generated with `--osl-level per_turn` or `per_rollout`. The targets are the
  packaged example distribution's published percentiles (p50=654, p95=33212,
  p99=57067) by default; pass `--distribution <path>` to `analyze`/`run` to
  derive targets from your own distribution's `osl_anchors` instead, so a
  custom-distribution run doesn't get a spurious failure against the
  example's numbers.
- **`validate_time`** — time-domain check: does the *shape* of the completion-
  time distribution (p99/p50 and max/p50 ratios) match a reference shape?
  `{"passed": bool, "ratios": {...}, "ref_ratios": {...}}`. The reference
  ratios are fixed — calibrated to the packaged example workload's
  completion-time shape, not derived from `--distribution`. When you use a
  custom distribution this check is informational: a mismatch doesn't
  necessarily mean anything is wrong, just that your workload's tail shape
  differs from the example's.

See [`docs/methodology.md`](methodology.md) for what these checks are for.

## `aiperf` summary block

`report.json["aiperf"]` folds in `aiperf`'s own native summary
(`profile_export_aiperf.json`, written alongside the per-request export) —
its authoritative throughput/latency/TTFT/ITL numbers, computed by `aiperf`
itself rather than derived from our per-request records. `null` if that
summary file isn't found next to the export (e.g. an older `aiperf` or a
hand-built export).

- **`faithful`** — `error_request_count == 0`. This is a coherence gate, not
  just informational: a run with any request errors did not actually replay
  the full batch, so every other metric in the report (ours and `aiperf`'s)
  is measuring a partial, non-representative run. Treat `aiperf.faithful:
  false` the same way you'd treat `validate_token.passed: false` — don't
  trust the timing numbers until you've fixed whatever caused the errors and
  reproduced with `faithful: true`.
- **`error_request_count`** / **`request_count`** — raw counts backing the
  gate above.
- **`output_token_throughput_tok_s`** / **`request_throughput_req_s`** —
  `aiperf`'s own batch-level throughput numbers; compare against this
  report's `output_tok_throughput` / `request_throughput` (computed
  independently from our per-request records) as a cross-check — they should
  be close but aren't required to be identical (different measurement
  windows/rounding).
- **`request_latency_ms`**, **`time_to_first_token_ms`**,
  **`inter_token_latency_ms`** — `{"avg", "p50", "p90", "p99"}` dicts, keys
  present only if `aiperf` reported them.
- **`input_sequence_length`** / **`output_sequence_length`** — `{"avg",
  "max"}` dicts from `aiperf`'s own token accounting, a cross-check against
  `validate_token.realized`.
- **`prefix_cache`** — passthrough dict of `aiperf`'s `*cache*`-named metrics
  (e.g. cache hit rate). Only present when the server was run with
  `--enable-prompt-tokens-details`; `null` otherwise.
- **`aiperf_version`** — the `aiperf` version that produced the export, for
  provenance when comparing reports across `aiperf` upgrades.

## Per-request metrics (from the aiperf export)

These come from the underlying `aiperf` per-request export that `run`
consumes, not from `report.json`'s top-level fields, but are worth knowing
about when digging into a run:

- **TTFT (time to first token)** — latency from request start to the first
  streamed output token. Reported per request in the aiperf export, and
  summarized (avg/p50/p90/p99) in `report.json["aiperf"]["time_to_first_token_ms"]`.
- **ITL (inter-token latency)** — time between successive output tokens
  during generation. Reported per request in the aiperf export, and
  summarized in `report.json["aiperf"]["inter_token_latency_ms"]`.

## Prefix-cache hit rate

Available in `report.json["aiperf"]["prefix_cache"]` when the server runs
with `--enable-prompt-tokens-details` (see the `aiperf` block above);
otherwise `null`. If you need it without that flag, capture it separately
from your server's own metrics endpoint (e.g. vLLM's `/metrics`) during the
run. A higher prefix-cache hit rate generally means less redundant prefill
work as each rollout's turns grow the shared prefix.

## See also

- [`docs/methodology.md`](methodology.md) — how traces are calibrated and
  validated.
- [`docs/editable-vllm.md`](editable-vllm.md) — using `tail_bubble_s` and
  build provenance together to judge whether a vLLM change helped.
