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

## Validation gate fields

- **`validate_token`** — token-domain check: do realized output-length
  percentiles (from the actual served responses) match the distribution's
  anchors within tolerance? `{"passed": bool, "realized": {...}, "checks": {...}}`.
- **`validate_time`** — time-domain check: does the shape of the completion-
  time distribution (p99/p50 and max/p50 ratios) match a reference shape?
  `{"passed": bool, "ratios": {...}, "ref_ratios": {...}}`.

See [`docs/methodology.md`](methodology.md) for what these checks are for.

## Per-request metrics (from the aiperf export)

These come from the underlying `aiperf` per-request export that `run`
consumes, not from `report.json`'s top-level fields, but are worth knowing
about when digging into a run:

- **TTFT (time to first token)** — latency from request start to the first
  streamed output token. Reported per request in the aiperf export.
- **ITL (inter-token latency)** — time between successive output tokens
  during generation. Reported per request in the aiperf export.

## Prefix-cache hit rate

Not currently part of the aiperf export or `report.json` — capture it
separately from your server's own metrics endpoint (e.g. vLLM's `/metrics`)
during the run if you want to correlate cache behavior with the tail. A
higher prefix-cache hit rate generally means less redundant prefill work as
each rollout's turns grow the shared prefix.

## See also

- [`docs/methodology.md`](methodology.md) — how traces are calibrated and
  validated.
- [`docs/editable-vllm.md`](editable-vllm.md) — using `tail_bubble_s` and
  build provenance together to judge whether a vLLM change helped.
