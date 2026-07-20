---
name: interpret-longtail-report
description: Use when reading or interpreting a rl-traces `report.json` — explains makespan, completion percentiles, tail bubble, goodput proxy, and the validation blocks, and how to judge whether a config change helped.
---

# Interpret a long-tail report.json

Fields to read, in order:

- **`makespan_s`** — wall-clock time from batch start until the last rollout finishes. The top-line number: smaller is better.
- **`completion_p50_s` / `completion_p90_s` / `completion_p99_s`** — per-rollout completion-time percentiles. These show the shape of the distribution, not just the tail.
- **`tail_bubble_s`** — `makespan_s - completion_p90_s`. This is the idle time the batch spends waiting on the last ~10% of rollouts after most of the work is already done. It is the single best number for "how bad is my long tail." Smaller is better; near zero means the batch finishes evenly.
- **`goodput_proxy`** — mean completion time divided by makespan (`mean / makespan_s`). A value close to **1.0 is good**: it means most rollouts finish close to when the batch as a whole finishes, i.e. little wasted tail time. Values well below 1.0 mean a lot of the batch is idle waiting for stragglers.
- **`output_tok_throughput` / `request_throughput`** — batch-level tokens/sec and requests/sec (`sum(osl)/makespan_s`, `num_requests/makespan_s`), computed from our own per-request records. Cross-check these against `aiperf.output_token_throughput_tok_s` / `aiperf.request_throughput_req_s` below — they should be close.
- **`validate_token.passed`** — whether the realized **per-rollout** output-token distribution (each session's OSL summed across its turns, then percentiled) matched the active distribution's `osl_anchors` within tolerance. Targets default to the packaged example distribution's published percentiles; if the trace was generated from a custom `--distribution`, `analyze`/`run` should have been passed the same `--distribution <path>` so the targets match — otherwise you're comparing a custom-distribution replay against the example's numbers, which can fail spuriously. If this is `false`, don't trust the timing numbers yet — the trace itself didn't replay faithfully (check `validate_token.checks` / `.realized` for which anchor drifted).
- **`validate_time.ratios`** — shape ratios (`p99/p50`, `max/p50`) of completion times, compared against `validate_time.ref_ratios`. These reference ratios are fixed (calibrated to the packaged example workload) regardless of `--distribution`, so `validate_time.passed` is informational rather than a hard gate when you're running a custom distribution — it's still useful for sanity-checking that the benchmark is exercising a long tail and not a flat, uniform load.
- **`aiperf.faithful`** — `aiperf`'s own coherence gate: `true` only if `aiperf.error_request_count == 0`. Check this **first** — if `false`, the run didn't replay the full batch and nothing else in the report (ours or `aiperf`'s) is trustworthy. `aiperf` also carries its own native `output_token_throughput_tok_s` / `request_throughput_req_s` / latency / TTFT / ITL summaries, useful as an independent cross-check against this report's own numbers, plus `aiperf.prefix_cache` (cache-hit metrics, only present when the server ran with `--enable-prompt-tokens-details`); `null` if `aiperf` didn't write a summary file next to the export.

## Judging whether a change helped

When comparing two reports (e.g. via `rl-traces compare baseline=... mine=...`), the read is:

> **Lower `tail_bubble_s` + higher `goodput_proxy` = the config helped.**

Also check that `makespan_s` moved in the same direction and that both reports have `validate_token.passed: true` and `aiperf.faithful: true` — otherwise you may be comparing runs that replayed different (or broken) effective workloads, which invalidates the comparison.
