---
name: author-distribution
description: Use when authoring a custom `--distribution` JSON file to calibrate rl-traces gen-trace to your own workload's output-length and turn-count shape.
---

# Author a custom `--distribution` file

## Schema

```json
{
  "osl_anchors": [[cum_prob, tokens], ...],
  "turn_counts": [int, ...]
}
```

- **`osl_anchors`** — a list of `[cumulative_probability, output_tokens]` pairs describing the output-sequence-length (OSL) CDF. Must be sorted by `cum_prob`, start near `0.0` and end at `1.0`, with `tokens` monotonically non-decreasing. `gen-trace` samples an OSL for each rollout (or turn, per `--osl-level`) by inverse-CDF interpolation between anchors — this is how the long tail gets shaped (e.g. most anchors clustered at low token counts, then a few high-percentile anchors stretching out to a much larger value).
- **`turn_counts`** — a flat array of integers; `gen-trace` samples a turn count per rollout uniformly from this array. Repeat a value to weight it higher (that's why the packaged example has 30 repeated many times: it's the modal turn count).

## Workflow

1. Start from the packaged example:

   ```
   cp examples/distributions/example_longtail.json my_dist.json
   ```

2. Edit `osl_anchors` to match your workload — replace the `[cum_prob, tokens]` pairs with anchors read off your own OSL distribution (e.g. p50, p80, p90, p95, p99, p100 output-token counts). Keep `cum_prob` monotonically increasing from `0.0` to `1.0` and `tokens` non-decreasing.

3. Edit `turn_counts` to reflect your workload's turn-count distribution (a list of observed/expected turn counts; repeats encode weight).

4. Regenerate the trace with your distribution:

   ```
   rl-traces gen-trace --num-rollouts 512 --seed 0 \
     --distribution my_dist.json --out t.jsonl
   ```

5. Check the fit: `gen-trace` writes `t.jsonl.stats.json` alongside the trace. Open it and confirm the realized percentiles track the anchors you specified — if a percentile is far off, your anchors are too sparse near that percentile; add an intermediate anchor and regenerate.
