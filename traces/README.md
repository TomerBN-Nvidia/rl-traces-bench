# Generated traces

The trace `*.jsonl` files here are **git-ignored** (large, and fully reproducible
from `--seed`). Only the `*.stats.json` files are committed — sample realized-stats
sidecars from example runs. To recreate the traces, run the commands below
(laptop-safe, no GPU/aiperf).

## Canonical traces

```bash
# production-like static batch (32 prompts x 16 gen = 512), default per_turn
rl-traces gen-trace --num-rollouts 512  --osl-level per_turn \
  --seed 0 --out traces/trace_b512_per_turn_seed0.jsonl        # ~31 MB, 15,355 records

# per_rollout interpretation variant (OSL total per rollout, split across turns)
rl-traces gen-trace --num-rollouts 512  --osl-level per_rollout \
  --seed 0 --out traces/trace_b512_per_rollout_seed0.jsonl     # ~3.6 MB, 15,333 records

# tiny trace for the mock-server smoke (examples/smoke.sh)
rl-traces gen-trace --num-rollouts 8    --osl-level per_turn \
  --seed 0 --out traces/smoke_b8_per_turn_seed0.jsonl          # ~0.36 MB, 240 records
```

## Calibration (token-domain gate) — PASS

Realized per-turn OSL percentiles vs the published anchors (p50=654, p95=33,212,
p99=57,067, max≈65,489). Larger `--num-rollouts` → tighter match:

- **B=4096, per_turn:** p50=643, p95=33,108, p99=57,265, max=65,482 (within ~1%).
- **B=512, per_turn:**  p50=654, p95=32,132, p99=57,003, max=65,401.
- B=512, per_rollout (rollout-total shape, expected to differ): p50=802, p95=29,918, p99=61,199.

(Exact numbers in the committed `*.stats.json` sidecars.)

## Size note

`per_turn` traces are large because each turn lists all its prefix-cache block
`hash_ids`, and accumulated ISL (hence block count) grows across turns; big-`B`
`per_turn` traces balloon (B=4096 ≈ 283 MB). For bigger batches prefer
`per_rollout` (smaller per-turn ISL → far smaller files) or generate on the target
host. A compact `hash_ids` encoding is possible future work.

## Getting a trace onto a remote serving host

`scp` the `*.jsonl` to the run host, or regenerate it there with the same
`rl-traces gen-trace` seed + params (byte-identical trace either way).
