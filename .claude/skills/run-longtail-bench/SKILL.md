---
name: run-longtail-bench
description: Use when running the rl-traces long-tail benchmark end to end — generate a trace, serve (or point at an existing endpoint), run the measurement client, analyze the results, and compare two configs or vLLM builds A/B.
---

# Run the long-tail benchmark

The loop is: **gen-trace → serve (optional) → run → analyze → compare**.

## 1. Generate a trace

```
rl-traces gen-trace --num-rollouts 512 --seed 0 --out t.jsonl
```

- Optional `--distribution <path.json>` to use a custom OSL/turn-count shape (default is a packaged example). See the `author-distribution` skill.
- Optional `--osl-level per_turn|per_rollout` controls whether the output-length target applies per turn or per whole rollout.
- This also writes `t.jsonl.stats.json` with realized percentiles for a sanity check.

## 2. Stand up an endpoint (optional)

Two options:

- **You host it**: `rl-traces serve --env baseline.env` (needs the `[serve]` extra and a GPU). This runs `vllm serve $VLLM_SERVE_ARGS` verbatim from the flat `.env` file and writes `serve_provenance.json` next to it for later comparison.
- **You already have an OpenAI-compatible URL**: skip this step and point `run` at it directly with `--url`.

`serve` is the **system under test** — it is not part of the measurement path.

## 3. Run the measurement client

```
rl-traces run --url localhost:8000 --trace t.jsonl --concurrency 512 \
  --tokenizer <hf-model-id> --out runs/baseline
```

- `run` is the **measurement client**: it replays the trace against the endpoint and records completion telemetry.
- **Concurrency == number of rollouts** — this is a static batch, not an open-loop load generator, so `--concurrency` should match `--num-rollouts` from step 1.
- Default endpoint is `/v1/completions` (`--endpoint-type completions`), which gives faithful exact-output-length replay. Override with `--endpoint-type chat --endpoint /v1/chat/completions` if you need chat-formatted requests.
- Pass `--vllm-src <path>` to stamp the editable vLLM build's provenance into `report.json`, which is what makes A/B comparisons trustworthy.

## 4. Analyze

`run` calls this for you automatically, but you can rerun it standalone:

```
rl-traces analyze --export runs/baseline/.../profile_export.jsonl \
  --out-json runs/baseline/report.json
```

If you generated the trace with a custom `--distribution <path.json>` in step 1, pass that same `--distribution <path.json>` to `analyze` (or `run`, which forwards it) so the token-domain validation gate checks against your distribution's anchors instead of the packaged example's — otherwise a faithful replay of a custom distribution can show a spurious `validate_token.passed: false`.

See the `interpret-longtail-report` skill for what the fields in `report.json` mean.

## 5. A/B two configs or vLLM builds

Repeat steps 2–4 with a second `.env` (different `VLLM_SERVE_ARGS`) or a second editable vLLM checkout (different `--vllm-src`), writing to a different `--out` directory. Then:

```
rl-traces compare baseline=runs/baseline/report.json mine=runs/mine/report.json
```

This prints a side-by-side diff of the key metrics so you can see whether the change helped or hurt the tail.
