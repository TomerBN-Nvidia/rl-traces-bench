# rl-traces-bench

A trace-driven, **static-batch** serving benchmark for long-tail LLM rollouts —
the workload shape you get from RL rollout generation, multi-turn agents, and
similar "fixed batch, wildly variable output length" traffic.

## What / why

When a batch of requests is dispatched together and the batch isn't "done"
until its *slowest* member finishes, a long-tail output-length distribution
turns a handful of stragglers into the thing that actually determines wall
time. A batch's p50 completion time can look fine while p99 (and the max) blow
out by an order of magnitude or more — and every other request in the batch
sits idle waiting on those stragglers. This shows up in RL rollout generation,
multi-turn agent traffic, and any other static-batch workload with heavy-tailed
generation lengths.

`rl-traces-bench` lets you reproduce that shape on purpose and measure it:

1. **Synthesize** a multi-turn trace whose per-turn output lengths are drawn
   from a calibrated long-tail distribution (`gen-trace`).
2. **Replay** it as a static batch against any OpenAI-compatible endpoint —
   your own vLLM server, an editable-vLLM build, or any other server that
   speaks the API (`run`).
3. **Analyze** the result into makespan / tail-bubble / goodput metrics, with
   a dual validation gate that checks the replay actually reproduced the
   intended distribution (`analyze`, done automatically by `run`).
4. **Compare** two or more configs — different vLLM flags, different builds —
   ranked by how much they shrink the tail (`compare`).

This is a serving-side static-batch harness (Phase 1). It does not orchestrate
an RL training loop.

## Install

```bash
pip install rl-traces-bench          # core: gen-trace, run, analyze, compare, doctor
pip install rl-traces-bench[serve]   # + the vLLM serve helper (adds vllm as a dependency)
pip install rl-traces-bench[dev]     # + pytest, for running the test suite
```

Requires Python 3.9+.

## Quickstart

No GPU required to see the pipeline work end to end:

```bash
git clone <this-repo> && cd rl-traces-bench
bash examples/smoke.sh   # mock OpenAI-compatible server -> gen-trace -> run -> report
```

Against a real model, the loop is:

```bash
rl-traces gen-trace --num-rollouts 512 --seed 0 --out t.jsonl
rl-traces serve --env .env &                          # boots vLLM (see "run vs serve" below)
rl-traces doctor --env .env                            # confirm the endpoint + setup are healthy
rl-traces run --url localhost:8000 --trace t.jsonl --concurrency 512 \
  --tokenizer nvidia/Llama-3_3-Nemotron-Super-49B-v1 --out results/
```

`results/report.json` has your makespan, tail bubble, goodput proxy, and the
dual validation gate result. See [`docs/quickstart.md`](docs/quickstart.md)
for the full copy-pasteable walkthrough.

## `run` vs `serve`

These are two different roles — most benchmark runs only need one of them:

- **`serve`** = the system under test. It boots vLLM (your build, your flags),
  holds the GPU, and listens on a port for as long as it runs. Run it **only
  when you need to host the model yourself** — for example, when you're
  measuring the effect of an editable vLLM change, or when no endpoint exists
  yet.
- **`run`** = the measurement client. It replays the trace against an
  OpenAI-compatible endpoint via `aiperf` and writes `report.json`. This is
  needed for **every** benchmark.

If you already have an endpoint running somewhere — your own server, someone
else's, a hosted API — **skip `serve` entirely** and just point
`rl-traces run --url <host:port>` at it.

The A/B loop for "did my vLLM change shrink the tail?" is: `gen-trace` once →
`serve` build X → `run` → edit vLLM → `serve` build Y → `run` → `compare`. See
[`docs/editable-vllm.md`](docs/editable-vllm.md) for the full walkthrough,
including how `report.json` provenance ties a measurement back to a specific
build.

## Distribution input

Trace generation is driven by one swappable distribution JSON — no code
changes needed to try a different long-tail shape:

```json
{
  "osl_anchors": [[0.0, 1], [0.5, 654], [0.9, 22000], [0.99, 57067], [1.0, 65489]],
  "turn_counts": [15, 21, 25, 30, 30, "..."]
}
```

- `osl_anchors` — `[cumulative_probability, output_tokens]` pairs. `gen-trace`
  builds an inverse-CDF sampler from these anchors (log-linear interpolation
  between them), so no parametric distribution family is assumed.
- `turn_counts` — an array of per-rollout turn counts, sampled to give each
  synthetic rollout a realistic multi-turn shape.

`rl-traces gen-trace` defaults to a packaged example long-tail profile
(`examples/distributions/example_longtail.json`) so you can run it with zero
setup; pass `--distribution <path>` to swap in your own. See
[`docs/methodology.md`](docs/methodology.md) for the calibration rationale.

## Docs

- [`docs/quickstart.md`](docs/quickstart.md) — no-GPU smoke test, then the
  real serve → run → analyze → compare loop.
- [`docs/editable-vllm.md`](docs/editable-vllm.md) — using this tool to A/B
  an editable vLLM checkout, with build provenance in every report.
- [`docs/methodology.md`](docs/methodology.md) — how traces are calibrated
  and validated.
- [`docs/metrics.md`](docs/metrics.md) — definitions of every metric in
  `report.json`.

## Running the tests

```bash
pip install -e '.[dev]'
python3 -m pytest -q
```

## License

Apache-2.0. See [`LICENSE`](LICENSE).
