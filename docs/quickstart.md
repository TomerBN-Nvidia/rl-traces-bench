# Quickstart

Two paths: a no-GPU smoke test that proves the pipeline works, and the real
path against a served model.

## 1. No-GPU smoke test

`examples/smoke.sh` runs the whole pipeline against a minimal mock
OpenAI-compatible server — no GPU, no vLLM, no external services.

```bash
git clone <this-repo> && cd rl-traces-bench
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

bash examples/smoke.sh
```

It starts `examples/mock_server.py`, generates an 8-rollout trace, runs it
through `rl-traces run`, and prints `SMOKE OK` once `report.json` (written to
`/tmp/rl_traces_smoke/report.json` by default; override with the `OUT` env
var) exists. Use this to sanity-check an install or a change to the pipeline
before touching a real GPU.

Note: the mock server only implements the chat-completions shape, which
matches the CLI's default (`--endpoint-type chat --endpoint
/v1/chat/completions`) — chat is required for multi-turn replay, so no
override is needed.

## 2. The real path

This needs `aiperf` on your `PATH` and a served OpenAI-compatible model
(any server that speaks the API works, not just vLLM — but the `serve`
subcommand and `[serve]` extra are vLLM-specific).

```bash
pip install rl-traces-bench[serve]
```

### a. Generate a trace

```bash
rl-traces gen-trace --num-rollouts 512 --seed 0 --out t.jsonl
```

Deterministic: the same `--seed` and distribution always produce the same
trace. Writes `t.jsonl.stats.json` alongside it with realized OSL
percentiles.

### b. Serve the model

Copy `.env.example` to `.env` and fill in your model:

```bash
cp .env.example .env
```

```
VLLM_SRC=/path/to/your/vllm
VLLM_SERVE_ARGS=nvidia/Llama-3_3-Nemotron-Super-49B-v1 --tensor-parallel-size 4 --port 8000 --enable-prefix-caching --no-enable-chunked-prefill
TOKENIZER=nvidia/Llama-3_3-Nemotron-Super-49B-v1
URL=localhost:8000
```

Any HF repo id your vLLM build can serve works here; the example above uses
a real, public, vLLM-ready one.

```bash
rl-traces serve --env .env &
```

This runs `vllm serve $VLLM_SERVE_ARGS` verbatim and writes
`serve_provenance.json` (vLLM version + `VLLM_SRC` git SHA/dirty flag) next
to your `.env`.

If you already have an endpoint running somewhere else, skip this step
entirely — see the "run vs serve" section in the top-level [README](../README.md).

### c. Check the setup

```bash
rl-traces doctor --env .env
```

Prints a checklist built from what your `.env` says you're doing: `aiperf`
on `PATH` and `TOKENIZER` set are always checked; if `URL` is set it probes
the endpoint; if `VLLM_SERVE_ARGS` is set (i.e. you're serving) it also
checks that `vllm` is importable and `VLLM_SRC` points at a real directory.
Each line prints a fix hint if it fails, and `doctor` exits non-zero if
anything is red. If you skipped `serve` and pointed `run` at an existing
endpoint instead, `doctor` won't demand a vLLM install — it just checks the
endpoint is reachable.

### d. Run the benchmark

```bash
rl-traces run --url localhost:8000 --trace t.jsonl --concurrency 512 \
  --tokenizer nvidia/Llama-3_3-Nemotron-Super-49B-v1 --out results/
```

`--concurrency` should match your trace's rollout count for a true static
batch. `run` calls `aiperf profile` under the hood, then automatically
analyzes the export and writes `results/report.json` and
`results/report.html`. Pass `--vllm-src /path/to/your/vllm` if you started
vLLM yourself (outside `rl-traces serve`) and still want build provenance
stamped into the report.

Use `--synth-max-osl <n>` to cap generated length during iteration — useful
for a fast sanity run before committing to a full-length trace.

### e. Compare configs

Once you have two or more `report.json` files:

```bash
rl-traces compare baseline=results-baseline/report.json mine=results-mine/report.json
```

Prints each config's tail bubble, goodput proxy, and makespan, sorted by
tail bubble (smallest first).

## Next

- [`docs/editable-vllm.md`](editable-vllm.md) — A/B two vLLM builds and
  attribute the difference via `report.json` provenance.
- [`docs/methodology.md`](methodology.md) — how the trace distribution and
  validation gate work.
- [`docs/metrics.md`](metrics.md) — what every field in `report.json` means.
