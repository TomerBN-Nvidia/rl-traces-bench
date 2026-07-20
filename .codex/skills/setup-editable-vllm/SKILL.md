---
name: setup-editable-vllm
description: Use when setting up an editable vLLM checkout to serve for the rl-traces long-tail benchmark, or when a `rl-traces doctor` check is failing and you need a healthy endpoint before running the benchmark.
---

# Set up an editable vLLM checkout

Goal: get `rl-traces doctor --env .env` to report all checks OK, then serve.

## 1. Create a venv

```
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install vLLM in editable mode

```
pip install -e $VLLM_SRC
```

`$VLLM_SRC` is your local vLLM source checkout (a repo with `setup.py`/`pyproject.toml` at its root, on the branch/commit you want to benchmark).

## 3. Install the benchmark tooling

```
pip install aiperf rl-traces-bench
```

`aiperf` is the load-generation dependency `rl-traces run` uses under the hood; `rl-traces-bench` provides the `rl-traces` CLI itself.

## 4. Write a flat `.env` and iterate with doctor

Create `.env` with flat keys:

```
VLLM_SRC=<path to your editable vLLM checkout>
VLLM_SERVE_ARGS=--model <hf-model-id> --port 8000
TOKENIZER=<hf-model-id>
URL=localhost:8000
```

Then loop:

```
rl-traces doctor --env .env
```

until every check prints `[OK ]`. The checks are: `vllm` importable, `aiperf` on PATH, `TOKENIZER` set, and `VLLM_SRC` pointing at a real directory. Each failing line prints its own fix hint — follow it, rerun `doctor`, repeat.

Example model id you can use for a smoke test: `nvidia/Llama-3_3-Nemotron-Super-49B-v1`.

## 5. Serve

Once `doctor` is all green:

```
rl-traces serve --env .env
```

This runs `vllm serve $VLLM_SERVE_ARGS` verbatim and writes `serve_provenance.json` for later comparison.

## Verify-not-install boundary

`doctor` only **verifies** — it never installs anything or mutates your environment. If a check fails, the fix (the `pip install ...` or the `.env` edit) is on you to run; then rerun `doctor` to confirm. Don't skip straight to `serve` with failing checks — a missing tokenizer or stale `VLLM_SRC` will surface as a confusing failure deep inside `run` instead of a clear preflight message.
