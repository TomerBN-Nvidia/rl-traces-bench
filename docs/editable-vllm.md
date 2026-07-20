# Using rl-traces-bench with an editable vLLM checkout

This tool is designed for the common "does my vLLM change actually shrink
the tail?" loop: serve a build, run the benchmark, edit vLLM, serve again,
run again, compare. Every report is stamped with enough build provenance to
tell the two runs apart after the fact.

## Verify, not install — the boundary

`rl-traces serve` **assumes vLLM is already installed** in your active
Python environment; it does not build, compile, or install it for you.
Editable vLLM builds typically live in containers, cluster wheel caches, or
custom CUDA toolchains — fragile things to automate generically. Instead:

- **You** do the one-time setup (venv + editable install, below).
- **`rl-traces doctor`** verifies the setup is correct before you spend time
  on a run.
- **`rl-traces serve`** trusts that verification and just runs
  `vllm serve $VLLM_SERVE_ARGS` verbatim.

## One-time setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install 'rl-traces-bench[serve]'      # or: pip install -e '.[serve]' inside a clone

export VLLM_SRC=/path/to/your/vllm        # your editable checkout
pip install -e "$VLLM_SRC"
```

Create a `.env` pointing at that checkout:

```
VLLM_SRC=/path/to/your/vllm
VLLM_SERVE_ARGS=nvidia/Llama-3_3-Nemotron-Super-49B-v1 --tensor-parallel-size 4 --port 8000 --enable-prefix-caching --no-enable-chunked-prefill
TOKENIZER=nvidia/Llama-3_3-Nemotron-Super-49B-v1
URL=localhost:8000
```

`VLLM_SERVE_ARGS` is passed to `vllm serve` **verbatim** — any flag your
vLLM build supports works, with zero translation layer and zero changes to
this tool as vLLM's CLI evolves.

Run `rl-traces doctor` before you serve anything:

```bash
rl-traces doctor --env .env
```

Since this `.env` sets both `VLLM_SERVE_ARGS` and `URL`, `doctor` checks
your vLLM setup plus (since `URL` is set) probes the endpoint:

```
[OK ] aiperf present
[OK ] tokenizer set
[FAIL] endpoint reachable  -> start your server or fix URL in .env
[OK ] vllm importable
[OK ] vllm_src exists
```

`endpoint reachable` failing here is expected — you haven't run `serve` yet.
Get every *other* line green first (a `FAIL` prints its own fix hint, e.g.
`pip install -e $VLLM_SRC`); that's the part worth debugging before a real
run. Once you `serve` and want to confirm the endpoint actually came up,
rerun `doctor` and `endpoint reachable` should flip to `OK`.

## The A/B loop

Generate the trace once — you want the same trace replayed against both
builds so the only variable is vLLM itself:

```bash
rl-traces gen-trace --num-rollouts 512 --seed 0 --out t.jsonl
```

**Build X** (your current vLLM checkout state):

```bash
rl-traces serve --env .env &
rl-traces run --url localhost:8000 --trace t.jsonl --concurrency 512 \
  --tokenizer nvidia/Llama-3_3-Nemotron-Super-49B-v1 --out results-x/ \
  --vllm-src "$VLLM_SRC"
kill %1   # stop the server before making changes
```

**Edit vLLM** — make your change (a new kernel, a scheduler tweak, a flag
default, whatever you're testing) and reinstall if needed for the change to
take effect in the editable install.

**Build Y** (vLLM after your change):

```bash
rl-traces serve --env .env &
rl-traces run --url localhost:8000 --trace t.jsonl --concurrency 512 \
  --tokenizer nvidia/Llama-3_3-Nemotron-Super-49B-v1 --out results-y/ \
  --vllm-src "$VLLM_SRC"
kill %1
```

**Compare:**

```bash
rl-traces compare build-x=results-x/report.json build-y=results-y/report.json
```

## How provenance attributes the change to a build

`rl-traces serve` writes `serve_provenance.json` next to your `.env`, and
`rl-traces run --vllm-src <path>` (or the provenance `serve` already
collected, if you started vLLM yourself with the same `VLLM_SRC`) stamps the
same information into `report.json`'s `provenance` field:

```json
{
  "vllm_version": "0.x.y",
  "vllm_src": "/path/to/your/vllm",
  "git_sha": "abcdef1234...",
  "dirty": false
}
```

- `vllm_version` — `vllm.__version__` from the environment that served the
  request.
- `git_sha` / `dirty` — the exact commit (and working-tree cleanliness) of
  `VLLM_SRC` at run time.

So when `results-x/report.json` and `results-y/report.json` show different
`tail_bubble_s`, the `provenance` block tells you exactly which commit (and
whether it was a clean checkout) produced each number — no need to trust
memory about which build was running when.

If `--vllm-src` is omitted and `rl-traces serve` wasn't used to start the
server, `provenance.git_sha` and `provenance.dirty` will be `null`; only
`vllm_version` (read from the environment `run` executes in) will be
populated.

## See also

- [`docs/quickstart.md`](quickstart.md) — the non-editable-vLLM basics.
- [`docs/metrics.md`](metrics.md) — what to look at in `report.json` when
  judging whether a change helped.
