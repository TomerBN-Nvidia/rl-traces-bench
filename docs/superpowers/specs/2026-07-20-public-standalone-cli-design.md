# Design: `rl-traces-bench` — public, standalone, editable-vLLM-friendly

Status: approved (2026-07-20)

## 1. Goal

Turn the internal RL long-tail serving reproducer into a **public, standalone,
installable benchmark tool** with a clean CLI, so external users can synthesize
trace-driven static-batch workloads, run them against any OpenAI-compatible
endpoint, and A/B serving configs — including their own **editable vLLM builds**.

NVIDIA-only cluster glue and internal references are removed from this repo and
handed to a private follow-up repo (created separately, not in this PR).

## 2. Non-negotiables

- **Pure refactor — no logic changes.** The sampling math (`distributions`,
  `turn_structure`, `prompt_model`, `gen_trace`), the metrics (`metrics`,
  `analyze`, `compare`), and `tau2_estimator` keep their algorithms
  byte-for-byte. Same seed + default distribution → identical trace as today.
  Only packaging, CLI surface, config mechanism, and wording change.
- **Two-level scrub.** Remove NVIDIA-only *files* AND strip internal
  *words/paths* from every file that stays — `HSG`, cluster names, lustre/home
  paths, Slurm accounts, `design §N` pointers, "we were given", File-1 /
  OpenHands / nemorl provenance, internal model codenames. Applies to README,
  docstrings, comments, tests — everything.
- **Keep** the public HF "super" model id in examples (it is a published HF
  release, not internal).
- **No config hierarchy.** No `configs/*.env` sourcing chain, no invented
  serve-knob vocabulary, no `EXTRA_SERVE_ARGS` array. A flat, explicit `.env`.

## 3. Package & CLI

Installable Python package with one console entry point.

- `pip install rl-traces-bench` — core, light (stdlib + pytest for tests).
- `pip install rl-traces-bench[serve]` — adds the vLLM serve helper (heavy, GPU).

```
rl-traces gen-trace   --distribution <json> --num-rollouts 512 --out t.jsonl
rl-traces run         --url localhost:8000 --trace t.jsonl --concurrency 512 \
                      --tokenizer <hf-id> --out <dir>
rl-traces analyze     --export profile_export.jsonl --out-json report.json
rl-traces compare     baseline=a.json mtp=b.json
rl-traces serve       [--env .env]     # [serve] extra: exec `vllm serve $VLLM_SERVE_ARGS` + provenance
rl-traces doctor      [--env .env]     # preflight: vLLM editable? matches VLLM_SRC? aiperf? tokenizer?
```

Layout — `src/` package; each existing script becomes a module (imports change
from `import distributions` to package-relative — mechanical, not logical):

```
rl-traces-bench/
├── pyproject.toml                  # packaging; entry point `rl-traces`; extras [serve]
├── README.md                       # rewritten, generic, good docs
├── .env.example                    # flat, explicit
├── src/rl_traces_bench/
│   ├── cli.py                      # subcommand dispatch into existing main() bodies
│   ├── distributions.py           # LOGIC UNCHANGED
│   ├── turn_structure.py           # LOGIC UNCHANGED
│   ├── prompt_model.py             # LOGIC UNCHANGED
│   ├── gen_trace.py                # LOGIC UNCHANGED (input source generalized, default = same numbers)
│   ├── metrics.py                  # LOGIC UNCHANGED
│   ├── analyze.py                  # LOGIC UNCHANGED (+ provenance field passthrough)
│   ├── compare.py                  # LOGIC UNCHANGED (+ surfaces provenance)
│   ├── tau2_estimator.py           # LOGIC UNCHANGED
│   ├── probe_traces.py             # kept (generic parser) — provenance wording scrubbed
│   ├── serve.py                    # NEW: thin runner + provenance stamp ([serve] extra)
│   ├── doctor.py                   # NEW: preflight checks
│   └── provenance.py               # NEW: vllm.__version__ + VLLM_SRC git SHA/dirty
├── examples/
│   ├── distributions/example_longtail.json   # current calibration numbers, as an example
│   ├── env/baseline.env
│   ├── env/mtp.env
│   └── mock_server.py              # generic (from run/mock_chat_server.py)
├── docs/
│   ├── quickstart.md
│   ├── editable-vllm.md
│   ├── methodology.md              # generic replacement for internal design spec
│   └── metrics.md
├── .claude/skills/<name>/SKILL.md  # 4 skills (see §8)
├── .codex/skills/<name>/SKILL.md   # identical mirror of the 4 skills
├── traces/                         # sample .stats.json outputs
└── tests/                          # pytest, imports updated to package path
```

## 4. `run` vs `serve` — roles (documented in README + quickstart)

- **`serve`** = the system under test. Boots vLLM (your editable build), holds
  the GPU, listens on a port, long-lived. Run it **only when you must host the
  model yourself** (measuring an editable-vLLM change, or no endpoint exists).
  If you already have an endpoint, **skip `serve`** and point `run --url` at it.
- **`run`** = the measurement client. aiperf replays the trace against the
  endpoint and writes `profile_export.jsonl`. Needed **every** benchmark.

Loop for "did my vLLM change shrink the tail?": `gen-trace` once → `serve`
build X → `run` → edit vLLM → `serve` build Y → `run` → `compare`.

## 5. Distribution as a first-class input

Unify calibration into one swappable JSON — the "create traces given
distribution" UX:

```json
{ "osl_percentiles": {"p50":654,"p90":22000,"p95":33212,"p99":57067,"max":65489},
  "turn_counts": [15, 21, 25, "..."] }
```

Ship the **current values** as `examples/distributions/example_longtail.json`
and default `--distribution` to it. Because the default holds today's exact
numbers, default output is unchanged (satisfies "no logic change"). Provenance
described generically in docs ("an example long-tail OSL profile + a real-world
multi-turn skeleton"); no internal source names.

## 6. Serving config: explicit, flat, no hierarchy

Flat `.env`, literal vLLM args passed straight through:

```
VLLM_SRC=/path/to/your/vllm                  # editable checkout (provenance + doctor only)
VLLM_SERVE_ARGS=<hf-id> --tensor-parallel-size 4 --port 8000 --enable-prefix-caching --no-enable-chunked-prefill
TOKENIZER=<hf-id>
URL=localhost:8000
```

- `serve` runs literally `vllm serve $VLLM_SERVE_ARGS` — never parses/re-maps
  vLLM flags. Any flag that exists works, today and future, zero tool changes.
- **Editable install boundary: verify + serve, not auto-install.** `serve`
  assumes vLLM is already installed (editable builds live in containers/cluster
  wheels/CUDA — fragile to automate). `doctor` + a one-time `scripts/setup.sh`
  (simple laptop/GPU-box case) cover setup.
- Each run's `report.json` is stamped with `vllm.__version__` + `VLLM_SRC` git
  SHA/dirty so `compare` attributes tail changes to a specific build.
- **A/B cells = separate labeled `.env` files** (`examples/env/baseline.env`,
  `mtp.env`), each one explicit `VLLM_SERVE_ARGS` line. No sourcing chain.
- `run` gains optional `--vllm-src` so provenance works when you start vLLM
  yourself.

## 7. What moves out (private repo, next step — NOT this PR)

- **Deleted from public:** `serve/serve_super_bf16_hsg.sh`,
  `run/hsg_static_batch.sbatch`, `run/run_batch.sh` (its generic aiperf core is
  reimplemented as `rl-traces run`), `configs/*.env`, `specs/*`, and
  `docs/2026-07-20-task-log.md` (internal working docs, full of §-refs / cluster
  / account names).
- **Kept, genericized:** `run/mock_chat_server.py` → `examples/mock_server.py`
  (drives the `run` smoke, no GPU); `traces/*.stats.json` as sample outputs.
- Generic `docs/methodology.md` replaces the internal design spec.

## 8. Agent skills (committed to the repo)

Four skills, authored once and committed **identically** under both
`.claude/skills/<name>/SKILL.md` and `.codex/skills/<name>/SKILL.md` (identical
SKILL.md format; only the root dir differs):

- `run-longtail-bench` — full loop `gen-trace → serve`(or verify endpoint)`→
  run → analyze → compare`, incl. A/B-ing two vLLM builds.
- `setup-editable-vllm` — venv, editable-install vLLM, `doctor`, healthy endpoint.
- `interpret-longtail-report` — makespan / tail-bubble / goodput / dual
  validation gate, and judging whether a change helped.
- `author-distribution` — craft a custom `--distribution` JSON.

Public-safe; reference only the public CLI + HF example model.

## 9. Tests

Existing pytest modules kept (logic unchanged), imports updated to the package
path; same count expected green. `examples/mock_server.py` smoke stays runnable.

## 10. Pre-merge checklist (executed right before flipping public)

- [ ] History squash / fresh-init (removes internal names from old commits).
- [ ] Remove `docs/superpowers/` design/plan docs from the tree.
- [ ] `grep -riE 'hsg|lustre|coreai|llmservice|nemorl|design §|/home/'` returns clean.
- [ ] Confirm the private follow-up repo captures the deleted HSG/Slurm glue.

## 11. Out of scope

- Creating the private repo (separate follow-up).
- Any change to the sampling/metric algorithms.
- Automating the vLLM source build/compile.
- Phase-2 RL-loop orchestration (unchanged from original design).
