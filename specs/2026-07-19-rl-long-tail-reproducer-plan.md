# RL Long-Tail Serving Reproducer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trace-driven, static-batch serving benchmark that reproduces the RL rollout long tail against a real vLLM server and reports the long-tail / goodput metric set for A/B comparison of serving configs.

**Architecture:** Pure-Python calibration + trace-synthesis modules (TDD'd on a laptop, no GPU) produce a Mooncake JSONL trace; a shell layer replays it with **aiperf** against a served model in static-batch mode; an analyzer ingests aiperf's per-request export + server metrics and computes makespan / tail-bubble / goodput / cache-hit, with a dual validation gate (token-domain vs published percentiles, time-domain vs the real `nemorl_trace` rollout-time shape).

**Tech Stack:** Python 3.10+ (stdlib only — no numpy/scipy dependency), pytest, aiperf (Mooncake trace replay), vLLM serve (Super BF16, HSG GB200), bash.

## Global Constraints

- All scripts live under `cheat-sheet/vllm-rl-long-tail-goodput/reproducer/` — never `/tmp`.
- Pure-logic modules use **stdlib only** (no numpy/scipy) so they TDD on the Mac login env; math done by hand (quantile interpolation, linear least-squares).
- Replay tool = **aiperf** Mooncake trace: `--custom-dataset-type mooncake_trace --input-file <trace.jsonl>`. No fork; **stock latest vLLM**.
- Exact OSL enforced via `ignore_eos` + `min_tokens`; **CUDA graphs always on — never `--enforce-eager`**.
- Prefix-cache block size = **512** tokens (aiperf `--isl-block-size` default).
- Static batch: `--concurrency B`, session-count `B`; default `B=512` (March: 32 prompts × 16 gen), parameterized.
- Target: **Super BF16, single node, HSG** (GB200). aiperf mock-server for GPU-free smoke.
- **Execution locality:** ALL aiperf work runs on **HSG** — serve, static-batch trace-replay, the mock-server smoke, and analysis of real aiperf output. **Nothing aiperf-related runs locally.** Only the pure-Python modules' unit tests (`pytest`) and the one-time File-1 turn-count extraction run on the laptop; the committed scripts execute on HSG for real traces + analysis. The aiperf mock smoke runs on an HSG node (no GPU required), not on the Mac.
- Published OSL calibration anchors (quantile, tokens): `(0.50,654) (0.80,10000) (0.90,22000†) (0.95,33212) (0.99,57067) (1.0,65489)`; mean≈5355. †p90 is an estimate, not measured.
- Real time-domain reference (`nemorl_trace` `timing/rollout/total`): p50≈84s, p90≈591s, p99≈909s, max≈1669s (shape/ratio match only — absolute times are config-dependent).
- Default OSL interpretation = **per-turn** (`--osl-level per_turn`): each `model_call` turn draws OSL from the anchors (corroborated by File-1 per-turn duration tail p50=3.6s→max=93.5s). `per_rollout` mode available; final interpretation confirmed on first real run (design §10 Q2).

---

### Task 1: OSL quantile sampler (calibration core)

**Files:**
- Create: `reproducer/scripts/distributions.py`
- Test: `reproducer/tests/test_distributions.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `OSL_ANCHORS: list[tuple[float,int]]`; `class QuantileSampler(anchors)` with `.sample(u: float) -> int` (u∈[0,1]) and `.sample_n(n: int, rng: random.Random) -> list[int]`; `osl_sampler() -> QuantileSampler`.

- [ ] **Step 1: Write the failing test**

```python
# reproducer/tests/test_distributions.py
import random
from scripts.distributions import osl_sampler, OSL_ANCHORS

def _pct(v, p):
    v = sorted(v); return v[min(len(v)-1, round((p/100)*(len(v)-1)))]

def test_realized_percentiles_match_anchors():
    s = osl_sampler()
    vals = s.sample_n(200_000, random.Random(0))
    # Each anchor percentile must land within 8% of its target value.
    for q, target in OSL_ANCHORS:
        if q in (0.0, 1.0):
            continue
        got = _pct(vals, q * 100)
        assert abs(got - target) <= 0.08 * target, (q, got, target)

def test_bounds_and_80pct_under_10k():
    s = osl_sampler()
    vals = s.sample_n(100_000, random.Random(1))
    assert min(vals) >= 1 and max(vals) <= 65_489
    frac = sum(1 for v in vals if v < 10_000) / len(vals)
    assert 0.78 <= frac <= 0.82   # published: 80% < 10k
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_distributions.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.distributions`.

- [ ] **Step 3: Write minimal implementation**

```python
# reproducer/scripts/distributions.py
"""Inverse-CDF OSL sampler calibrated to published RL rollout percentiles.

We do NOT assume a parametric family (the real distribution is bimodal). Instead
we build the quantile function from the measured anchor points and interpolate
between them in log-token space, so realized percentiles match by construction.
"""
import math
import random

# (cumulative_prob, generated_tokens). p90 is estimated (design §10 / task header).
OSL_ANCHORS = [
    (0.00, 1),
    (0.50, 654),
    (0.80, 10_000),
    (0.90, 22_000),
    (0.95, 33_212),
    (0.99, 57_067),
    (1.00, 65_489),
]


class QuantileSampler:
    def __init__(self, anchors):
        self.anchors = sorted(anchors)

    def sample(self, u):
        u = min(max(u, 0.0), 1.0)
        a = self.anchors
        for i in range(1, len(a)):
            q0, v0 = a[i - 1]
            q1, v1 = a[i]
            if u <= q1:
                if q1 == q0:
                    return int(round(v1))
                t = (u - q0) / (q1 - q0)
                # log-linear interpolation between anchor token values
                logv = math.log(v0) + t * (math.log(v1) - math.log(v0))
                return max(1, int(round(math.exp(logv))))
        return int(round(a[-1][1]))

    def sample_n(self, n, rng):
        return [self.sample(rng.random()) for _ in range(n)]


def osl_sampler():
    return QuantileSampler(OSL_ANCHORS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_distributions.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add reproducer/scripts/distributions.py reproducer/tests/test_distributions.py
git commit -m "feat(reproducer): OSL quantile sampler calibrated to published percentiles"
```

---

### Task 2: Turn-structure reference + sampler

**Files:**
- Create: `reproducer/scripts/turn_structure.py`
- Create: `reproducer/data/turn_counts.json` (extracted from File-1)
- Test: `reproducer/tests/test_turn_structure.py`

**Interfaces:**
- Consumes: `reproducer/data/turn_counts.json` (a JSON list of ints — turns-per-rollout observed in File-1).
- Produces: `load_turn_counts(path: str) -> list[int]`; `sample_turn_count(rng: random.Random, counts: list[int]) -> int`; `split_osl(total: int, turns: int, rng: random.Random) -> list[int]` (per_rollout mode helper — Dirichlet-free proportional split, each turn ≥1).

- [ ] **Step 1: Extract the empirical turn-count reference (one-time data step)**

Run (produces the committed data file from the real trace via the existing probe helper):

```bash
cd reproducer
PYTHONPATH=. python3 - <<'PY'
import json, scripts.probe_traces as p   # probe_traces already parses File-1
data = open("/Users/tbarnatan/Downloads/all_rollouts_timeline_first_rollout_zero.html",
            encoding="utf-8", errors="replace").read()
i1 = data.find("Plotly.newPlot"); i2 = data.find("Plotly.newPlot", i1+1)
traces = json.loads(p._slice_bracket(data, i2, "[", "]"))
mc = next(t for t in traces if t.get("name") == "model_call")
from collections import defaultdict
per = defaultdict(int)
for txt in mc["text"]:
    rid = turn = None
    for kv in str(txt).split("<br>"):
        if kv.startswith("Rollout Id:"): rid = kv.split(":",1)[1]
        if kv.startswith("Turn:"): turn = int(kv.split(":",1)[1])
    if rid is not None and turn is not None:
        per[rid] = max(per[rid], turn)   # max turn idx = turn count for that rollout
counts = sorted(per.values())
json.dump(counts, open("data/turn_counts.json","w"))
print("rollouts:", len(counts), "min/med/max:", counts[0], counts[len(counts)//2], counts[-1])
PY
```
Expected: prints `rollouts: 1024 min/med/max: 15 30 30` (matches design §4). Commit `data/turn_counts.json`.

Note: `probe_traces.py` must expose `_slice_bracket` at module scope (it already defines it as a top-level function — importable).

- [ ] **Step 2: Write the failing test**

```python
# reproducer/tests/test_turn_structure.py
import random
from scripts.turn_structure import load_turn_counts, sample_turn_count, split_osl

def test_load_counts_nonempty():
    counts = load_turn_counts("data/turn_counts.json")
    assert len(counts) == 1024
    assert min(counts) >= 1 and max(counts) <= 30

def test_sample_turn_count_in_range():
    counts = load_turn_counts("data/turn_counts.json")
    rng = random.Random(0)
    for _ in range(1000):
        t = sample_turn_count(rng, counts)
        assert t in counts

def test_split_osl_sums_and_floor():
    rng = random.Random(0)
    parts = split_osl(65489, 30, rng)
    assert len(parts) == 30
    assert all(p >= 1 for p in parts)
    assert sum(parts) == 65489
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_turn_structure.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.turn_structure`.

- [ ] **Step 4: Write minimal implementation**

```python
# reproducer/scripts/turn_structure.py
"""Turn-count sampling (empirical, from File-1) + per-rollout OSL splitting."""
import json


def load_turn_counts(path):
    return json.load(open(path))


def sample_turn_count(rng, counts):
    return counts[rng.randrange(len(counts))]


def split_osl(total, turns, rng):
    """Split `total` output tokens across `turns` turns, each >=1, summing exactly."""
    if turns <= 1:
        return [max(1, total)]
    total = max(total, turns)   # each turn needs >=1 output token; ensure feasibility
    # random positive weights -> proportional integer split with remainder fix-up
    weights = [rng.random() + 1e-6 for _ in range(turns)]
    s = sum(weights)
    parts = [max(1, int(total * w / s)) for w in weights]
    diff = total - sum(parts)
    i = 0
    while diff != 0:                       # distribute rounding remainder
        j = i % turns
        if diff > 0:
            parts[j] += 1; diff -= 1
        elif parts[j] > 1:
            parts[j] -= 1; diff += 1
        i += 1
    return parts
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_turn_structure.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add reproducer/scripts/turn_structure.py reproducer/data/turn_counts.json reproducer/tests/test_turn_structure.py
git commit -m "feat(reproducer): empirical turn-count reference + sampler + OSL split"
```

---

### Task 3: Prompt/ISL growth + cumulative prefix hash_ids

**Files:**
- Create: `reproducer/scripts/prompt_model.py`
- Test: `reproducer/tests/test_prompt_model.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `per_turn_isl(osls, system_tokens, user_turn_tokens) -> list[int]`; `hash_ids_for(isls, block_size, rollout_base, shared_blocks) -> list[list[int]]`.

Model: turn *t*'s prompt = system + Σ_{i<t}(user_i + assistant_osl_i) + user_t. So ISL grows monotonically. hash_ids block-align the ISL; the first `shared_blocks` blocks are globally shared (system prompt cached across the whole batch), the rest are per-rollout (offset by `rollout_base`), and each turn's block list is a **superset** of the previous turn's (cumulative prefix cache).

- [ ] **Step 1: Write the failing test**

```python
# reproducer/tests/test_prompt_model.py
from scripts.prompt_model import per_turn_isl, hash_ids_for

def test_isl_monotonic_growth():
    osls = [654, 654, 5000]
    isls = per_turn_isl(osls, system_tokens=300, user_turn_tokens=200)
    assert len(isls) == 3
    assert isls[0] == 300 + 200                      # system + first user msg
    assert isls[1] > isls[0] and isls[2] > isls[1]   # accumulates prior turns

def test_hash_ids_cumulative_and_shared():
    isls = [1024, 2048, 4096]
    hids = hash_ids_for(isls, block_size=512, rollout_base=1000, shared_blocks=1)
    # turn N block set is a superset of turn N-1 (growing prefix reuse)
    assert set(hids[0]).issubset(set(hids[1]))
    assert set(hids[1]).issubset(set(hids[2]))
    # first block is the globally-shared system block (id 0), rest are per-rollout
    assert hids[0][0] == 0
    assert all(b == 0 or b >= 1000 for b in hids[2])
    # block counts match ceil(ISL/512)
    assert len(hids[0]) == 2 and len(hids[1]) == 4 and len(hids[2]) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_prompt_model.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.prompt_model`.

- [ ] **Step 3: Write minimal implementation**

```python
# reproducer/scripts/prompt_model.py
"""Per-turn ISL growth (accumulated conversation) + cumulative prefix hash_ids."""
import math


def per_turn_isl(osls, system_tokens, user_turn_tokens):
    isls = []
    ctx = system_tokens
    for i in range(len(osls)):
        ctx += user_turn_tokens          # this turn's incoming user/tool message
        isls.append(ctx)
        ctx += osls[i]                    # assistant response folds into next prompt
    return isls


def hash_ids_for(isls, block_size, rollout_base, shared_blocks):
    """Block-align each ISL; first `shared_blocks` are global (system), rest per-rollout.
    Cumulative: growing ISL => later turns reuse earlier blocks + append new ones."""
    out = []
    for isl in isls:
        nblocks = math.ceil(isl / block_size)
        blocks = []
        for b in range(nblocks):
            if b < shared_blocks:
                blocks.append(b)                       # global shared prefix (e.g. system)
            else:
                blocks.append(rollout_base + b)        # per-rollout unique block
        out.append(blocks)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_prompt_model.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add reproducer/scripts/prompt_model.py reproducer/tests/test_prompt_model.py
git commit -m "feat(reproducer): ISL growth + cumulative prefix hash_ids model"
```

---

### Task 4: Mooncake trace generator (CLI)

**Files:**
- Create: `reproducer/scripts/gen_trace.py`
- Test: `reproducer/tests/test_gen_trace.py`

**Interfaces:**
- Consumes: `distributions.osl_sampler`, `turn_structure.{load_turn_counts,sample_turn_count,split_osl}`, `prompt_model.{per_turn_isl,hash_ids_for}`.
- Produces: `build_trace(num_rollouts, seed, block_size, osl_level, system_tokens, user_turn_tokens, shared_blocks, turn_counts) -> tuple[list[dict], dict]` (records + stats); a `main()` CLI writing `<out>` JSONL + `<out>.stats.json`.

Each record: `{"session_id": int, "turn_idx": int, "timestamp": 0, "input_length": int, "output_length": int, "hash_ids": list[int]}`. `timestamp=0` for all (static batch); intra-session ordering by `turn_idx` (aiperf sequences dependent turns — confirmed in Task 7 smoke). Stats sidecar records realized OSL percentiles at the configured `osl_level`.

- [ ] **Step 1: Write the failing test**

```python
# reproducer/tests/test_gen_trace.py
from scripts.gen_trace import build_trace
from scripts.turn_structure import load_turn_counts

def test_trace_schema_and_sessions():
    counts = load_turn_counts("data/turn_counts.json")
    recs, stats = build_trace(num_rollouts=50, seed=0, block_size=512,
                              osl_level="per_turn", system_tokens=300,
                              user_turn_tokens=200, shared_blocks=1, turn_counts=counts)
    sessions = {r["session_id"] for r in recs}
    assert len(sessions) == 50
    for r in recs:
        assert set(r) == {"session_id","turn_idx","timestamp","input_length","output_length","hash_ids"}
        assert r["output_length"] >= 1 and r["input_length"] >= 1
        assert r["hash_ids"] == sorted(r["hash_ids"])
    # within a session, ISL grows and hash_ids are cumulative
    s0 = sorted([r for r in recs if r["session_id"] == 0], key=lambda r: r["turn_idx"])
    for a, b in zip(s0, s0[1:]):
        assert b["input_length"] >= a["input_length"]
        assert set(a["hash_ids"]).issubset(set(b["hash_ids"]))

def test_per_turn_osl_matches_anchors():
    counts = load_turn_counts("data/turn_counts.json")
    recs, stats = build_trace(num_rollouts=3000, seed=1, block_size=512,
                              osl_level="per_turn", system_tokens=300,
                              user_turn_tokens=200, shared_blocks=1, turn_counts=counts)
    # realized per-turn OSL median within 12% of published 654
    assert abs(stats["osl_p50"] - 654) <= 0.12 * 654
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_gen_trace.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.gen_trace`.

- [ ] **Step 3: Write minimal implementation**

```python
# reproducer/scripts/gen_trace.py
"""Synthesize a Mooncake multi-turn trace reproducing the RL rollout long tail."""
import argparse
import json
import random

from scripts.distributions import osl_sampler
from scripts.turn_structure import load_turn_counts, sample_turn_count, split_osl
from scripts.prompt_model import per_turn_isl, hash_ids_for


def _pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, round((p / 100) * (len(v) - 1)))]


def build_trace(num_rollouts, seed, block_size, osl_level, system_tokens,
                user_turn_tokens, shared_blocks, turn_counts):
    rng = random.Random(seed)
    sampler = osl_sampler()
    records, all_osl = [], []
    # reserve a per-rollout block namespace wide enough for the largest prompt
    base_stride = 1_000_000
    for sid in range(num_rollouts):
        turns = sample_turn_count(rng, turn_counts)
        if osl_level == "per_turn":
            osls = [sampler.sample(rng.random()) for _ in range(turns)]
        else:  # per_rollout: draw a rollout total, split across turns
            osls = split_osl(sampler.sample(rng.random()), turns, rng)
        isls = per_turn_isl(osls, system_tokens, user_turn_tokens)
        hids = hash_ids_for(isls, block_size, rollout_base=(sid + 1) * base_stride,
                            shared_blocks=shared_blocks)
        for t in range(turns):
            records.append({
                "session_id": sid, "turn_idx": t, "timestamp": 0,
                "input_length": isls[t], "output_length": osls[t],
                "hash_ids": hids[t],
            })
            if osl_level == "per_turn":
                all_osl.append(osls[t])
        if osl_level == "per_rollout":
            all_osl.append(sum(osls))   # one value per rollout, not per turn
    stats = {
        "num_rollouts": num_rollouts, "num_records": len(records),
        "osl_level": osl_level,
        "osl_p50": _pct(all_osl, 50), "osl_p90": _pct(all_osl, 90),
        "osl_p95": _pct(all_osl, 95), "osl_p99": _pct(all_osl, 99),
        "osl_max": max(all_osl),
    }
    return records, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-rollouts", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--osl-level", choices=["per_turn", "per_rollout"], default="per_turn")
    ap.add_argument("--system-tokens", type=int, default=300)
    ap.add_argument("--user-turn-tokens", type=int, default=200)
    ap.add_argument("--shared-blocks", type=int, default=1)
    ap.add_argument("--turn-counts", default="data/turn_counts.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    counts = load_turn_counts(a.turn_counts)
    recs, stats = build_trace(a.num_rollouts, a.seed, a.block_size, a.osl_level,
                              a.system_tokens, a.user_turn_tokens, a.shared_blocks, counts)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump(stats, open(a.out + ".stats.json", "w"), indent=2)
    print(f"wrote {len(recs)} records / {a.num_rollouts} rollouts -> {a.out}")
    print("stats:", json.dumps(stats))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_gen_trace.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add reproducer/scripts/gen_trace.py reproducer/tests/test_gen_trace.py
git commit -m "feat(reproducer): Mooncake multi-turn trace generator"
```

---

### Task 5: Long-tail metrics (pure functions)

**Files:**
- Create: `reproducer/scripts/metrics.py`
- Test: `reproducer/tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing (operates on normalized per-request dicts with `session_id`, `start`, `end`).
- Produces: `percentiles(vals, ps) -> dict`; `sessionize(records) -> dict[int,list]`; `session_completions(records, batch_start=0.0) -> list[float]`; `makespan(records) -> float`; `tail_bubble(completions) -> float`; `goodput_proxy(completions) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# reproducer/tests/test_metrics.py
from scripts.metrics import (percentiles, sessionize, session_completions,
                             makespan, tail_bubble, goodput_proxy)

REC = [  # two sessions: s0 finishes at 10s (2 turns), s1 is the straggler at 100s
    {"session_id": 0, "start": 0.0, "end": 4.0},
    {"session_id": 0, "start": 4.0, "end": 10.0},
    {"session_id": 1, "start": 0.0, "end": 100.0},
]

def test_session_completions_and_makespan():
    comps = session_completions(REC)
    assert sorted(comps) == [10.0, 100.0]
    assert makespan(REC) == 100.0

def test_tail_bubble_and_goodput():
    comps = [10.0, 100.0]
    # p90 of a 2-point set = the larger; bubble = max - p90
    assert tail_bubble(comps) == 0.0
    # goodput proxy = mean/max = 55/100
    assert abs(goodput_proxy(comps) - 0.55) < 1e-9

def test_percentiles():
    p = percentiles([1,2,3,4,5,6,7,8,9,10], [50,90])
    assert p[50] == 5 and p[90] == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.metrics`.

- [ ] **Step 3: Write minimal implementation**

```python
# reproducer/scripts/metrics.py
"""Static-batch long-tail metrics over normalized per-request records."""
from collections import defaultdict


def percentiles(vals, ps):
    v = sorted(vals)
    return {p: v[min(len(v) - 1, round((p / 100) * (len(v) - 1)))] for p in ps}


def sessionize(records):
    by = defaultdict(list)
    for r in records:
        by[r["session_id"]].append(r)
    return dict(by)


def session_completions(records, batch_start=0.0):
    """Per-session completion time = last turn end - batch start."""
    return [max(r["end"] for r in rs) - batch_start
            for rs in sessionize(records).values()]


def makespan(records):
    return max(r["end"] for r in records)


def tail_bubble(completions):
    """Idle bubble = makespan - p90(completion): time spent waiting on the tail."""
    p90 = percentiles(completions, [90])[90]
    return max(completions) - p90


def goodput_proxy(completions):
    """Fraction of batch wall time that is useful (mean completion / makespan)."""
    return (sum(completions) / len(completions)) / max(completions)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_metrics.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add reproducer/scripts/metrics.py reproducer/tests/test_metrics.py
git commit -m "feat(reproducer): static-batch long-tail metrics"
```

---

### Task 6: Analyzer — ingest aiperf export, compute report, dual validation

**Files:**
- Create: `reproducer/scripts/analyze.py`
- Create: `reproducer/tests/fixtures/profile_export_sample.jsonl`
- Test: `reproducer/tests/test_analyze.py`

**Interfaces:**
- Consumes: `metrics.*`, `distributions.OSL_ANCHORS`.
- Produces: `load_profile_export(path) -> list[dict]` (normalizes aiperf fields → `{session_id,start,end,isl,osl,ttft}`); `compute_report(records) -> dict`; `validate_token_domain(records, tol=0.15) -> dict`; `validate_time_domain(completions, ref=(84,909,1669), tol=0.35) -> dict`; `main()` writing a table + `report.html`.

Field normalization: aiperf `profile_export.jsonl` per-request keys vary by version (e.g. `request_latency`/`time_to_first_token`/`input_token_count`/`output_token_count`/`session_id`/`turn_index`/`start_ns`). `load_profile_export` maps known aliases and derives `start`/`end` from timestamp+latency. **First real run:** confirm actual key names and update the alias map (validation checkpoint in Task 7).

- [ ] **Step 1: Create the fixture** (a tiny synthetic aiperf-style export — 2 sessions, one straggler)

```jsonl
{"session_id": 0, "turn_index": 0, "start_ns": 0,          "request_latency_ns": 4000000000,  "input_token_count": 500,  "output_token_count": 654,  "time_to_first_token_ns": 200000000}
{"session_id": 0, "turn_index": 1, "start_ns": 4000000000, "request_latency_ns": 6000000000,  "input_token_count": 1200, "output_token_count": 654,  "time_to_first_token_ns": 250000000}
{"session_id": 1, "turn_index": 0, "start_ns": 0,          "request_latency_ns": 100000000000,"input_token_count": 500,  "output_token_count": 57067,"time_to_first_token_ns": 200000000}
```

- [ ] **Step 2: Write the failing test**

```python
# reproducer/tests/test_analyze.py
from scripts.analyze import (load_profile_export, compute_report,
                             validate_token_domain, validate_time_domain)

FIX = "tests/fixtures/profile_export_sample.jsonl"

def test_load_and_report():
    recs = load_profile_export(FIX)
    assert len(recs) == 3
    assert {r["session_id"] for r in recs} == {0, 1}
    rep = compute_report(recs)
    assert rep["makespan_s"] == 100.0            # straggler dominates
    assert rep["completion_p50_s"] in (10.0, 100.0)
    assert rep["num_sessions"] == 2

def test_validate_time_domain_passes_for_matching_shape():
    # 100-point distribution whose p50/p99/max match the real trace shape (84/909/1669).
    # (A 3-element list can't: nearest-rank p50 lands on the middle element.)
    comps = [84.0] * 51 + [909.0] * 48 + [1669.0]
    assert validate_time_domain(comps)["passed"] is True

def test_validate_time_domain_fails_for_flat_distribution():
    comps = [100.0] * 100          # no tail -> ratios ~1.0, far from ref -> fails
    assert validate_time_domain(comps)["passed"] is False
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_analyze.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.analyze`.

- [ ] **Step 4: Write minimal implementation**

```python
# reproducer/scripts/analyze.py
"""Ingest aiperf per-request export, compute long-tail report + dual validation."""
import argparse
import json

from scripts.metrics import (percentiles, session_completions, makespan,
                             tail_bubble, goodput_proxy)

_ALIASES = {
    "session_id": ["session_id", "conversation_id", "session"],
    "turn": ["turn_index", "turn_idx", "turn"],
    "start_ns": ["start_ns", "timestamp_ns", "start"],
    "latency_ns": ["request_latency_ns", "request_latency", "latency_ns"],
    "isl": ["input_token_count", "num_input_tokens", "isl", "prompt_tokens"],
    "osl": ["output_token_count", "num_output_tokens", "osl", "completion_tokens"],
    "ttft_ns": ["time_to_first_token_ns", "ttft_ns", "time_to_first_token"],
}


def _pick(d, key):
    for k in _ALIASES[key]:
        if k in d:
            return d[k]
    return None


def load_profile_export(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        start = (_pick(d, "start_ns") or 0) / 1e9
        lat = (_pick(d, "latency_ns") or 0) / 1e9
        out.append({
            "session_id": _pick(d, "session_id"),
            "turn": _pick(d, "turn"),
            "start": start, "end": start + lat,
            "isl": _pick(d, "isl"), "osl": _pick(d, "osl"),
            "ttft": (_pick(d, "ttft_ns") or 0) / 1e9,
        })
    return out


def compute_report(records):
    comps = session_completions(records)
    cp = percentiles(comps, [50, 90, 99])
    return {
        "num_sessions": len(comps),
        "num_requests": len(records),
        "makespan_s": makespan(records),
        "completion_p50_s": cp[50], "completion_p90_s": cp[90], "completion_p99_s": cp[99],
        "tail_bubble_s": tail_bubble(comps),
        "goodput_proxy": goodput_proxy(comps),
    }


def validate_token_domain(records, tol=0.15):
    osl = [r["osl"] for r in records if r["osl"]]
    got = percentiles(osl, [50, 95, 99])
    targets = {50: 654, 95: 33212, 99: 57067}
    checks = {p: abs(got[p] - t) <= tol * t for p, t in targets.items()}
    return {"passed": all(checks.values()), "realized": got, "checks": checks}


def validate_time_domain(completions, ref=(84.0, 909.0, 1669.0), tol=0.35):
    """Compare SHAPE (p99/p50 and max/p50 ratios), not absolute times."""
    p = percentiles(completions, [50, 99])
    got_r1 = p[99] / p[50]
    got_r2 = max(completions) / p[50]
    ref_r1 = ref[1] / ref[0]
    ref_r2 = ref[2] / ref[0]
    ok = (abs(got_r1 - ref_r1) <= tol * ref_r1) and (abs(got_r2 - ref_r2) <= tol * ref_r2)
    return {"passed": ok, "ratios": {"p99/p50": got_r1, "max/p50": got_r2},
            "ref_ratios": {"p99/p50": ref_r1, "max/p50": ref_r2}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out-html", default="report.html")
    ap.add_argument("--out-json", default=None)   # report.json consumed by compare.py
    a = ap.parse_args()
    recs = load_profile_export(a.export)
    rep = compute_report(recs)
    rep["validate_token"] = validate_token_domain(recs)
    rep["validate_time"] = validate_time_domain(session_completions(recs))
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rep.items())
    open(a.out_html, "w").write(f"<html><body><h1>RL long-tail report</h1>"
                                f"<table border=1>{rows}</table></body></html>")
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(rep, f, indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_analyze.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add reproducer/scripts/analyze.py reproducer/tests/test_analyze.py reproducer/tests/fixtures/profile_export_sample.jsonl
git commit -m "feat(reproducer): analyzer with report + dual validation gate"
```

---

### Task 7: Serve + aiperf runner + configs (cluster) — mock smoke then real Super

**Files:**
- Create: `reproducer/serve/serve_super_bf16_hsg.sh`
- Create: `reproducer/run/run_batch.sh`
- Create: `reproducer/run/smoke_mock.sh`
- Create: `reproducer/configs/baseline.env`
- Create: `reproducer/configs/mtp.env`
- Create: `reproducer/configs/chunked_prefill.env`

**Interfaces:**
- Consumes: `gen_trace.py` output trace; produces aiperf `profile_export.jsonl` consumed by `analyze.py`.

This task has no unit test (cluster/GPU). Its gates are two smoke runs with expected output. Follow the HSG facts in the design (aarch64 login, `uv` at `~/.local/bin`, `batch` partition needs `--gpus-per-node=4`, shared nemo_rl sqsh not needed here — use stock `vllm/vllm-openai` container).

- [ ] **Step 1: Write `configs/baseline.env`** (serving knobs; one file per A/B cell)

```bash
# reproducer/configs/baseline.env
MODEL="<SUPER_BF16_CKPT_PATH>"          # fill from design §10 Q4 on first real run
TP=4
EXTRA_SERVE_ARGS=(--enable-prefix-caching --no-enable-chunked-prefill)  # bash array
# CUDA graphs ON (never --enforce-eager). ignore_eos handled by run_batch client args.
```

```bash
# reproducer/configs/mtp.env  — same as baseline + MTP speculative decoding
MODEL="<SUPER_BF16_CKPT_PATH>"
TP=4
EXTRA_SERVE_ARGS=(--enable-prefix-caching --speculative-config '{"method":"...","num_speculative_tokens":5}')  # bash array preserves the JSON token
```

```bash
# reproducer/configs/chunked_prefill.env
MODEL="<SUPER_BF16_CKPT_PATH>"
TP=4
EXTRA_SERVE_ARGS=(--enable-prefix-caching --enable-chunked-prefill)  # bash array
```

- [ ] **Step 2: Write `run/smoke_mock.sh`** (GPU-free harness validation via aiperf mock server)

```bash
#!/usr/bin/env bash
# Validate the full trace->aiperf->analyze pipeline with NO GPU using aiperf's mock server.
# RUNS ON HSG (login/compute node), NOT locally — aiperf is not installed on the laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=. python3 scripts/gen_trace.py --num-rollouts 8 --osl-level per_turn --out /tmp/smoke_trace.jsonl
# Start aiperf mock server (see aiperf tests/aiperf_mock_server/README.md), then:
aiperf profile --model mock --endpoint-type chat --endpoint /v1/chat/completions \
  --url localhost:8000 --custom-dataset-type mooncake_trace --input-file /tmp/smoke_trace.jsonl \
  --concurrency 8 --streaming --export-level records --output-dir /tmp/smoke_out
PYTHONPATH=. python3 scripts/analyze.py --export /tmp/smoke_out/profile_export.jsonl --out-html /tmp/smoke_report.html
echo "SMOKE OK"
```

- [ ] **Step 3: Run the mock smoke; confirm field names**

Run: `bash reproducer/run/smoke_mock.sh`
Expected: prints `SMOKE OK` and a JSON report. **Checkpoint:** open `/tmp/smoke_out/profile_export.jsonl`, confirm the actual per-request key names (session id, turn, latency, tokens), and if they differ from the fixture, update `_ALIASES` in `analyze.py` + re-run Task 6 tests. Also confirm aiperf issues turn 2 of a session only after turn 1 returns (dependent-turn sequencing, design §10 Q3) — inspect request ordering/timestamps in the export.

- [ ] **Step 4: Write `serve/serve_super_bf16_hsg.sh`** (single-node Super BF16 serve)

```bash
#!/usr/bin/env bash
# Single-node Super BF16 vLLM serve on HSG (GB200). Sourced env picks the config.
set -euo pipefail
CFG="${1:?usage: serve_super_bf16_hsg.sh <configs/xxx.env>}"; source "$CFG"
vllm serve "$MODEL" --tensor-parallel-size "${TP}" --port 8000 \
  --gpu-memory-utilization 0.9 "${EXTRA_SERVE_ARGS[@]}"
# NOTE: CUDA graphs remain ON. Do not add --enforce-eager.
```

- [ ] **Step 5: Write `run/run_batch.sh`** (static-batch aiperf replay against the real server)

```bash
#!/usr/bin/env bash
# Replay a Mooncake trace as a STATIC BATCH: concurrency == #sessions, exact OSL.
set -euo pipefail
cd "$(dirname "$0")/.."
TRACE="${1:?usage: run_batch.sh <trace.jsonl> <B> <out_dir>}"; B="${2:?}"; OUT="${3:?}"
aiperf profile --model "${AIPERF_MODEL:-super}" --endpoint-type chat \
  --endpoint /v1/chat/completions --url localhost:8000 --streaming \
  --custom-dataset-type mooncake_trace --input-file "$TRACE" \
  --concurrency "$B" --export-level records --output-dir "$OUT" \
  --extra-inputs ignore_eos:true --extra-inputs min_tokens:1
PYTHONPATH=. python3 scripts/analyze.py --export "$OUT/profile_export.jsonl" \
  --out-html "$OUT/report.html" --out-json "$OUT/report.json"   # report.json for compare.py
```

- [ ] **Step 6: Real Super static-batch run (HSG, combined build+serve+bench in one alloc)**

Per memory (combine allocations): one sbatch that (a) starts the serve script, (b) waits for health, (c) generates a `B=512` trace, (d) runs `run_batch.sh`, (e) runs the analyzer. Submit under both accounts if applicable.
Run (example): `sbatch reproducer/run/hsg_static_batch.sbatch configs/baseline.env`
Expected: `report.html` with `makespan_s`, `tail_bubble_s`, `goodput_proxy`, and both validation gates. **Checkpoint:** `validate_token.passed == true`; inspect `validate_time` ratios vs the real trace.

- [ ] **Step 7: Commit**

```bash
git add reproducer/serve reproducer/run reproducer/configs
git commit -m "feat(reproducer): serve + static-batch aiperf runner + A/B configs + mock smoke"
```

---

### Task 8: A/B comparison across configs

**Files:**
- Create: `reproducer/scripts/compare.py`
- Test: `reproducer/tests/test_compare.py`

**Interfaces:**
- Consumes: multiple `report.json`-style dicts (from `analyze.compute_report`).
- Produces: `compare_reports(named_reports: dict[str,dict]) -> list[dict]` (one row per config, sorted by `tail_bubble_s` ascending — "which approach helps the tail most").

- [ ] **Step 1: Write the failing test**

```python
# reproducer/tests/test_compare.py
from scripts.compare import compare_reports

def test_ranks_by_tail_bubble():
    reports = {
        "baseline": {"tail_bubble_s": 900, "goodput_proxy": 0.33, "makespan_s": 1600},
        "mtp":      {"tail_bubble_s": 400, "goodput_proxy": 0.55, "makespan_s": 900},
    }
    rows = compare_reports(reports)
    assert [r["config"] for r in rows] == ["mtp", "baseline"]
    assert rows[0]["tail_bubble_s"] == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_compare.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.compare`.

- [ ] **Step 3: Write minimal implementation**

```python
# reproducer/scripts/compare.py
"""Rank A/B serving configs by long-tail metrics."""
import argparse
import json


def compare_reports(named_reports):
    rows = [dict(config=name, **rep) for name, rep in named_reports.items()]
    rows.sort(key=lambda r: r["tail_bubble_s"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", help="name=path/report.json pairs")
    a = ap.parse_args()
    named = {}
    for spec in a.reports:
        name, path = spec.split("=", 1)
        named[name] = json.load(open(path))
    for r in compare_reports(named):
        print(f"{r['config']:16s} bubble={r['tail_bubble_s']:.1f}s "
              f"goodput={r['goodput_proxy']:.2f} makespan={r['makespan_s']:.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_compare.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add reproducer/scripts/compare.py reproducer/tests/test_compare.py
git commit -m "feat(reproducer): A/B config comparison ranked by tail bubble"
```

---

### Task 9 (optional, non-blocking): Tau2-grounded token↔time estimator

Design §6.1. Do NOT block Phase-1 completion on this; sequence it once the core pipeline runs. Requires the Tau2 per-turn artifact (`usage.prompt_tokens`, `usage.completion_tokens`, `generation_time_seconds`) — locate the local candidate and/or pull fuller per-request results from HSG first.

**Files:**
- Create: `reproducer/scripts/tau2_estimator.py`
- Test: `reproducer/tests/test_tau2_estimator.py`

**Interfaces:**
- Produces: `fit_time_model(rows) -> dict` (`{"ttft_a","ttft_b","itl"}` from OLS on `dur = ttft_a + ttft_b*isl + itl*osl`); `invert_duration(dur, isl, model) -> int` (solve for OSL); `load_tau2(path) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
# reproducer/tests/test_tau2_estimator.py
from scripts.tau2_estimator import fit_time_model, invert_duration

def test_fit_and_invert_recovers_osl():
    # synthetic ground truth: dur = 0.2 + 0.0001*isl + 0.01*osl
    rows = [{"isl": isl, "osl": osl, "dur": 0.2 + 0.0001*isl + 0.01*osl}
            for isl in (500, 1000, 2000) for osl in (100, 500, 2000, 5000)]
    m = fit_time_model(rows)
    assert abs(m["itl"] - 0.01) < 1e-3
    got = invert_duration(0.2 + 0.0001*1000 + 0.01*654, isl=1000, model=m)
    assert abs(got - 654) <= 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_tau2_estimator.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts.tau2_estimator`.

- [ ] **Step 3: Write minimal implementation** (stdlib OLS via normal equations, 3 unknowns)

```python
# reproducer/scripts/tau2_estimator.py
"""Fit duration ~ ttft(isl) + itl*osl from Tau2 per-turn data; invert to estimate OSL."""
import argparse
import json


def _solve3(A, b):
    """Solve 3x3 linear system by Gaussian elimination (stdlib only)."""
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [x / piv for x in M[c]]
        for r in range(3):
            if r != c:
                f = M[r][c]
                M[r] = [M[r][k] - f * M[c][k] for k in range(4)]
    return [M[i][3] for i in range(3)]


def fit_time_model(rows):
    """OLS for dur = a + b*isl + c*osl. Features x=[1, isl, osl]."""
    ATA = [[0.0] * 3 for _ in range(3)]
    ATb = [0.0] * 3
    for r in rows:
        x = [1.0, float(r["isl"]), float(r["osl"])]
        for i in range(3):
            ATb[i] += x[i] * r["dur"]
            for j in range(3):
                ATA[i][j] += x[i] * x[j]
    a, b, c = _solve3(ATA, ATb)
    return {"ttft_a": a, "ttft_b": b, "itl": c}


def invert_duration(dur, isl, model):
    """Given a measured duration + ISL, back out OSL = (dur - a - b*isl)/itl."""
    osl = (dur - model["ttft_a"] - model["ttft_b"] * isl) / model["itl"]
    return max(1, int(round(osl)))


def load_tau2(path):
    """Load per-turn Tau2 records: usage.prompt_tokens/completion_tokens + generation_time_seconds."""
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        u = d.get("usage", {})
        if "prompt_tokens" in u and "completion_tokens" in u and "generation_time_seconds" in d:
            rows.append({"isl": u["prompt_tokens"], "osl": u["completion_tokens"],
                         "dur": d["generation_time_seconds"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau2", required=True)
    a = ap.parse_args()
    m = fit_time_model(load_tau2(a.tau2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd reproducer && PYTHONPATH=. pytest tests/test_tau2_estimator.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add reproducer/scripts/tau2_estimator.py reproducer/tests/test_tau2_estimator.py
git commit -m "feat(reproducer): Tau2 token<->time estimator (optional calibration)"
```

---

### Task 10: README + cheat-sheet index entry

**Files:**
- Create: `reproducer/README.md`
- Modify: `cheat-sheet/index.md` (add under a new "RL Long-Tail Serving Reproducer" section)

- [ ] **Step 1: Write `reproducer/README.md`** — quickstart: install aiperf, `gen_trace.py` → `smoke_mock.sh` → serve → `run_batch.sh` → `analyze.py` → `compare.py`; the metric definitions (makespan / tail bubble / goodput proxy); the dual validation gate; the three calibration inputs; Phase-2 pointer. Copy-pasteable commands with `<PLACEHOLDER>` paths.

- [ ] **Step 2: Add index entry** under a new section in `cheat-sheet/index.md` linking the design spec, this plan, and the README (one line each, matching the file's existing link style). Stage ONLY the index line you add (the file has unrelated in-flight edits — do not commit those).

- [ ] **Step 3: Run the full unit suite once more**

Run: `cd reproducer && PYTHONPATH=. pytest -v`
Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add reproducer/README.md
git commit -m "docs(reproducer): README + index entry"
```

---

## Self-Review

**Spec coverage:**
- §2 DoD (mimic tail, measure, A/B) → Tasks 4 (trace), 5–6 (metrics/report), 7 (real run), 8 (compare). ✅
- §3 static-batch, synthetic-first, swappable → Task 4 (`--osl-level`, plain Mooncake JSONL), Task 7 (`--concurrency B`). ✅
- §4 three data sources → Task 1 (percentiles), Task 2 (File-1 turn skeleton), Task 6 `validate_time_domain` (nemorl_trace shape), Task 9 (Tau2). ✅
- §5 components (gen_trace/run_batch/analyze/configs/serve) → Tasks 4/7/6/7/7. ✅
- §6 synthesis model → Tasks 1–4; §6.1 Tau2 estimator → Task 9. ✅
- §6 dual validation gate → Task 6 `validate_token_domain` + `validate_time_domain`. ✅
- §7 metrics (makespan, tail bubble, goodput, percentiles, cache-hit) → Task 5 + Task 6. Cache-hit rate read from server Prometheus at real-run time (Task 7 checkpoint) — noted, not a unit. ⚠ add to README.
- §8 Super BF16 single-node HSG, B=512, CUDA graphs on → Task 7 + Global Constraints. ✅
- §9 A/B knobs (MTP, chunked-prefill, routing) → Task 7 configs + Task 8. Routing (multi-engine) deferred — single-node first. ✅
- §11 non-goals (lag-1/refit/RL loop) → correctly absent (Phase 2). ✅

**Placeholder scan:** `<SUPER_BF16_CKPT_PATH>` and `<PLACEHOLDER>` are intentional runtime values (design §10 Q4, resolved on first cluster run), not plan placeholders — every code step has complete code. No TBD/TODO in logic.

**Type consistency:** normalized record keys (`session_id`,`start`,`end`,`isl`,`osl`) consistent across `metrics.py`, `analyze.py`; `compute_report` output keys (`tail_bubble_s`,`goodput_proxy`,`makespan_s`) consistent with `compare.py` input. `build_trace` record keys consistent with `metrics`/`analyze` expectations (session_id/turn_idx/input_length/output_length). ✅
