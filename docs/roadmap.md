# Roadmap / Future work

Phase 1 (this repo) synthesizes a calibrated long-tail trace, replays it as a
static batch against any OpenAI-compatible endpoint, and reports the long-tail /
goodput metric set for A/B-ing serving configs. Natural next steps:

## Interactive HTML report — `analyze` (shipped) → `compare` (next)

`analyze --out-html` now emits a **self-contained interactive report** (inline
SVG + vanilla JS, no CDN deps) from the existing `report.json` + folded aiperf
summary: headline stat tiles, coherence-gate status chips, the per-rollout
**completion-time CDF**, a goodput-decomposition bar, a completion-vs-OSL
scatter, and token-domain validation — with hover tooltips, a table view, and
dark mode. See `rl_traces_bench/report_html.py`.

Still open, to extend it:

- a per-rollout **OSL histogram** and the ISL/OSL distributions alongside the CDF;
- **prefix-cache hit rate** surfaced from the `aiperf.prefix_cache` block;
- an optional per-turn **timeline (Gantt)** of one batch;
- an **A/B `compare` view** (`compare.html`): configs ranked by tail bubble,
  goodput/throughput bars, and overlaid completion CDFs, reusing the same
  renderer so two runs can be eyeballed side by side.

## Automated prefix-cache / server-metrics scrape

Capture the served engine's `/metrics` (prefix-cache hit rate, running/waiting
queue depth) around a run and fold it into `report.json`, instead of relying only
on what the client-side export exposes.

## More calibration profiles

Ship additional example `--distribution` profiles (beyond the bundled long-tail
one) for different workload shapes, and a small helper to fit a distribution
JSON from a user's own measured OSL percentiles.

## Phase 2 — RL-loop orchestration

Layer the RL loop on top of this static-batch harness: lag-1 admission control
and a refit-bubble model, measured against goodput anchors. The Phase-1 static
batch maps directly onto a lag-1-gated fixed batch, so this harness is the
substrate the RL-loop model builds on.
