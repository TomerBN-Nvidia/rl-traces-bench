#!/usr/bin/env python3
"""Probe the downloaded RL rollout HTML traces for embedded data.

Extracts the JS/JSON data blobs and reports schema + distributions WITHOUT
loading the 40MB+ files into a chat context.

Usage:
    python3 probe_traces.py plotly <all_rollouts_timeline_first_rollout_zero.html>
    python3 probe_traces.py nemorl <nemorl_trace.html>

Findings (2026-07-19):
  - Plotly file: per-turn (Rollout Id, Turn, Start, Duration) for 30,680 model_call
    bars across 1024 rollouts. NO token counts (ISL/OSL), NO prefix-cache signal.
  - nemorl file: 107,617 phase events; real time-domain long tail in
    timing/rollout/total (p50=84s, p99=909s, max=1669s) + goodput SUMMARY
    (exposed_generation ~50%, buffer_starvation ~49%, refit_bubble ~120s).
    meta=null on all events -> NO tokens, NO per-turn granularity.
"""
import json
import sys


def _slice_bracket(text, start_idx, open_c, close_c):
    """Return the balanced open_c..close_c substring starting at/after start_idx."""
    j = text.find(open_c, start_idx)
    depth = 0
    for k in range(j, len(text)):
        c = text[k]
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return text[j:k + 1]
    return None


def _pct(v, p):
    v = sorted(v)
    if not v:
        return float("nan")
    idx = min(len(v) - 1, int(round((p / 100) * (len(v) - 1))))
    return v[idx]


def probe_nemorl(path):
    data = open(path, encoding="utf-8", errors="replace").read()
    summ = _slice_bracket(data, data.find("const SUMMARY"), "{", "}")
    print("=== SUMMARY (goodput phase breakdown, seconds) ===")
    print(summ[:2500] if summ else "NOT FOUND")

    events = json.loads(_slice_bracket(data, data.find("const ALL_EVENTS"), "[", "]"))
    print(f"\n=== ALL_EVENTS n={len(events)} keys={list(events[0].keys())} ===")

    from collections import defaultdict
    dur = defaultdict(list)
    for e in events:
        el = e.get("elapsed")
        if isinstance(el, (int, float)) and el > 0:
            dur[e["label"]].append(el)
    for lab in ("timing/rollout/total", "timing/rollout/await_results",
                "exposed_generation", "idle/buffer_starvation",
                "idle/refit_bubble", "weight_sync", "policy_training"):
        v = dur.get(lab, [])
        if not v:
            print(f"{lab:32s} (no per-event elapsed)")
            continue
        print(f"{lab:32s} n={len(v):6d} min={min(v):8.2f} p50={_pct(v,50):8.2f} "
              f"p90={_pct(v,90):8.2f} p99={_pct(v,99):8.2f} max={max(v):8.2f} sum={sum(v):9.1f}")


def probe_plotly(path):
    """Plotly Gantt: per-bar metadata is baked into the `text` field as
    'Key: value<br>...'. Reports label set + model_call turn distribution."""
    data = open(path, encoding="utf-8", errors="replace").read()
    # 2nd Plotly.newPlot is the figure (1st is library code).
    i1 = data.find("Plotly.newPlot")
    i2 = data.find("Plotly.newPlot", i1 + 1)
    traces = json.loads(_slice_bracket(data, i2, "[", "]"))
    print(f"=== {len(traces)} bar traces; names: "
          f"{sorted({t.get('name') for t in traces})} ===")
    mc = next((t for t in traces if t.get("name") == "model_call"), None)
    if not mc:
        print("no model_call trace")
        return
    turns, durs = [], []
    for txt, dur in zip(mc.get("text", []), mc.get("x", [])):
        for kv in str(txt).split("<br>"):
            if kv.startswith("Turn:"):
                turns.append(int(kv.split(":", 1)[1]))
        if isinstance(dur, (int, float)):
            durs.append(dur)
    print(f"model_call bars={len(durs)}")
    print(f"turn idx    min={min(turns)} p50={_pct(turns,50)} p90={_pct(turns,90)} max={max(turns)}")
    print(f"duration(s) min={min(durs):.2f} p50={_pct(durs,50):.2f} "
          f"p90={_pct(durs,90):.2f} p99={_pct(durs,99):.2f} max={max(durs):.2f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    mode, path = sys.argv[1], sys.argv[2]
    {"plotly": probe_plotly, "nemorl": probe_nemorl}[mode](path)
