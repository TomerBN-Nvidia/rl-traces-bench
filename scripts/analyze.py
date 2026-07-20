"""Ingest aiperf per-request export, compute long-tail report + dual validation."""
import argparse
import json

from scripts.metrics import (percentiles, session_completions, makespan,
                             tail_bubble, goodput_proxy,
                             output_token_throughput, request_throughput)

def _metric(metrics, *names):
    """Return the .value of the first present metric. aiperf stores each metric as
    {"value": X, "unit": "..."} inside the record's `metrics` dict."""
    for n in names:
        m = metrics.get(n)
        if isinstance(m, dict) and "value" in m:
            return m["value"]
    return None


def load_profile_export(path):
    """Parse an aiperf `profile_export.jsonl` (aiperf >=0.11 nested schema).

    Each record is {"metadata": {...}, "metrics": {name: {value, unit}}}:
      - metadata.conversation_id  -> session id (multi-turn grouping key)
      - metadata.turn_index       -> turn
      - metadata.request_start_ns / request_end_ns -> absolute ns timestamps
      - metadata.benchmark_phase  -> filter to "profiling" (drop warmup)
      - metrics.input_sequence_length / output_sequence_length -> ISL / OSL
      - metrics.time_to_first_token (ms) -> TTFT
    Absolute timestamps are normalized to batch-relative seconds (t0 = earliest
    request start) so makespan/goodput are meaningful rather than ~1.0."""
    raw = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        meta = d.get("metadata", d)          # tolerate a flat export too
        metrics = d.get("metrics", {})
        if meta.get("benchmark_phase") not in (None, "profiling"):
            continue                         # skip warmup / non-profiling records
        raw.append({
            "session_id": meta.get("conversation_id", meta.get("session_id")),
            "turn": meta.get("turn_index", meta.get("turn")),
            "start_ns": meta.get("request_start_ns"),
            "end_ns": meta.get("request_end_ns"),
            "isl": _metric(metrics, "input_sequence_length"),
            "osl": _metric(metrics, "output_sequence_length", "output_token_count",
                           "usage_completion_tokens"),
            "ttft_ms": _metric(metrics, "time_to_first_token"),
        })
    starts = [r["start_ns"] for r in raw if r["start_ns"] is not None]
    t0 = min(starts) if starts else 0
    out = []
    for r in raw:
        s = ((r["start_ns"] or t0) - t0) / 1e9
        e = ((r["end_ns"] or r["start_ns"] or t0) - t0) / 1e9
        out.append({
            "session_id": r["session_id"], "turn": r["turn"],
            "start": s, "end": e,
            "isl": r["isl"], "osl": r["osl"],
            "ttft": (r["ttft_ms"] / 1000.0) if r["ttft_ms"] is not None else 0.0,
        })
    return out


def load_aiperf_summary(export_path):
    """Fold aiperf's native summary (profile_export_aiperf.json, in the same
    artifact dir) into our report — its authoritative throughput/latency/TTFT/ITL
    metrics, plus error_request_count (a coherence gate: a run with errors did NOT
    replay the full batch, so its rollout metrics are untrustworthy)."""
    import glob
    import os
    d = os.path.dirname(os.path.abspath(export_path))
    fs = glob.glob(os.path.join(d, "**", "profile_export_aiperf.json"), recursive=True) \
        or glob.glob(os.path.join(d, "profile_export_aiperf.json"))
    if not fs:
        return None
    s = json.load(open(fs[0]))

    def val(name):                       # scalar metric -> its avg
        m = s.get(name)
        return m.get("avg") if isinstance(m, dict) else m

    def stats(name, keys=("avg", "p50", "p90", "p99")):
        m = s.get(name) or {}
        return {k: m.get(k) for k in keys if isinstance(m, dict) and m.get(k) is not None}

    errs = val("error_request_count") or 0
    return {
        "aiperf_version": s.get("aiperf_version"),
        "error_request_count": errs,
        "request_count": val("request_count"),
        "faithful": errs == 0,           # coherence gate
        "output_token_throughput_tok_s": val("output_token_throughput"),
        "request_throughput_req_s": val("request_throughput"),
        "request_latency_ms": stats("request_latency"),
        "time_to_first_token_ms": stats("time_to_first_token"),
        "inter_token_latency_ms": stats("inter_token_latency"),
        "input_sequence_length": stats("input_sequence_length", ("avg", "max")),
        "output_sequence_length": stats("output_sequence_length", ("avg", "max")),
    }


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
        "output_tok_throughput": output_token_throughput(records),
        "request_throughput": request_throughput(records),
    }


def validate_token_domain(records, tol=0.15):
    """The published OSL distribution is per-SAMPLE (per-rollout total), so compare
    per-ROLLOUT total OSL (summed over a session's turns) — not per-turn OSL."""
    from collections import defaultdict
    tot = defaultdict(int)
    for r in records:
        if r["osl"]:
            tot[r["session_id"]] += r["osl"]
    osl = list(tot.values())
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
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    recs = load_profile_export(a.export)
    rep = compute_report(recs)
    rep["validate_token"] = validate_token_domain(recs)
    rep["validate_time"] = validate_time_domain(session_completions(recs))
    rep["aiperf"] = load_aiperf_summary(a.export)   # native perf metrics + error gate
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rep.items())
    open(a.out_html, "w").write(f"<html><body><h1>RL long-tail report</h1>"
                                f"<table border=1>{rows}</table></body></html>")
    if a.out_json:
        with open(a.out_json, "w") as f:
            json.dump(rep, f, indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
