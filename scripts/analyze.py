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
    ap.add_argument("--out-json", default=None)
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
