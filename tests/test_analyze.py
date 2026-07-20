import json

from rl_traces_bench.analyze import (load_profile_export, compute_report,
                             validate_token_domain, validate_time_domain,
                             token_targets_from_distribution, load_aiperf_summary)
from rl_traces_bench.distributions import default_distribution_path
from rl_traces_bench.metrics import output_token_throughput, request_throughput

FIX = "tests/fixtures/profile_export_sample.jsonl"


def _per_rollout_records():
    """Three sessions, split across multiple turns each, whose per-ROLLOUT
    (session) OSL totals are engineered to land exactly on/off the default
    token-domain targets (p50=654, p95=33212, p99=57067). If validation summed
    per-TURN instead of per-ROLLOUT, these per-turn values (100/327 x2/etc.)
    would not reproduce the totals below, so this also guards the aggregation
    semantics, not just the pass/fail outcome."""
    return [
        {"session_id": "s0", "osl": 40},
        {"session_id": "s0", "osl": 60},          # s0 total = 100 (smallest)
        {"session_id": "s1", "osl": 327},
        {"session_id": "s1", "osl": 327},          # s1 total = 654 (== p50 target, exact)
        {"session_id": "s2", "osl": 30000},
        {"session_id": "s2", "osl": 27067},        # s2 total = 57067 (== p99 target, exact;
                                                    # far from p95 target 33212)
    ]

def test_load_and_report():
    recs = load_profile_export(FIX)
    assert len(recs) == 3
    assert {r["session_id"] for r in recs} == {"0", "1"}
    rep = compute_report(recs)
    assert rep["makespan_s"] == 100.0            # straggler dominates
    assert rep["completion_p50_s"] in (10.0, 100.0)
    assert rep["num_sessions"] == 2

def test_validate_time_domain_passes_for_matching_shape():
    # 100-point completion distribution whose p50/p99/max match the real trace shape
    # (84 / 909 / 1669): index math -> v[50]=84, v[98]=909, v[99]=1669.
    comps = [84.0] * 51 + [909.0] * 48 + [1669.0]
    assert len(comps) == 100
    assert validate_time_domain(comps)["passed"] is True

def test_validate_time_domain_fails_for_flat_distribution():
    comps = [100.0] * 100          # no tail -> ratios ~1.0, far from ref -> fails
    assert validate_time_domain(comps)["passed"] is False

def test_validate_token_domain_aggregates_per_rollout_not_per_turn():
    # each session's OSL is split across two turns; the realized percentiles
    # must reflect the per-session SUM, not the individual per-turn values
    # (100/327/327/30000/27067 would never produce 654 or 57067 on their own).
    recs = _per_rollout_records()
    out = validate_token_domain(recs)
    assert out["realized"] == {50: 654, 95: 57067, 99: 57067}

def test_validate_token_domain_default_unchanged():
    recs = _per_rollout_records()
    default = validate_token_domain(recs)
    explicit_default = validate_token_domain(recs, targets={50: 654, 95: 33212, 99: 57067})
    assert default == explicit_default
    # per-rollout p50 (654) and p99 (57067) hit the default targets exactly;
    # p95 (57067, same index as p99 with only 3 sessions) is far from 33212.
    assert default["checks"][50] is True
    assert default["checks"][95] is False
    assert default["checks"][99] is True
    assert default["passed"] is False

def test_validate_token_domain_custom_targets_change_result():
    recs = _per_rollout_records()
    # a custom target matching the realized p95 should pass where the
    # hardcoded default target fails
    custom = validate_token_domain(recs, targets={50: 654, 95: 57067, 99: 57067})
    assert custom["checks"] == {50: True, 95: True, 99: True}
    assert custom["passed"] is True

def test_token_targets_from_distribution_matches_default_anchors():
    targets = token_targets_from_distribution(default_distribution_path())
    assert targets == {50: 654, 95: 33212, 99: 57067}

def test_analyze_main_accepts_distribution_flag(tmp_path):
    import sys, json as _json
    from rl_traces_bench.analyze import main
    out_json = tmp_path / "report.json"
    argv = ["analyze", "--export", FIX, "--distribution", default_distribution_path(),
            "--out-html", str(tmp_path / "r.html"), "--out-json", str(out_json)]
    old = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old
    data = _json.loads(out_json.read_text())
    # with the default distribution's derived targets, this should match the
    # no-flag default behavior exactly. FIX's two sessions have per-rollout
    # (session) OSL totals of 1308 (session "0": 654+654 across two turns) and
    # 57067 (session "1"), so p50=1308 misses the 654 target, p95=57067 misses
    # the 33212 target, and p99=57067 hits the 57067 target exactly.
    assert data["validate_token"]["checks"] == {"50": False, "95": False, "99": True}

def test_main_writes_json_report(tmp_path):
    import sys, json as _json
    from rl_traces_bench.analyze import main
    out_json = tmp_path / "report.json"
    argv = ["analyze", "--export", "tests/fixtures/profile_export_sample.jsonl",
            "--out-html", str(tmp_path / "r.html"), "--out-json", str(out_json)]
    old = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = old
    data = _json.loads(out_json.read_text())
    assert data["num_sessions"] == 2 and data["makespan_s"] == 100.0


def test_output_token_throughput_and_request_throughput_known_values():
    # makespan = 10.0s (max end); total osl = 654+654+57067 = 58375 tokens
    # over 3 records -> 58375/10 tok/s, 3/10 req/s.
    recs = [
        {"end": 4.0, "osl": 654},
        {"end": 10.0, "osl": 654},
        {"end": 5.0, "osl": 57067},
    ]
    assert output_token_throughput(recs) == 58375 / 10.0
    assert request_throughput(recs) == 3 / 10.0


def test_compute_report_includes_throughput_fields():
    recs = load_profile_export(FIX)
    rep = compute_report(recs)
    assert rep["output_tok_throughput"] == output_token_throughput(recs)
    assert rep["request_throughput"] == request_throughput(recs)


def test_load_aiperf_summary_folds_native_metrics_and_faithful_gate(tmp_path):
    export_dir = tmp_path / "artifacts"
    export_dir.mkdir()
    export_path = export_dir / "profile_export.jsonl"
    export_path.write_text("")
    summary = {
        "aiperf_version": "0.11.0",
        "error_request_count": {"avg": 0},
        "request_count": {"avg": 8},
        "output_token_throughput": {"avg": 1234.5},
        "request_throughput": {"avg": 6.7},
        "request_latency": {"avg": 100.0, "p50": 90.0, "p90": 150.0, "p99": 200.0},
        "time_to_first_token": {"avg": 20.0, "p50": 18.0, "p90": 30.0, "p99": 40.0},
        "inter_token_latency": {"avg": 5.0, "p50": 4.5, "p90": 8.0, "p99": 10.0},
        "input_sequence_length": {"avg": 500.0, "max": 1200.0},
        "output_sequence_length": {"avg": 654.0, "max": 57067.0},
    }
    (export_dir / "profile_export_aiperf.json").write_text(json.dumps(summary))

    out = load_aiperf_summary(str(export_path))
    assert out["aiperf_version"] == "0.11.0"
    assert out["error_request_count"] == 0
    assert out["request_count"] == 8
    assert out["faithful"] is True                 # errs == 0 -> coherence gate passes
    assert out["output_token_throughput_tok_s"] == 1234.5
    assert out["request_throughput_req_s"] == 6.7
    assert out["request_latency_ms"] == {"avg": 100.0, "p50": 90.0, "p90": 150.0, "p99": 200.0}
    assert out["input_sequence_length"] == {"avg": 500.0, "max": 1200.0}


def test_load_aiperf_summary_faithful_false_when_errors_present(tmp_path):
    export_dir = tmp_path / "artifacts"
    export_dir.mkdir()
    export_path = export_dir / "profile_export.jsonl"
    export_path.write_text("")
    summary = {"error_request_count": {"avg": 2}, "request_count": {"avg": 8}}
    (export_dir / "profile_export_aiperf.json").write_text(json.dumps(summary))

    out = load_aiperf_summary(str(export_path))
    assert out["error_request_count"] == 2
    assert out["faithful"] is False                # errs > 0 -> run did not replay faithfully


def test_load_aiperf_summary_returns_none_when_missing(tmp_path):
    export_path = tmp_path / "profile_export.jsonl"
    export_path.write_text("")
    assert load_aiperf_summary(str(export_path)) is None
