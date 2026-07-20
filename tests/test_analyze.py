from rl_traces_bench.analyze import (load_profile_export, compute_report,
                             validate_token_domain, validate_time_domain,
                             token_targets_from_distribution)
from rl_traces_bench.distributions import default_distribution_path

FIX = "tests/fixtures/profile_export_sample.jsonl"

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

def test_validate_token_domain_default_unchanged():
    recs = load_profile_export(FIX)
    default = validate_token_domain(recs)
    explicit_default = validate_token_domain(recs, targets={50: 654, 95: 33212, 99: 57067})
    assert default == explicit_default
    # sanity: fixture's realized p95 (57067) is far from the default target (33212)
    assert default["checks"][95] is False
    assert default["checks"][50] is True and default["checks"][99] is True

def test_validate_token_domain_custom_targets_change_result():
    recs = load_profile_export(FIX)
    # a custom target matching the fixture's realized p95 should pass where
    # the hardcoded default target fails
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
    # no-flag default behavior exactly
    assert data["validate_token"]["checks"] == {"50": True, "95": False, "99": True}

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
