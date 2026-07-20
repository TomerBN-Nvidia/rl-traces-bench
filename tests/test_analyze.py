from rl_traces_bench.analyze import (load_profile_export, compute_report,
                             validate_token_domain, validate_time_domain)

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
