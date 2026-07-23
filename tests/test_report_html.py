from rl_traces_bench.report_html import render_report

_REP = {
    "num_sessions": 3, "num_requests": 7, "makespan_s": 582.6,
    "completion_p50_s": 66.6, "completion_p90_s": 344.0, "completion_p99_s": 573.0,
    "tail_bubble_s": 239.0, "goodput_proxy": 0.23,
    "output_tok_throughput": 826.0, "request_throughput": 2.1,
    "rollouts": [
        {"completion_s": 40.0, "total_osl": 500, "turns": 2},
        {"completion_s": 344.0, "total_osl": 33000, "turns": 12},
        {"completion_s": 573.0, "total_osl": 57000, "turns": 25},
    ],
    "validate_token": {"passed": True, "realized": {50: 640, 95: 34000, 99: 57200},
                       "checks": {50: True, 95: True, 99: True}},
    "validate_time": {"passed": True},
    "aiperf": {"faithful": True, "error_request_count": 0, "aiperf_version": "0.11.0"},
}


def test_render_report_is_self_contained_html():
    h = render_report(_REP)
    assert h.startswith("<!doctype html>")
    # no external/CDN dependencies — the file must work offline
    assert "http://" not in h and "https://" not in h
    assert "<script src" not in h and "<link" not in h


def test_render_report_includes_all_six_visuals_and_controls():
    h = render_report(_REP)
    for section in ("Completion-time CDF", "Goodput decomposition",
                    "Completion vs. rollout output length", "Token-domain validation"):
        assert section in h
    # headline tiles + status chips + table view + dark-mode toggle
    assert "Makespan" in h and "Goodput" in h
    assert "Faithful" in h and "Token-domain" in h and "Time-domain" in h
    assert 'id="tableview"' in h and 'id="themebtn"' in h and 'id="tablebtn"' in h


def test_render_report_embeds_per_rollout_data_for_client_charts():
    h = render_report(_REP)
    # the CDF crosshair reads window.__ROLLOUTS__; the scatter reads per-dot data-*
    assert "window.__ROLLOUTS__=" in h
    assert '"completion_s": 573.0' in h
    assert 'data-osl="57000"' in h


def test_token_validation_draws_custom_distribution_targets():
    # a custom --distribution supplies its own anchors; the chart must show THOSE,
    # not the packaged defaults, as the target reference lines.
    rep = dict(_REP, validate_token={
        "passed": True, "realized": {50: 900, 95: 12000, 99: 20000},
        "checks": {50: True, 95: True, 99: True},
        "targets": {50: 950, 95: 11800, 99: 21000}})
    h = render_report(rep)
    assert "target 21.0k" in h          # custom p99 anchor rendered
    assert "target 57.1k" not in h      # packaged default NOT shown


def test_render_report_status_chips_reflect_gate_state():
    # a failed token gate should surface the serious status color, not the good one
    rep = dict(_REP, validate_token={"passed": False, "realized": {}, "checks": {}})
    h = render_report(rep)
    assert "var(--serious)" in h


def test_render_report_tolerates_missing_optional_sections():
    # a minimal report (no aiperf, no validations, no rollouts) must still render
    minimal = {"num_sessions": 0, "num_requests": 0, "makespan_s": None,
               "goodput_proxy": None, "output_tok_throughput": 0}
    h = render_report(minimal)
    assert h.startswith("<!doctype html>") and "Makespan" in h


def test_render_report_escapes_title():
    h = render_report(_REP, title="<script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in h
    assert "&lt;script&gt;" in h


from rl_traces_bench.report_html import render_compare

_A = dict(_REP)
_B = dict(_REP, makespan_s=250.0, tail_bubble_s=90.0, goodput_proxy=0.40,
          output_tok_throughput=1900.0,
          rollouts=[{"completion_s": 20.0, "total_osl": 500, "turns": 2},
                    {"completion_s": 150.0, "total_osl": 33000, "turns": 12},
                    {"completion_s": 250.0, "total_osl": 57000, "turns": 25}])


def test_render_compare_is_self_contained_and_has_both_configs():
    h = render_compare({"baseline": _A, "mtp": _B})
    assert h.startswith("<!doctype html>")
    assert "http://" not in h and "https://" not in h
    assert "baseline" in h and "mtp" in h
    assert "Completion-time CDF" in h and "Tail bubble" in h


def test_render_compare_delta_tiles_only_for_two_configs():
    two = render_compare({"a": _A, "b": _B})
    assert "vs a" in two  # B-vs-A delta annotation present
    three = render_compare({"a": _A, "b": _B, "c": _A})
    assert "vs a" not in three  # deltas suppressed for >2 configs


def test_render_compare_embeds_per_config_series_for_overlaid_cdf():
    h = render_compare({"baseline": _A, "mtp": _B})
    assert "window.__COMPARE__=" in h
    assert '"name": "baseline"' in h and '"name": "mtp"' in h


def test_render_compare_color_follows_entity_not_rank():
    # baseline is config A (blue --cfgA) whether or not it wins on tail bubble
    h = render_compare({"baseline": _A, "mtp": _B})
    assert "--cfgA" in h and "--cfgB" in h
