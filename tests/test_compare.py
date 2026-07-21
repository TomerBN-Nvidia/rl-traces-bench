from rl_traces_bench.compare import compare_reports

def test_ranks_by_tail_bubble():
    reports = {
        "baseline": {"tail_bubble_s": 900, "goodput_proxy": 0.33, "makespan_s": 1600},
        "mtp":      {"tail_bubble_s": 400, "goodput_proxy": 0.55, "makespan_s": 900},
    }
    rows = compare_reports(reports)
    assert [r["config"] for r in rows] == ["mtp", "baseline"]
    assert rows[0]["tail_bubble_s"] == 400


def test_main_writes_compare_html(tmp_path):
    import json
    from rl_traces_bench.compare import main
    rep = {"num_sessions": 2, "makespan_s": 100.0, "completion_p90_s": 40.0,
           "tail_bubble_s": 60.0, "goodput_proxy": 0.3, "output_tok_throughput": 500.0,
           "rollouts": [{"completion_s": 40.0, "total_osl": 500, "turns": 1},
                        {"completion_s": 100.0, "total_osl": 40000, "turns": 5}]}
    a = tmp_path / "a.json"; b = tmp_path / "b.json"
    a.write_text(json.dumps(rep))
    b.write_text(json.dumps(dict(rep, makespan_s=60.0, tail_bubble_s=25.0, output_tok_throughput=1200.0)))
    out = tmp_path / "compare.html"
    main([f"baseline={a}", f"mtp={b}", "--out-html", str(out)])
    h = out.read_text()
    assert h.startswith("<!doctype html>")
    assert "baseline" in h and "mtp" in h and "Completion-time CDF" in h
