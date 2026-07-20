from rl_traces_bench.compare import compare_reports

def test_ranks_by_tail_bubble():
    reports = {
        "baseline": {"tail_bubble_s": 900, "goodput_proxy": 0.33, "makespan_s": 1600},
        "mtp":      {"tail_bubble_s": 400, "goodput_proxy": 0.55, "makespan_s": 900},
    }
    rows = compare_reports(reports)
    assert [r["config"] for r in rows] == ["mtp", "baseline"]
    assert rows[0]["tail_bubble_s"] == 400
