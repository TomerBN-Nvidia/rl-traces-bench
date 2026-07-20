from rl_traces_bench.metrics import (percentiles, sessionize, session_completions,
                             makespan, tail_bubble, goodput_proxy)

REC = [  # two sessions: s0 finishes at 10s (2 turns), s1 is the straggler at 100s
    {"session_id": 0, "start": 0.0, "end": 4.0},
    {"session_id": 0, "start": 4.0, "end": 10.0},
    {"session_id": 1, "start": 0.0, "end": 100.0},
]

def test_session_completions_and_makespan():
    comps = session_completions(REC)
    assert sorted(comps) == [10.0, 100.0]
    assert makespan(REC) == 100.0

def test_tail_bubble_and_goodput():
    comps = [10.0, 100.0]
    # p90 of a 2-point set = the larger; bubble = max - p90
    assert tail_bubble(comps) == 0.0
    # goodput proxy = mean/max = 55/100
    assert abs(goodput_proxy(comps) - 0.55) < 1e-9

def test_percentiles():
    p = percentiles([1,2,3,4,5,6,7,8,9,10], [50,90])
    assert p[50] == 5 and p[90] == 9
