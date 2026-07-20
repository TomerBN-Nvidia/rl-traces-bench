import json, hashlib
from rl_traces_bench.gen_trace import build_trace
from rl_traces_bench.distributions import load_distribution, default_distribution_path


def _lines(recs):
    return "".join(json.dumps(r) + "\n" for r in recs)

def _turn_counts():
    return load_distribution(default_distribution_path())["turn_counts"]

def test_golden_per_turn_byte_identical():
    counts = _turn_counts()
    recs, _ = build_trace(8, 0, 512, "per_turn", 300, 200, 1, counts)
    got = hashlib.sha256(_lines(recs).encode()).hexdigest()
    assert got == "783b5825a3305960036b0dbaf8f480056f233e86992fedc489be25c51207f8e9"

def test_golden_per_rollout_byte_identical():
    counts = _turn_counts()
    recs, _ = build_trace(8, 0, 512, "per_rollout", 300, 200, 1, counts)
    got = hashlib.sha256(_lines(recs).encode()).hexdigest()
    assert got == "d74a451fa5b3fe8a50ab5a212362651de91071ec6d4bd8a9cd29594c9a72de1c"
