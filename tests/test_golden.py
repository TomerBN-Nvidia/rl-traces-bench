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
    assert got == "e5cc63399dbb2f6a591c357cfcf7b010f4188f85d82c285576148be85d4806cd"

def test_golden_per_rollout_byte_identical():
    counts = _turn_counts()
    recs, _ = build_trace(8, 0, 512, "per_rollout", 300, 200, 1, counts)
    got = hashlib.sha256(_lines(recs).encode()).hexdigest()
    assert got == "4e410a236dae849c5f931d94269cbff192b2f28678e5d89a0aed0d3f2d3b4662"
