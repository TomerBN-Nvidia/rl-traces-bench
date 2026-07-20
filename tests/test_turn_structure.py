import random
from rl_traces_bench.turn_structure import load_turn_counts, sample_turn_count, split_osl

def test_load_counts_nonempty():
    counts = load_turn_counts("tests/fixtures/turn_counts_sample.json")
    assert len(counts) == 1024
    assert min(counts) >= 1 and max(counts) <= 30

def test_sample_turn_count_in_range():
    counts = load_turn_counts("tests/fixtures/turn_counts_sample.json")
    rng = random.Random(0)
    for _ in range(1000):
        t = sample_turn_count(rng, counts)
        assert t in counts

def test_split_osl_sums_and_floor():
    rng = random.Random(0)
    parts = split_osl(65489, 30, rng)
    assert len(parts) == 30
    assert all(p >= 1 for p in parts)
    assert sum(parts) == 65489

def test_split_osl_total_less_than_turns_terminates_and_feasible():
    import random
    parts = split_osl(15, 30, random.Random(0))   # total < turns: clamp to 30
    assert len(parts) == 30
    assert all(p >= 1 for p in parts)
    assert sum(parts) == 30

def test_split_osl_total_equals_turns():
    import random
    parts = split_osl(30, 30, random.Random(0))
    assert len(parts) == 30 and all(p >= 1 for p in parts) and sum(parts) == 30

def test_split_osl_single_turn():
    import random
    assert split_osl(0, 1, random.Random(0)) == [1]
    assert split_osl(5, 1, random.Random(0)) == [5]
