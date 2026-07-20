from rl_traces_bench.gen_trace import build_trace
from rl_traces_bench.distributions import load_distribution, default_distribution_path


def _turn_counts():
    return load_distribution(default_distribution_path())["turn_counts"]

def test_trace_schema_and_sessions():
    counts = _turn_counts()
    recs, stats = build_trace(num_rollouts=50, seed=0, block_size=512,
                              osl_level="per_turn", system_tokens=300,
                              user_turn_tokens=200, shared_blocks=1, turn_counts=counts)
    sessions = {r["session_id"] for r in recs}
    assert len(sessions) == 50
    for r in recs:
        assert set(r) == {"session_id","turn_idx","timestamp","input_length","output_length","hash_ids"}
        assert isinstance(r["session_id"], str)   # aiperf Mooncake requires str session_id
        assert r["output_length"] >= 1 and r["input_length"] >= 1
        assert r["hash_ids"] == sorted(r["hash_ids"])
    # within a session, ISL grows and hash_ids are cumulative
    s0 = sorted([r for r in recs if r["session_id"] == "0"], key=lambda r: r["turn_idx"])
    for a, b in zip(s0, s0[1:]):
        assert b["input_length"] >= a["input_length"]
        assert set(a["hash_ids"]).issubset(set(b["hash_ids"]))

def test_per_turn_osl_matches_anchors():
    counts = _turn_counts()
    recs, stats = build_trace(num_rollouts=3000, seed=1, block_size=512,
                              osl_level="per_turn", system_tokens=300,
                              user_turn_tokens=200, shared_blocks=1, turn_counts=counts)
    # realized per-turn OSL median within 12% of published 654
    assert abs(stats["osl_p50"] - 654) <= 0.12 * 654

def test_per_rollout_stats_are_per_rollout_level():
    from collections import defaultdict
    counts = _turn_counts()
    recs, stats = build_trace(num_rollouts=500, seed=2, block_size=512,
                              osl_level="per_rollout", system_tokens=300,
                              user_turn_tokens=200, shared_blocks=1, turn_counts=counts)
    # reconstruct rollout totals from the records themselves
    tot = defaultdict(int)
    for r in recs:
        tot[r["session_id"]] += r["output_length"]
    totals = sorted(tot.values())
    def pct(v, p): return v[min(len(v) - 1, round((p / 100) * (len(v) - 1)))]
    assert len(totals) == 500                       # one value per rollout
    assert stats["osl_p50"] == pct(totals, 50)      # stats computed over per-rollout totals
    assert stats["osl_p99"] == pct(totals, 99)
