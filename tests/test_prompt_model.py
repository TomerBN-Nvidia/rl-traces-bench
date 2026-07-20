from scripts.prompt_model import per_turn_isl, hash_ids_for

def test_isl_monotonic_growth():
    osls = [654, 654, 5000]
    isls = per_turn_isl(osls, system_tokens=300, user_turn_tokens=200)
    assert len(isls) == 3
    assert isls[0] == 300 + 200                      # system + first user msg
    assert isls[1] > isls[0] and isls[2] > isls[1]   # accumulates prior turns

def test_hash_ids_cumulative_and_shared():
    isls = [1024, 2048, 4096]
    hids = hash_ids_for(isls, block_size=512, rollout_base=1000, shared_blocks=1)
    # turn N block set is a superset of turn N-1 (growing prefix reuse)
    assert set(hids[0]).issubset(set(hids[1]))
    assert set(hids[1]).issubset(set(hids[2]))
    # first block is the globally-shared system block (id 0), rest are per-rollout
    assert hids[0][0] == 0
    assert all(b == 0 or b >= 1000 for b in hids[2])
    # block counts match ceil(ISL/512)
    assert len(hids[0]) == 2 and len(hids[1]) == 4 and len(hids[2]) == 8

def test_hash_ids_rejects_non_monotonic_isls():
    import pytest
    with pytest.raises(ValueError):
        hash_ids_for([4096, 1024], block_size=512, rollout_base=1000, shared_blocks=1)
