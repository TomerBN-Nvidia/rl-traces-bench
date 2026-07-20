import random
from scripts.distributions import osl_sampler, OSL_ANCHORS

def _pct(v, p):
    v = sorted(v); return v[min(len(v)-1, round((p/100)*(len(v)-1)))]

def test_realized_percentiles_match_anchors():
    s = osl_sampler()
    vals = s.sample_n(200_000, random.Random(0))
    # Each anchor percentile must land within 8% of its target value.
    for q, target in OSL_ANCHORS:
        if q in (0.0, 1.0):
            continue
        got = _pct(vals, q * 100)
        assert abs(got - target) <= 0.08 * target, (q, got, target)

def test_bounds_and_80pct_under_10k():
    s = osl_sampler()
    vals = s.sample_n(100_000, random.Random(1))
    assert min(vals) >= 1 and max(vals) <= 65_489
    frac = sum(1 for v in vals if v < 10_000) / len(vals)
    assert 0.78 <= frac <= 0.82   # published: 80% < 10k
