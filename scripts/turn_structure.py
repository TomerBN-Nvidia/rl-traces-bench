"""Turn-count sampling (empirical, from File-1) + per-rollout OSL splitting."""
import json


def load_turn_counts(path):
    return json.load(open(path))


def sample_turn_count(rng, counts):
    return counts[rng.randrange(len(counts))]


def split_osl(total, turns, rng):
    """Split `total` output tokens across `turns` turns, each >=1, summing exactly."""
    if turns <= 1:
        return [max(1, total)]
    # every turn must emit >=1 output token (aiperf/vLLM requires >=1 output
    # token per request), so total must be >= turns to be feasible
    total = max(total, turns)
    # random positive weights -> proportional integer split with remainder fix-up
    weights = [rng.random() + 1e-6 for _ in range(turns)]
    s = sum(weights)
    parts = [max(1, int(total * w / s)) for w in weights]
    diff = total - sum(parts)
    i = 0
    while diff != 0:                       # distribute rounding remainder
        j = i % turns
        if diff > 0:
            parts[j] += 1; diff -= 1
        elif parts[j] > 1:
            parts[j] -= 1; diff += 1
        i += 1
    return parts
