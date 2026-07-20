"""Synthesize a Mooncake multi-turn trace reproducing the RL rollout long tail."""
import argparse
import json
import random

from rl_traces_bench.distributions import osl_sampler
from rl_traces_bench.turn_structure import load_turn_counts, sample_turn_count, split_osl
from rl_traces_bench.prompt_model import per_turn_isl, hash_ids_for


def _pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, round((p / 100) * (len(v) - 1)))]


def build_trace(num_rollouts, seed, block_size, osl_level, system_tokens,
                user_turn_tokens, shared_blocks, turn_counts):
    rng = random.Random(seed)
    sampler = osl_sampler()
    records, all_osl = [], []
    # reserve a per-rollout block namespace wide enough for the largest prompt
    base_stride = 1_000_000
    for sid in range(num_rollouts):
        turns = sample_turn_count(rng, turn_counts)
        if osl_level == "per_turn":
            osls = [sampler.sample(rng.random()) for _ in range(turns)]
        else:  # per_rollout: draw a rollout total, split across turns
            osls = split_osl(sampler.sample(rng.random()), turns, rng)
        isls = per_turn_isl(osls, system_tokens, user_turn_tokens)
        hids = hash_ids_for(isls, block_size, rollout_base=(sid + 1) * base_stride,
                            shared_blocks=shared_blocks)
        for t in range(turns):
            records.append({
                "session_id": str(sid), "turn_idx": t, "timestamp": 0,  # aiperf Mooncake requires str session_id
                "input_length": isls[t], "output_length": osls[t],
                "hash_ids": hids[t],
            })
            if osl_level == "per_turn":
                all_osl.append(osls[t])
        if osl_level == "per_rollout":
            all_osl.append(sum(osls))
    stats = {
        "num_rollouts": num_rollouts, "num_records": len(records),
        "osl_level": osl_level,
        "osl_p50": _pct(all_osl, 50), "osl_p90": _pct(all_osl, 90),
        "osl_p95": _pct(all_osl, 95), "osl_p99": _pct(all_osl, 99),
        "osl_max": max(all_osl),
    }
    return records, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-rollouts", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--osl-level", choices=["per_turn", "per_rollout"], default="per_turn")
    ap.add_argument("--system-tokens", type=int, default=300)
    ap.add_argument("--user-turn-tokens", type=int, default=200)
    ap.add_argument("--shared-blocks", type=int, default=1)
    ap.add_argument("--turn-counts", default="data/turn_counts.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    counts = load_turn_counts(a.turn_counts)
    recs, stats = build_trace(a.num_rollouts, a.seed, a.block_size, a.osl_level,
                              a.system_tokens, a.user_turn_tokens, a.shared_blocks, counts)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump(stats, open(a.out + ".stats.json", "w"), indent=2)
    print(f"wrote {len(recs)} records / {a.num_rollouts} rollouts -> {a.out}")
    print("stats:", json.dumps(stats))


if __name__ == "__main__":
    main()
