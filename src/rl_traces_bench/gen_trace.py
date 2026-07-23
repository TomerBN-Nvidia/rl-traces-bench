"""Synthesize a Mooncake multi-turn trace reproducing the RL rollout long tail."""
import argparse
import json
import random

from rl_traces_bench.distributions import osl_sampler, OSL_ANCHORS, load_distribution, default_distribution_path
from rl_traces_bench.turn_structure import load_turn_counts, sample_turn_count, split_osl
from rl_traces_bench.prompt_model import per_turn_isl, hash_ids_for


def _pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, round((p / 100) * (len(v) - 1)))]


def build_trace(num_rollouts, seed, block_size, osl_level, system_tokens,
                user_turn_tokens, shared_blocks, turn_counts, anchors=OSL_ANCHORS):
    rng = random.Random(seed)
    sampler = osl_sampler(anchors)
    records, all_osl = [], []
    peak_contexts = []          # per-rollout max served context (input_length + accumulated OSL)
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
        # served context while generating turn t = cumulative user-side input_length[t]
        # (system + prior user turns) + all assistant OSL up to and including t (aiperf
        # accumulates the conversation). This is what must fit under --max-model-len.
        acc_osl = 0
        peak = 0
        for t in range(turns):
            records.append({
                "session_id": str(sid), "turn_idx": t, "timestamp": 0,  # aiperf Mooncake requires str session_id
                "input_length": isls[t], "output_length": osls[t],
                "hash_ids": hids[t],
            })
            peak = max(peak, isls[t] + acc_osl + osls[t])
            acc_osl += osls[t]
            if osl_level == "per_turn":
                all_osl.append(osls[t])
        peak_contexts.append(peak)
        if osl_level == "per_rollout":
            all_osl.append(sum(osls))
    stats = {
        "num_rollouts": num_rollouts, "num_records": len(records),
        "osl_level": osl_level,
        "osl_p50": _pct(all_osl, 50), "osl_p90": _pct(all_osl, 90),
        "osl_p95": _pct(all_osl, 95), "osl_p99": _pct(all_osl, 99),
        "osl_max": max(all_osl),
        "max_context": max(peak_contexts),         # largest served context across rollouts
        "context_p99": _pct(peak_contexts, 99),
    }
    return records, stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-rollouts", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--osl-level", choices=["per_turn", "per_rollout"], default="per_turn")
    ap.add_argument("--system-tokens", type=int, default=300)
    ap.add_argument("--user-turn-tokens", type=int, default=200)
    ap.add_argument("--shared-blocks", type=int, default=1)
    ap.add_argument("--distribution", default=None,
                    help="path to a distribution JSON with osl_anchors + turn_counts "
                         "(default: packaged example_longtail.json)")
    ap.add_argument("--turn-counts", default=None,
                    help="override turn_counts, ignoring --distribution's turn_counts")
    ap.add_argument("--max-model-len", type=int, default=131072,
                    help="target serving context window; gen-trace warns if the trace's "
                         "accumulated multi-turn context would exceed it (default 131072)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dist_path = a.distribution if a.distribution is not None else default_distribution_path()
    dist = load_distribution(dist_path)
    anchors = [tuple(pair) for pair in dist["osl_anchors"]]
    counts = load_turn_counts(a.turn_counts) if a.turn_counts is not None else dist["turn_counts"]
    recs, stats = build_trace(a.num_rollouts, a.seed, a.block_size, a.osl_level,
                              a.system_tokens, a.user_turn_tokens, a.shared_blocks, counts,
                              anchors=anchors)
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    json.dump(stats, open(a.out + ".stats.json", "w"), indent=2)
    print(f"wrote {len(recs)} records / {a.num_rollouts} rollouts -> {a.out}")
    print("stats:", json.dumps(stats))
    # Guard the pathology that silently breaks a run: in multi-turn CHAT replay aiperf
    # accumulates each turn's OSL into the context, so a per_turn heavy-tail trace can
    # accumulate far past the serving window -> HTTP 400s -> faithful=false. Warn loudly
    # and point at the fix rather than letting it surface as mystery errors on-cluster.
    if stats["max_context"] > a.max_model_len:
        import sys
        hint = (" Use --osl-level per_rollout (one long-tail OSL total per rollout, split "
                "across turns), which bounds accumulation." if a.osl_level == "per_turn"
                else " Lower the distribution's OSL tail, --user-turn-tokens, or turn counts.")
        print(f"WARNING: max accumulated context {stats['max_context']} exceeds "
              f"--max-model-len {a.max_model_len}. Serving this trace will 400 on the "
              f"largest rollouts (faithful=false).{hint}", file=sys.stderr)


if __name__ == "__main__":
    main()
