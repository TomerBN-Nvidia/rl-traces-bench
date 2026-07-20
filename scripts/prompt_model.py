"""Per-turn ISL growth (accumulated conversation) + cumulative prefix hash_ids."""
import math


def per_turn_isl(osls, system_tokens, user_turn_tokens):
    isls = []
    ctx = system_tokens
    for i in range(len(osls)):
        ctx += user_turn_tokens          # this turn's incoming user/tool message
        isls.append(ctx)
        ctx += osls[i]                    # assistant response folds into next prompt
    return isls


def hash_ids_for(isls, block_size, rollout_base, shared_blocks):
    """Block-align each ISL; first `shared_blocks` are global (system), rest per-rollout.
    Cumulative: growing ISL => later turns reuse earlier blocks + append new ones.
    Precondition: `isls` must be non-decreasing."""
    if any(a > b for a, b in zip(isls, isls[1:])):
        raise ValueError("hash_ids_for requires non-decreasing isls (cumulative prefix growth)")
    out = []
    for isl in isls:
        nblocks = math.ceil(isl / block_size)
        blocks = []
        for b in range(nblocks):
            if b < shared_blocks:
                blocks.append(b)                       # global shared prefix (e.g. system)
            else:
                blocks.append(rollout_base + b)        # per-rollout unique block
        out.append(blocks)
    return out
