"""Per-turn ISL growth (accumulated conversation) + cumulative prefix hash_ids."""
import math


def per_turn_isl(osls, system_tokens, user_turn_tokens):
    """Per-turn `input_length` for CHAT multi-turn replay.

    On the chat endpoint aiperf ACCUMULATES the conversation itself — it appends each
    turn's assistant response (OSL) to the context for the next turn. So `input_length`
    must carry only the system prompt + this-and-prior *user/tool* messages, NOT the
    assistant OSL. If we folded OSL in here too, aiperf would double-count it and the
    context overflows max-model-len on long rollouts (observed: ISL ~2x, 400 errors).
    The server-side cumulative ISL that aiperf reconstructs is this value + ΣOSL, which
    is exactly the intended growing context.

    `osls` is used only for the turn count; ISL grows by `user_turn_tokens` per turn
    (still monotonic, so cumulative hash_ids remain valid)."""
    isls = []
    ctx = system_tokens
    for _ in osls:
        ctx += user_turn_tokens          # this turn's incoming user/tool message
        isls.append(ctx)                 # aiperf adds the assistant OSL responses itself
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
