"""Static-batch long-tail metrics over normalized per-request records."""
from collections import defaultdict


def percentiles(vals, ps):
    v = sorted(vals)
    return {p: v[min(len(v) - 1, round((p / 100) * (len(v) - 1)))] for p in ps}


def sessionize(records):
    by = defaultdict(list)
    for r in records:
        by[r["session_id"]].append(r)
    return dict(by)


def session_completions(records, batch_start=0.0):
    """Per-session completion time = last turn end - batch start."""
    return [max(r["end"] for r in rs) - batch_start
            for rs in sessionize(records).values()]


def makespan(records):
    return max(r["end"] for r in records)


def tail_bubble(completions):
    """Idle bubble = makespan - p90(completion): time spent waiting on the tail."""
    p90 = percentiles(completions, [90])[90]
    return max(completions) - p90


def goodput_proxy(completions):
    """Fraction of batch wall time that is useful (mean completion / makespan)."""
    return (sum(completions) / len(completions)) / max(completions)


def output_token_throughput(records):
    """Total generated output tokens per second over the batch (tokens/sec)."""
    ms = makespan(records)
    return (sum(r["osl"] or 0 for r in records) / ms) if ms else 0.0


def request_throughput(records):
    """Requests (turns) completed per second over the batch (req/sec)."""
    ms = makespan(records)
    return (len(records) / ms) if ms else 0.0
