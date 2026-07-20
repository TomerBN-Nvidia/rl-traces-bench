"""Fit duration ~ ttft(isl) + itl*osl from Tau2 per-turn data; invert to estimate OSL."""
import argparse
import json


def _solve3(A, b):
    """Solve 3x3 linear system by Gaussian elimination (stdlib only)."""
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [x / piv for x in M[c]]
        for r in range(3):
            if r != c:
                f = M[r][c]
                M[r] = [M[r][k] - f * M[c][k] for k in range(4)]
    return [M[i][3] for i in range(3)]


def fit_time_model(rows):
    """OLS for dur = a + b*isl + c*osl. Features x=[1, isl, osl]."""
    ATA = [[0.0] * 3 for _ in range(3)]
    ATb = [0.0] * 3
    for r in rows:
        x = [1.0, float(r["isl"]), float(r["osl"])]
        for i in range(3):
            ATb[i] += x[i] * r["dur"]
            for j in range(3):
                ATA[i][j] += x[i] * x[j]
    a, b, c = _solve3(ATA, ATb)
    return {"ttft_a": a, "ttft_b": b, "itl": c}


def invert_duration(dur, isl, model):
    """Given a measured duration + ISL, back out OSL = (dur - a - b*isl)/itl."""
    osl = (dur - model["ttft_a"] - model["ttft_b"] * isl) / model["itl"]
    return max(1, int(round(osl)))


def load_tau2(path):
    """Load per-turn Tau2 records: usage.prompt_tokens/completion_tokens + generation_time_seconds."""
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        u = d.get("usage", {})
        if "prompt_tokens" in u and "completion_tokens" in u and "generation_time_seconds" in d:
            rows.append({"isl": u["prompt_tokens"], "osl": u["completion_tokens"],
                         "dur": d["generation_time_seconds"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau2", required=True)
    a = ap.parse_args()
    m = fit_time_model(load_tau2(a.tau2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
