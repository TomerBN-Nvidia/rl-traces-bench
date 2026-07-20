"""Inverse-CDF OSL sampler calibrated to published RL rollout percentiles.

We do NOT assume a parametric family (the real distribution is bimodal). Instead
we build the quantile function from the measured anchor points and interpolate
between them in log-token space, so realized percentiles match by construction.
"""
import math
import random

# (cumulative_prob, generated_tokens). p90 is estimated (design §10 / task header).
OSL_ANCHORS = [
    (0.00, 1),
    (0.50, 654),
    (0.80, 10_000),
    (0.90, 22_000),
    (0.95, 33_212),
    (0.99, 57_067),
    (1.00, 65_489),
]


class QuantileSampler:
    def __init__(self, anchors):
        self.anchors = sorted(anchors)

    def sample(self, u):
        u = min(max(u, 0.0), 1.0)
        a = self.anchors
        for i in range(1, len(a)):
            q0, v0 = a[i - 1]
            q1, v1 = a[i]
            if u <= q1:
                if q1 == q0:
                    return int(round(v1))
                t = (u - q0) / (q1 - q0)
                # log-linear interpolation between anchor token values
                logv = math.log(v0) + t * (math.log(v1) - math.log(v0))
                return max(1, int(round(math.exp(logv))))
        return int(round(a[-1][1]))

    def sample_n(self, n, rng):
        return [self.sample(rng.random()) for _ in range(n)]


def osl_sampler():
    return QuantileSampler(OSL_ANCHORS)
