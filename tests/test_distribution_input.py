import json, hashlib
from rl_traces_bench.distributions import load_distribution, osl_sampler, OSL_ANCHORS, default_distribution_path
from rl_traces_bench.gen_trace import build_trace


def test_example_distribution_reproduces_default_anchors():
    d = load_distribution(default_distribution_path())
    assert [tuple(a) for a in d["osl_anchors"]] == OSL_ANCHORS

def test_build_trace_with_example_distribution_is_byte_identical():
    d = load_distribution(default_distribution_path())
    recs, _ = build_trace(8, 0, 512, "per_turn", 300, 200, 1,
                          d["turn_counts"], anchors=[tuple(a) for a in d["osl_anchors"]])
    blob = "".join(json.dumps(r) + "\n" for r in recs).encode()
    assert hashlib.sha256(blob).hexdigest() == \
        "783b5825a3305960036b0dbaf8f480056f233e86992fedc489be25c51207f8e9"

def test_packaged_and_example_distribution_files_are_byte_identical():
    packaged = default_distribution_path()
    with open(packaged, "rb") as f:
        packaged_bytes = f.read()
    with open("examples/distributions/example_longtail.json", "rb") as f:
        example_bytes = f.read()
    assert packaged_bytes == example_bytes
