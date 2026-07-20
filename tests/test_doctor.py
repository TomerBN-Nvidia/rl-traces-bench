from rl_traces_bench.doctor import run_checks
def test_run_checks_reports_each_dimension():
    checks = run_checks({"VLLM_SRC": "/nope/vllm", "TOKENIZER": "org/m", "URL": "localhost:8000"})
    names = {c[0] for c in checks}
    assert {"vllm importable", "aiperf present", "tokenizer set", "vllm_src exists"} <= names
    assert all(len(c) == 3 for c in checks)
