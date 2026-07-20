from rl_traces_bench.provenance import collect_provenance, git_sha
def test_collect_provenance_no_src():
    p = collect_provenance(None)
    assert set(p) == {"vllm_version", "vllm_src", "git_sha", "dirty"}
    assert p["vllm_src"] is None
def test_git_sha_of_this_repo():
    assert git_sha(".") is None or len(git_sha(".")) == 40
