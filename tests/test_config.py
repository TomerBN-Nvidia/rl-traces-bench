from rl_traces_bench.config import load_env
def test_load_env_flat(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# comment\nVLLM_SRC=/x/vllm\nVLLM_SERVE_ARGS=org/m --tensor-parallel-size 4 --port 8000\nURL=localhost:8000\n")
    env = load_env(str(p))
    assert env["VLLM_SRC"] == "/x/vllm"
    assert env["VLLM_SERVE_ARGS"].endswith("--port 8000")
    assert "comment" not in env
