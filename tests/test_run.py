from rl_traces_bench.run import build_aiperf_cmd, find_export
import os

def test_build_aiperf_cmd_has_required_flags():
    cmd = build_aiperf_cmd("t.jsonl", "localhost:8000", 512, "org/model", "/out", model="m")
    s = " ".join(cmd)
    assert cmd[0] == "aiperf" and "profile" in cmd
    assert "--url" in cmd and "localhost:8000" in cmd
    assert "--custom-dataset-type" in cmd and "mooncake_trace" in cmd
    assert "--concurrency" in cmd and "512" in cmd
    assert "--tokenizer" in cmd and "org/model" in cmd
    assert "ignore_eos:true" in s
    assert "min_tokens:1" not in s
    assert "--tokenizer-trust-remote-code" in cmd
    assert "--use-server-token-count" in cmd

def test_build_aiperf_cmd_defaults_to_completions_endpoint():
    cmd = build_aiperf_cmd("t.jsonl", "localhost:8000", 512, "org/model", "/out", model="m")
    assert "--endpoint-type" in cmd
    assert cmd[cmd.index("--endpoint-type") + 1] == "completions"
    assert "--endpoint" in cmd
    assert cmd[cmd.index("--endpoint") + 1] == "/v1/completions"
    assert "chat" not in cmd
    assert "/v1/chat/completions" not in cmd

def test_build_aiperf_cmd_endpoint_override_to_chat():
    cmd = build_aiperf_cmd("t.jsonl", "localhost:8000", 512, "org/model", "/out", model="m",
                            endpoint_type="chat", endpoint="/v1/chat/completions")
    assert cmd[cmd.index("--endpoint-type") + 1] == "chat"
    assert cmd[cmd.index("--endpoint") + 1] == "/v1/chat/completions"

def test_build_aiperf_cmd_synth_max_osl_included_when_set():
    cmd = build_aiperf_cmd("t.jsonl", "localhost:8000", 512, "org/model", "/out", model="m",
                            synth_max_osl=1024)
    assert "--synthesis-max-osl" in cmd
    assert cmd[cmd.index("--synthesis-max-osl") + 1] == "1024"

def test_build_aiperf_cmd_synth_max_osl_omitted_by_default():
    cmd = build_aiperf_cmd("t.jsonl", "localhost:8000", 512, "org/model", "/out", model="m")
    assert "--synthesis-max-osl" not in cmd

def test_find_export_locates_nested(tmp_path):
    d = tmp_path / "a" / "b"; d.mkdir(parents=True)
    f = d / "profile_export.jsonl"; f.write_text("{}\n")
    assert find_export(str(tmp_path)) == str(f)
