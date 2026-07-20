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
    assert "ignore_eos:true" in s and "min_tokens:1" in s

def test_find_export_locates_nested(tmp_path):
    d = tmp_path / "a" / "b"; d.mkdir(parents=True)
    f = d / "profile_export.jsonl"; f.write_text("{}\n")
    assert find_export(str(tmp_path)) == str(f)
