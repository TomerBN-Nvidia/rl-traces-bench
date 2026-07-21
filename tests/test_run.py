from rl_traces_bench.run import build_aiperf_cmd, find_export
from rl_traces_bench import run as run_mod
import json
import os
import pytest

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

def test_build_aiperf_cmd_defaults_to_chat_endpoint():
    # chat is required for multi-turn replay: only chat accumulates the
    # conversation itself turn over turn; completions would only ever
    # generate the first turn of a mooncake rollout.
    cmd = build_aiperf_cmd("t.jsonl", "localhost:8000", 512, "org/model", "/out", model="m")
    assert "--endpoint-type" in cmd
    assert cmd[cmd.index("--endpoint-type") + 1] == "chat"
    assert "--endpoint" in cmd
    assert cmd[cmd.index("--endpoint") + 1] == "/v1/chat/completions"

def test_build_aiperf_cmd_endpoint_override_to_completions():
    cmd = build_aiperf_cmd("t.jsonl", "localhost:8000", 512, "org/model", "/out", model="m",
                            endpoint_type="completions", endpoint="/v1/completions")
    assert cmd[cmd.index("--endpoint-type") + 1] == "completions"
    assert cmd[cmd.index("--endpoint") + 1] == "/v1/completions"
    assert "chat" not in cmd
    assert "/v1/chat/completions" not in cmd

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

def test_main_raises_systemexit_with_helpful_message_when_aiperf_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(run_mod.shutil, "which", lambda name: None)
    trace = tmp_path / "t.jsonl"; trace.write_text('{}\n')
    argv = ["--trace", str(trace), "--concurrency", "1", "--tokenizer", "org/m",
            "--out", str(tmp_path / "out")]
    with pytest.raises(SystemExit) as exc:
        run_mod.main(argv)
    msg = str(exc.value)
    assert "aiperf" in msg
    assert "pip install aiperf" in msg
    assert "rl-traces-bench[run]" in msg

def test_check_aiperf_available_reflects_shutil_which(monkeypatch):
    monkeypatch.setattr(run_mod.shutil, "which", lambda name: "/usr/bin/aiperf")
    assert run_mod._check_aiperf_available() is True
    monkeypatch.setattr(run_mod.shutil, "which", lambda name: None)
    assert run_mod._check_aiperf_available() is False


def test_assemble_report_includes_aiperf_summary_and_validation(tmp_path):
    # A small profile_export.jsonl (two sessions, one turn each) plus a sibling
    # profile_export_aiperf.json — same pairing `analyze.main` relies on via
    # load_aiperf_summary. assemble_report must fold BOTH the aiperf block
    # (throughput/faithful/prefix_cache) AND the token/time validation blocks
    # into the same report `run.main` writes, matching analyze's report shape.
    export_dir = tmp_path / "artifacts"
    export_dir.mkdir()
    export = export_dir / "profile_export.jsonl"
    records = [
        {"metadata": {"conversation_id": "0", "turn_index": 0,
                       "request_start_ns": 1_000_000_000_000_000_000,
                       "request_end_ns": 1_000_000_004_000_000_000,
                       "benchmark_phase": "profiling"},
         "metrics": {"input_sequence_length": {"value": 500, "unit": "tokens"},
                     "output_sequence_length": {"value": 654, "unit": "tokens"},
                     "time_to_first_token": {"value": 200.0, "unit": "ms"}}},
        {"metadata": {"conversation_id": "1", "turn_index": 0,
                       "request_start_ns": 1_000_000_000_000_000_000,
                       "request_end_ns": 1_000_000_010_000_000_000,
                       "benchmark_phase": "profiling"},
         "metrics": {"input_sequence_length": {"value": 500, "unit": "tokens"},
                     "output_sequence_length": {"value": 57067, "unit": "tokens"},
                     "time_to_first_token": {"value": 250.0, "unit": "ms"}}},
    ]
    export.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    summary = {
        "aiperf_version": "0.11.0",
        "error_request_count": {"avg": 0},
        "request_count": {"avg": 2},
        "output_token_throughput": {"avg": 1234.5},
        "request_throughput": {"avg": 6.7},
        "time_to_first_token": {"avg": 200.0, "p50": 200.0, "p90": 250.0, "p99": 250.0},
        "gpu_cache_hit_rate": {"avg": 0.42},
    }
    (export_dir / "profile_export_aiperf.json").write_text(json.dumps(summary))

    rep = run_mod.assemble_report(str(export))

    assert rep["aiperf"] is not None
    assert rep["aiperf"]["faithful"] is True
    assert rep["aiperf"]["prefix_cache"] == {"gpu_cache_hit_rate": 0.42}
    assert "validate_token" in rep and "checks" in rep["validate_token"]
    assert "validate_time" in rep and "ratios" in rep["validate_time"]


def test_main_writes_report_html_beside_json(tmp_path, monkeypatch):
    # run.main should emit the interactive report.html next to report.json, so a
    # completed `run` is browsable with no separate `analyze --out-html` step.
    out = tmp_path / "results"
    export_dir = out / "artifacts"
    export_dir.mkdir(parents=True)
    export = export_dir / "profile_export.jsonl"
    export.write_text(json.dumps({
        "metadata": {"conversation_id": "0", "turn_index": 0,
                     "request_start_ns": 1_000_000_000_000_000_000,
                     "request_end_ns": 1_000_000_004_000_000_000,
                     "benchmark_phase": "profiling"},
        "metrics": {"input_sequence_length": {"value": 500, "unit": "tokens"},
                    "output_sequence_length": {"value": 654, "unit": "tokens"}}}) + "\n")
    monkeypatch.setattr(run_mod, "_check_aiperf_available", lambda: True)
    monkeypatch.setattr(run_mod.subprocess, "run", lambda *a, **k: None)  # don't shell out to aiperf
    run_mod.main(["--trace", "t.jsonl", "--url", "localhost:8000",
                  "--concurrency", "1", "--tokenizer", "gpt2", "--out", str(out)])
    html = (out / "report.html").read_text()
    assert (out / "report.json").exists()
    assert html.startswith("<!doctype html>") and "Completion-time CDF" in html
