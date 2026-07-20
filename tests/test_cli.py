import subprocess, sys, json, os

def test_cli_gen_trace_writes_trace(tmp_path):
    out = tmp_path / "t.jsonl"
    r = subprocess.run([sys.executable, "-m", "rl_traces_bench.cli", "gen-trace",
                        "--num-rollouts", "4", "--seed", "0", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.exists() and (tmp_path / "t.jsonl.stats.json").exists()

def test_cli_unknown_command_errors():
    r = subprocess.run([sys.executable, "-m", "rl_traces_bench.cli", "bogus"],
                       capture_output=True, text=True)
    assert r.returncode != 0
