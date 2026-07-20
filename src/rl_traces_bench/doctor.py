"""Preflight for the serve/run path: prints a checklist with fix hints."""
import argparse, os, shutil
from rl_traces_bench.config import load_env

def _importable(mod):
    try:
        __import__(mod); return True
    except Exception:
        return False

def run_checks(env):
    src = env.get("VLLM_SRC")
    checks = [
        ("vllm importable", _importable("vllm"),
         "pip install -e $VLLM_SRC  (editable build in your active venv)"),
        ("aiperf present", shutil.which("aiperf") is not None, "pip install aiperf"),
        ("tokenizer set", bool(env.get("TOKENIZER")), "set TOKENIZER=<hf-model-id> in .env"),
        ("vllm_src exists", bool(src) and os.path.isdir(src),
         "set VLLM_SRC=<your vllm checkout> in .env"),
    ]
    return checks

def main(argv=None):
    ap = argparse.ArgumentParser(prog="rl-traces doctor")
    ap.add_argument("--env", default=".env")
    a = ap.parse_args(argv)
    env = load_env(a.env) if os.path.exists(a.env) else {}
    ok_all = True
    for name, ok, hint in run_checks(env):
        print(f"[{'OK ' if ok else 'FAIL'}] {name}" + ("" if ok else f"  -> {hint}"))
        ok_all = ok_all and ok
    return 0 if ok_all else 1
