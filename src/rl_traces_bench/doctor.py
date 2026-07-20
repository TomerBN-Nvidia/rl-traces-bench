"""Preflight for the serve/run path: prints a checklist with fix hints.

Checks are built by intent, not by hard-requiring every possible dependency:
- always: aiperf on PATH, TOKENIZER set.
- if URL is set: probe the endpoint (works for the client-only path, where
  you point rl-traces at someone else's already-running server).
- if VLLM_SERVE_ARGS is set (i.e. you intend to `rl-traces serve`): vllm
  importable + VLLM_SRC pointing at a real directory.
"""
import argparse, os, shutil, urllib.request, urllib.error
from rl_traces_bench.config import load_env


def _importable(mod):
    try:
        __import__(mod); return True
    except Exception:
        return False


def _endpoint_reachable(url):
    """Best-effort reachability probe against <url>/v1/models. Any HTTP
    response (even a non-2xx one) counts as reachable; connection failures /
    timeouts do not."""
    u = url
    if u.startswith("http://"):
        u = u[len("http://"):]
    elif u.startswith("https://"):
        u = u[len("https://"):]
    target = "http://" + u.rstrip("/") + "/v1/models"
    try:
        urllib.request.urlopen(target, timeout=3)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def run_checks(env):
    checks = [
        ("aiperf present", shutil.which("aiperf") is not None, "pip install aiperf"),
        ("tokenizer set", bool(env.get("TOKENIZER")), "set TOKENIZER=<hf-model-id> in .env"),
    ]
    if env.get("URL"):
        checks.append(("endpoint reachable", _endpoint_reachable(env["URL"]),
                       "start your server or fix URL in .env"))
    if env.get("VLLM_SERVE_ARGS"):
        src = env.get("VLLM_SRC", "")
        checks.append(("vllm importable", _importable("vllm"),
                       "pip install -e $VLLM_SRC  (editable build in your active venv)"))
        checks.append(("vllm_src exists", bool(src) and os.path.isdir(src),
                       "set VLLM_SRC=<your vllm checkout> in .env"))
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
