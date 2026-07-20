"""Thin runner: exec `vllm serve $VLLM_SERVE_ARGS` verbatim; stamp provenance sidecar."""
import argparse, json, os, shlex, subprocess
from rl_traces_bench.config import load_env
from rl_traces_bench.provenance import collect_provenance

def build_serve_cmd(serve_args):
    return ["vllm", "serve", *shlex.split(serve_args)]

def main(argv=None):
    ap = argparse.ArgumentParser(prog="rl-traces serve")
    ap.add_argument("--env", default=".env")
    a = ap.parse_args(argv)
    env = load_env(a.env)
    serve_args = env.get("VLLM_SERVE_ARGS")
    if not serve_args:
        raise SystemExit(f"VLLM_SERVE_ARGS missing in {a.env}")
    prov = collect_provenance(env.get("VLLM_SRC"))
    with open(os.path.join(os.path.dirname(a.env) or ".", "serve_provenance.json"), "w") as f:
        json.dump(prov, f, indent=2)
    print("provenance:", json.dumps(prov))
    subprocess.run(build_serve_cmd(serve_args), check=True)
