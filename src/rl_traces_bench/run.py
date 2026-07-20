"""Replay a Mooncake trace as a static batch (concurrency == #sessions) via aiperf,
then analyze. Endpoint-agnostic: point --url at any OpenAI-compatible server."""
import argparse, json, os, subprocess

def build_aiperf_cmd(trace, url, concurrency, tokenizer, out_dir, model="model",
                     endpoint_type="completions", endpoint="/v1/completions",
                     synth_max_osl=None):
    # Default to the completions endpoint for faithful exact-OSL replay: a chat
    # template can emit a turn-end stop token that ignore_eos (which only suppresses
    # the EOS token id) does not cover, so the server stops early and OSL is not
    # enforced. Completions has no chat template, so ignore_eos forces exactly
    # max_tokens. Override with endpoint_type="chat", endpoint="/v1/chat/completions".
    # --tokenizer-trust-remote-code: required for custom tokenizers.
    # --use-server-token-count: take OSL from the server's usage.completion_tokens
    #   instead of a client-side re-tokenize roundtrip (avoids spurious OSL mismatch).
    cmd = ["aiperf", "profile", "--model", model, "--tokenizer", tokenizer,
           "--tokenizer-trust-remote-code", "--use-server-token-count",
           "--endpoint-type", endpoint_type, "--endpoint", endpoint,
           "--url", url, "--streaming", "--custom-dataset-type", "mooncake_trace",
           "--input-file", trace, "--concurrency", str(concurrency),
           "--export-level", "records", "--output-artifact-dir", out_dir]
    if synth_max_osl is not None:
        cmd += ["--synthesis-max-osl", str(synth_max_osl)]
    cmd += ["--extra-inputs", "ignore_eos:true"]
    return cmd

def find_export(out_dir):
    for root, _dirs, files in os.walk(out_dir):
        if "profile_export.jsonl" in files:
            return os.path.join(root, "profile_export.jsonl")
    return None

def main(argv=None):
    ap = argparse.ArgumentParser(prog="rl-traces run")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--url", default="localhost:8000")
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--tokenizer", required=True,
                    help="served model HF repo id (namespace/name); local paths rejected by aiperf")
    ap.add_argument("--model", default="model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vllm-src", default=None, help="editable vLLM checkout, for provenance stamp")
    ap.add_argument("--endpoint-type", default="completions")
    ap.add_argument("--endpoint", default="/v1/completions")
    ap.add_argument("--synth-max-osl", type=int, default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    subprocess.run(build_aiperf_cmd(a.trace, a.url, a.concurrency, a.tokenizer, a.out, a.model,
                                     endpoint_type=a.endpoint_type, endpoint=a.endpoint,
                                     synth_max_osl=a.synth_max_osl), check=True)
    export = find_export(a.out)
    if not export:
        raise SystemExit(f"no profile_export.jsonl under {a.out}")
    from rl_traces_bench.analyze import (load_profile_export, compute_report,
        validate_token_domain, validate_time_domain)
    from rl_traces_bench.metrics import session_completions
    recs = load_profile_export(export)
    rep = compute_report(recs)
    rep["validate_token"] = validate_token_domain(recs)
    rep["validate_time"] = validate_time_domain(session_completions(recs))
    try:
        from rl_traces_bench.provenance import collect_provenance
        rep["provenance"] = collect_provenance(a.vllm_src)
    except Exception:
        pass
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str))
