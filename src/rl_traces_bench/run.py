"""Replay a Mooncake trace as a static batch (concurrency == #sessions) via aiperf,
then analyze. Endpoint-agnostic: point --url at any OpenAI-compatible server."""
import argparse, json, os, shutil, subprocess

def build_aiperf_cmd(trace, url, concurrency, tokenizer, out_dir, model="model",
                     endpoint_type="chat", endpoint="/v1/chat/completions",
                     synth_max_osl=None):
    # Default to the chat endpoint: it's required for multi-turn replay. Only chat
    # accumulates the conversation itself, appending each turn's assistant response
    # to the context before sending the next turn — completions has no such notion
    # of a conversation, so it only ever generates the FIRST turn of a mooncake
    # trace's rollout, silently collapsing the rest of the tail. Exact per-turn OSL
    # is still enforced without a chat-template stop token cutting generation short:
    # ignore_eos (below) suppresses the EOS token id, --use-server-token-count reads
    # OSL from the server's own usage.completion_tokens (no client re-tokenize
    # mismatch), and the tokenizer is the actual served one, not an approximation —
    # so max_tokens is reached exactly, with no min_tokens padding required. Override
    # with endpoint_type="completions", endpoint="/v1/completions" for single-turn use.
    # --tokenizer-trust-remote-code: required for custom tokenizers.
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

def _check_aiperf_available():
    return shutil.which("aiperf") is not None


def find_export(out_dir):
    for root, _dirs, files in os.walk(out_dir):
        if "profile_export.jsonl" in files:
            return os.path.join(root, "profile_export.jsonl")
    return None

def assemble_report(export, targets=None, vllm_src=None):
    """Build the same report shape `analyze.main` writes: the compute_report
    core metrics, both validation gates, the aiperf native-summary block
    (throughput/faithful/prefix_cache — folded in via load_aiperf_summary), and
    (best-effort) vLLM build provenance. Shared by `run.main` so run-generated
    reports never drift from analyze-generated ones."""
    from rl_traces_bench.analyze import (load_profile_export, compute_report,
        validate_token_domain, validate_time_domain, load_aiperf_summary)
    from rl_traces_bench.metrics import session_completions
    recs = load_profile_export(export)
    rep = compute_report(recs)
    rep["validate_token"] = validate_token_domain(recs, targets=targets)
    rep["validate_time"] = validate_time_domain(session_completions(recs))
    rep["aiperf"] = load_aiperf_summary(export)   # native perf metrics + error gate
    try:
        from rl_traces_bench.provenance import collect_provenance
        rep["provenance"] = collect_provenance(vllm_src)
    except Exception:
        pass
    return rep


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
    ap.add_argument("--endpoint-type", default="chat")
    ap.add_argument("--endpoint", default="/v1/chat/completions")
    ap.add_argument("--synth-max-osl", type=int, default=None)
    ap.add_argument("--distribution", default=None,
                    help="distribution JSON to derive token-domain validation targets from "
                         "(defaults to the packaged example long-tail profile's targets)")
    a = ap.parse_args(argv)
    if not _check_aiperf_available():
        raise SystemExit("aiperf not found on PATH — install with: pip install aiperf"
                          "  (or: pip install 'rl-traces-bench[run]')")
    os.makedirs(a.out, exist_ok=True)
    subprocess.run(build_aiperf_cmd(a.trace, a.url, a.concurrency, a.tokenizer, a.out, a.model,
                                     endpoint_type=a.endpoint_type, endpoint=a.endpoint,
                                     synth_max_osl=a.synth_max_osl), check=True)
    export = find_export(a.out)
    if not export:
        raise SystemExit(f"no profile_export.jsonl under {a.out}")
    from rl_traces_bench.analyze import token_targets_from_distribution
    targets = token_targets_from_distribution(a.distribution) if a.distribution else None
    rep = assemble_report(export, targets=targets, vllm_src=a.vllm_src)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=2, default=str)
    from rl_traces_bench.report_html import render_report
    with open(os.path.join(a.out, "report.html"), "w") as f:
        f.write(render_report(rep))
    print(json.dumps(rep, indent=2, default=str))
