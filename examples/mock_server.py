#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat endpoint for a GPU-free smoke of the
trace->aiperf->analyze pipeline.

NOT a load/perf server. Its job is to let `aiperf profile` run the full
mooncake-trace replay with no GPU, so we can validate the
trace -> aiperf -> analyze plumbing and (optionally) demonstrate an A/B compare.

Two modes:

* Default (MOCK_MS_PER_TOKEN unset/0) — streams up to MOCK_MAX_TOKENS visible
  tokens per request, instantly. Fast plumbing smoke; the reported OSL is capped.

* Latency model (MOCK_MS_PER_TOKEN > 0) — reports the *requested* OSL faithfully
  (so the long tail is preserved) and sleeps a simulated decode time
      sim_s = requested_osl * (MOCK_MS_PER_TOKEN / 1000) / MOCK_ACCEPT_LEN
  before finishing. MOCK_ACCEPT_LEN stands in for a speculative-decoding mean
  accepted-tokens-per-step (e.g. MTP with a forced/synthetic acceptance rate):
  a higher value decodes proportionally faster. This lets you A/B two configs
  (baseline vs "MTP") over the SAME trace with no GPU. It is a MODEL of decode
  time, not a real engine — it does not reproduce prefill/KV/batching effects.

Usage: python3 examples/mock_server.py [PORT]   (default 8000)
Env:   MOCK_MAX_TOKENS   visible streamed tokens cap (default 32)
       MOCK_MS_PER_TOKEN base ms per output token; 0 disables the latency model
       MOCK_ACCEPT_LEN   mean accepted tokens/step (decode speedup); default 1
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CAP = int(os.environ.get("MOCK_MAX_TOKENS", "32"))
MS_PER_TOKEN = float(os.environ.get("MOCK_MS_PER_TOKEN", "0"))
ACCEPT_LEN = max(float(os.environ.get("MOCK_ACCEPT_LEN", "1")), 0.01)


def _sim_seconds(requested_osl):
    """Modelled decode time for `requested_osl` output tokens at the configured
    per-token latency and acceptance length. 0 when the latency model is off."""
    if MS_PER_TOKEN <= 0:
        return 0.0
    return requested_osl * (MS_PER_TOKEN / 1000.0) / ACCEPT_LEN


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._json({"object": "list", "data": [{"id": "mock-model", "object": "model"}]})
        else:
            self._json({"status": "ok"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        want = int(body.get("max_completion_tokens") or body.get("max_tokens") or 16)
        want = max(1, want)
        visible = min(want, CAP)                       # actually-streamed content tokens
        # With the latency model on, report the full requested OSL (keeps the long
        # tail faithful for --use-server-token-count); otherwise the legacy capped count.
        osl = want if MS_PER_TOKEN > 0 else visible
        sim_s = _sim_seconds(want)
        cid = "chatcmpl-mock"
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def sse(d):
                self.wfile.write(("data: " + json.dumps(d) + "\n\n").encode())
                self.wfile.flush()

            sse({"id": cid, "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            for _ in range(visible):
                sse({"id": cid, "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]})
            if sim_s:
                time.sleep(sim_s)                      # modelled decode time for the tail
            sse({"id": cid, "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 0, "completion_tokens": osl, "total_tokens": osl}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            if sim_s:
                time.sleep(sim_s)
            self._json({"id": cid, "object": "chat.completion",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "x" * visible},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": osl, "total_tokens": osl}})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
