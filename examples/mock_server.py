#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat endpoint for a GPU-free smoke of the
trace->aiperf->analyze pipeline.

NOT a load/perf server. It streams up to MOCK_MAX_TOKENS tokens per request
(capped so a tail straggler in the trace doesn't make the smoke slow). Its only
job is to let `aiperf profile` run the full mooncake-trace replay with no GPU, so
we can validate the trace -> aiperf -> analyze plumbing and discover aiperf's real
per-request `profile_export` field names.

Usage: python3 examples/mock_server.py [PORT]   (default 8000)
Env:   MOCK_MAX_TOKENS (default 32)
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CAP = int(os.environ.get("MOCK_MAX_TOKENS", "32"))


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
        want = body.get("max_completion_tokens") or body.get("max_tokens") or 16
        ntok = max(1, min(int(want), CAP))
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
            for _ in range(ntok):
                sse({"id": cid, "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]})
            sse({"id": cid, "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 0, "completion_tokens": ntok, "total_tokens": ntok}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            self._json({"id": cid, "object": "chat.completion",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "x" * ntok},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": ntok, "total_tokens": ntok}})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
