"""Sanitized CPU SSE fixture, never a model performance result."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import hashlib
import secrets
import urllib.request
from urllib.parse import urlsplit


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_GET(self):
        if self.path == "/infrx-fixture":
            raw=json.dumps({"proof":hashlib.sha256(self.server.token.encode()).hexdigest()}).encode()
        elif self.path == "/v1/models":
            raw = json.dumps({"data":[{"id":"fixture/model"}]}).encode()
        else: raw = b"fixture_requests_waiting 0\nfixture_prefix_cache_hits_total 1\n"
        self.send_response(200); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock: self.server.bodies.append(body)
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        content = body["messages"][-1]["content"]
        result = self.server.answers.get(content, "fixture-answer")
        if self.server.fail_quality: result = "wrong"
        time.sleep(self.server.delay)
        usage = {"prompt_tokens":sum(len(x["content"].split()) for x in body["messages"]), "completion_tokens":min(2,body["max_tokens"])}
        packets = [{"model":body["model"], "choices":[{"delta":{"content":result}}]},
            {"model":body["model"], "choices":[{"delta":{},"finish_reason":"stop"}], "usage":usage}]
        try:
            for packet in packets: self.wfile.write(b"data: "+json.dumps(packet).encode()+b"\n\n"); self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): pass


@contextmanager
def server(delay=.002, fail_quality=False):
    srv = ThreadingHTTPServer(("127.0.0.1",0),Handler); srv.daemon_threads = True
    cases = json.loads((Path(__file__).parent/"cases.json").read_text())
    srv.answers = {c["prompt"]:json.dumps(c["expected"]) for c in cases}
    srv.bodies=[]; srv.lock=threading.Lock(); srv.delay=delay; srv.fail_quality=fail_quality
    srv.token=secrets.token_hex(32)
    thread = threading.Thread(target=srv.serve_forever, daemon=True); thread.start()
    try: yield srv, f"http://127.0.0.1:{srv.server_port}"
    finally: srv.shutdown(); srv.server_close(); thread.join()


def verify(endpoints,token):
    if not token: raise ValueError("CPU mock requires the controlled in-process fixture; use rehearse")
    for base in endpoints:
        if urlsplit(base).hostname != "127.0.0.1": raise ValueError("CPU fixture is loopback only")
        with urllib.request.urlopen(base+"/infrx-fixture",timeout=1) as response:
            proof=json.loads(response.read(4096)).get("proof")
        if proof!=hashlib.sha256(token.encode()).hexdigest(): raise ValueError("Fixture proof mismatch")
