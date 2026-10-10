#!/usr/bin/env python3
"""Bounded stdlib OpenAI SSE load driver with append-only request evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import datetime as dt
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import socket
import statistics
import subprocess
import sys
import tarfile
import threading
import time
from urllib.parse import urlsplit
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
MAX_RESPONSE = 32 * 1024 * 1024


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def driver_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def resume_cell(root, cfg, cases=None, corpus=None):
    """Retain the latest complete logical-cell attempt with identical inputs."""
    root = Path(root)
    attempts = ([root] if root.exists() else []) + sorted(root.parent.glob(root.name + "-retry-*"))
    completed = []
    for attempt in attempts:
        if not attempt.is_dir():
            continue
        manifest = json.loads((attempt / "manifest.json").read_text())
        old_cfg = {k: v for k, v in manifest["config"].items() if k != "run_nonce"}
        identity = (manifest.get("driver_sha256"), manifest.get("cases_sha256"), manifest.get("corpus_sha256"))
        expected = (driver_hash(), digest(cases) if cases else None, digest(corpus) if corpus else None)
        if old_cfg != cfg or identity != expected:
            raise ValueError("Resume configuration, corpus, cases or driver changed; use a new output directory")
        previous = summarize(attempt)
        if previous["complete"] and not previous["malformed_journal_lines"] and not previous["counts"]["unfinished"]:
            ended = json.loads((attempt / "completion.json").read_text())["end_at"]
            completed.append((ended, str(attempt), attempt, previous))
    if completed:
        _, _, attempt, previous = max(completed)
        return attempt, previous
    return (root.with_name(root.name + "-retry-" + uuid.uuid4().hex[:8]) if attempts else root), None


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def endpoint(value):
    u = urlsplit(value.rstrip("/"))
    if u.scheme not in ("http", "https") or not u.hostname or u.username or u.password or u.query or u.fragment or u.path not in ("", "/"):
        raise ValueError("Endpoint must be an HTTP(S) origin without credentials, query or path")
    return value.rstrip("/")


def headers():
    key = os.environ.get("BENCH_API_KEY")
    return {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}


def wait_ready(endpoints, model, timeout):
    deadline = time.monotonic() + timeout
    ready = set()
    while time.monotonic() < deadline:
        for base in endpoints:
            req = urllib.request.Request(endpoint(base) + "/v1/models", headers=headers())
            try:
                with urllib.request.urlopen(req, timeout=min(3, max(.01, deadline - time.monotonic()))) as r:
                    data = json.loads(r.read(1024 * 1024))
                if model in {x.get("id") for x in data.get("data", [])}:
                    ready.add(base)
            except (OSError, ValueError):
                pass
        if len(ready) == len(endpoints):
            return
        time.sleep(min(.2, max(0, deadline - time.monotonic())))
    raise ValueError("Readiness/served model identity deadline exceeded")


def number(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def request(base, model, messages, output, timeout, scheduled, attempt, body_extra=None, retain=False, deadline_epoch=None):
    started = time.time()
    if deadline_epoch is not None:
        timeout = min(timeout, max(.001, deadline_epoch-started))
    clock = time.monotonic()
    deadline = clock + timeout
    record = {"attempt_id": attempt, "scheduled_at": scheduled, "send_at": started,
              "first_generated_at": None, "first_content_at": None, "last_generated_at": None,
              "end_at": None, "model_requested": model, "models_observed": [], "endpoint": base,
              "prompt_sha256": digest(messages), "requested_max_output": output, "status": "failed",
              "http_status": None, "error": None, "finish_reason": None, "usage": None,
              "chunk_intervals_s": [], "generated_events_at": [], "retry": 0, "done": False}
    u = urlsplit(endpoint(base))
    conn = (http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection)(u.hostname, u.port, timeout=min(timeout, 15))
    body = {"model": model, "messages": messages, "max_tokens": output, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}, **(body_extra or {})}
    forbidden = {"model", "messages", "stream", "max_tokens", "stream_options"} & (body_extra or {}).keys()
    if forbidden:
        raise ValueError("Extra body cannot override identity, bounded output or streaming/accounting")
    chunks, reasoning, tools = [], [], []
    models = set()
    identity_missing = False
    observed_bytes = 0
    timed_out = threading.Event()
    sockets = []
    def abort():
        timed_out.set()
        for active in [conn.sock, *sockets]:
            if active:
                try: active.shutdown(socket.SHUT_RDWR)
                except OSError: pass
    timer = threading.Timer(timeout, abort)
    timer.daemon = True
    timer.start()
    try:
        if deadline_epoch is not None and started >= deadline_epoch:
            raise TimeoutError("absolute_request_deadline")
        conn.request("POST", "/v1/chat/completions", json.dumps(body), headers())
        if conn.sock: conn.sock.settimeout(max(.001, deadline - time.monotonic()))
        response = conn.getresponse()
        sock = conn.sock or getattr(getattr(response.fp, "raw", None), "_sock", None)
        sockets.append(sock)
        record["http_status"] = response.status
        if response.status != 200:
            record["error"] = "http_" + str(response.status)
            return record
        if "text/event-stream" not in response.getheader("Content-Type", ""):
            raise ValueError("non_sse_response")
        event_lines = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("request_deadline")
            if sock:
                sock.settimeout(min(remaining, 30))
            line = response.readline(65537)
            observed_bytes += len(line)
            if len(line) > 65536 or observed_bytes > MAX_RESPONSE:
                raise ValueError("response_byte_limit")
            if not line:
                break
            line = line.rstrip(b"\r\n")
            if line.startswith(b"data:"):
                event_lines.append(line[5:].lstrip())
                continue
            if line or not event_lines:
                continue
            payload = b"\n".join(event_lines)
            event_lines = []
            if payload == b"[DONE]":
                record["done"] = True
                break
            packet = json.loads(payload)
            if packet.get("error"):
                raise ValueError("stream_error")
            seen = packet.get("model")
            if seen is not None and not isinstance(seen, str):
                raise ValueError("response_identity_missing_or_mismatched")
            if seen:
                models.add(seen)
            usage = packet.get("usage")
            if usage is not None:
                record["usage"] = usage
            choices = packet.get("choices", [])
            if len(choices) > 1:
                raise ValueError("unexpected_multiple_choices")
            for choice in choices:
                if choice.get("finish_reason") is not None:
                    record["finish_reason"] = choice["finish_reason"]
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                tool = delta.get("tool_calls") or []
                if not isinstance(text, str) or not isinstance(thought, str):
                    raise ValueError("unsupported_content_shape")
                chunks.append(text); reasoning.append(thought); tools.extend(tool)
                if text or thought or tool:
                    now = started + time.monotonic() - clock
                    identity_missing |= seen is None
                    if record["first_generated_at"] is None:
                        record["first_generated_at"] = now
                    if text and record["first_content_at"] is None:
                        record["first_content_at"] = now
                    if record["last_generated_at"] is not None:
                        record["chunk_intervals_s"].append(now - record["last_generated_at"])
                    record["last_generated_at"] = now
                    record["generated_events_at"].append(now)
        usage = record["usage"]
        if models != {model} or identity_missing:
            raise ValueError("response_identity_missing_or_mismatched")
        if not record["done"] or record["finish_reason"] is None:
            raise ValueError("incomplete_stream")
        if not isinstance(usage, dict) or not number(usage.get("prompt_tokens")) or not number(usage.get("completion_tokens")):
            raise ValueError("missing_or_invalid_token_usage")
        if usage["completion_tokens"] > output:
            raise ValueError("output_limit_exceeded")
        if not record["generated_events_at"] or usage["completion_tokens"] < 1:
            raise ValueError("empty_generation")
        record["status"] = "success"
    except (TimeoutError, socket.timeout):
        record["error"] = "request_timeout"
    except (OSError, http.client.HTTPException) as exc:
        record["error"] = "request_timeout" if timed_out.is_set() else "transport_" + type(exc).__name__
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        record["error"] = str(exc) if str(exc) in {"non_sse_response", "response_byte_limit", "stream_error", "unexpected_multiple_choices", "unsupported_content_shape", "response_identity_missing_or_mismatched", "incomplete_stream", "missing_or_invalid_token_usage", "empty_generation", "output_limit_exceeded"} else "malformed_stream"
    finally:
        timer.cancel()
        conn.close()
        if timed_out.is_set(): record["status"] = "failed"; record["error"] = "request_timeout"
        record["end_at"] = started + time.monotonic() - clock
        record["models_observed"] = sorted(models)
        record["output_sha256"] = digest({"content": chunks, "reasoning": reasoning, "tools": tools})
        if retain:
            record["content"] = "".join(chunks)
            record["reasoning_content"] = "".join(reasoning)
            record["tool_calls"] = tools
    return record


def synthetic(index, chars, seed):
    # Unique prefix at the beginning avoids accidental shared warm-prefix capacity.
    rng = random.Random(seed + index)
    prefix = "/* independent request " + str(uuid.UUID(int=rng.getrandbits(128))) + " */\n"
    line = "// Review bounded queue scheduling, timeout handling, cache invalidation and deterministic tests.\n"
    text = (prefix + line * (chars // len(line) + 1))[:chars]
    return [{"role": "user", "content": text + "\nWrite a long numbered list of concrete code review checks. Continue until the output budget. Avoid repeating the prompt."}]


class Telemetry:
    def __init__(self, root, endpoints, interval=5, deadline_epoch=None):
        self.root, self.endpoints, self.interval = root, endpoints, interval
        self.deadline_epoch = deadline_epoch
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        with (self.root / "telemetry.jsonl").open("x", buffering=1) as f:
            while not self.stop.is_set():
                if self.deadline_epoch is not None and time.time() >= self.deadline_epoch: break
                data = {"utc": utc(), "at": time.time(), "gpu": None, "proc": {}, "metrics": {}}
                try:
                    r = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,temperature.gpu,pstate,ecc.errors.uncorrected.volatile.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=2)
                    data["gpu"] = {"exit": r.returncode, "csv": r.stdout[:65536]}
                except (OSError, subprocess.TimeoutExpired) as exc:
                    data["gpu"] = {"error": type(exc).__name__}
                for name in ("stat", "meminfo", "vmstat", "net/dev", "diskstats", "pressure/cpu", "pressure/memory", "pressure/io"):
                    p = Path("/proc") / name
                    if p.exists(): data["proc"][name] = p.read_text()[:65536]
                for base in self.endpoints:
                    if self.deadline_epoch is not None and time.time() >= self.deadline_epoch: break
                    try:
                        req = urllib.request.Request(base + "/metrics", headers=headers())
                        limit=min(1,max(.001,self.deadline_epoch-time.time())) if self.deadline_epoch else 1
                        with urllib.request.urlopen(req, timeout=limit) as response:
                            raw = response.read(1024 * 1024).decode("utf-8", errors="replace")
                        # Raw Prometheus evidence from the explicitly selected test endpoint.
                        data["metrics"][base] = raw
                    except (OSError, ValueError) as exc:
                        data["metrics"][base] = {"error": type(exc).__name__}
                f.write(json.dumps(data) + "\n")
                self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *args):
        self.stop.set(); self.thread.join(timeout=5 + len(self.endpoints))


def validate(cfg):
    if not 1 <= cfg["concurrency"] <= 256 or not 1 <= cfg["requests"] <= 10000:
        raise ValueError("Concurrency/request cap outside allowed bounds")
    if not 0 < cfg["duration"] <= 3600 or not 0 < cfg["timeout"] <= 600 or not 1 <= cfg["output"] <= 8192:
        raise ValueError("Duration, timeout or output cap outside allowed bounds")
    if cfg.get("rate") is not None and not 0 < cfg["rate"] <= 1000:
        raise ValueError("Arrival rate must be finite, positive and <=1000/s")
    if not 128 <= cfg["prompt_chars"] <= 256000:
        raise ValueError("Synthetic prompt character cap outside allowed bounds")
    if not math.isfinite(cfg["duration"]) or not math.isfinite(cfg["timeout"]):
        raise ValueError("Finite time caps required")
    if not all(math.isfinite(cfg[x]) and cfg[x] > 0 for x in ("ttft_slo", "e2e_slo")):
        raise ValueError("Explicit finite positive SLOs required")
    extra = cfg.get("body_extra", {})
    if not isinstance(extra, dict) or {"model", "messages", "stream", "max_tokens", "stream_options"} & extra.keys():
        raise ValueError("Extra body cannot override bounded request identity/accounting")
    for value in cfg["endpoints"]: endpoint(value)


def run_cell(cfg, root, cases=None, corpus=None):
    validate(cfg)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    cfg = {**cfg, "run_nonce": cfg.get("run_nonce") or uuid.uuid4().hex}
    start_wall, start = time.time(), time.monotonic()
    manifest = {"schema": 1, "id": root.name, "utc": utc(), "config": cfg, "config_sha256": digest(cfg),
                "driver_sha256": driver_hash(),
                "python": sys.version, "clock": "epoch timestamps derived from monotonic intervals; host clock synchronization required",
                "prompt_length_protocol": "corpus tokenization provenance" if corpus else "characters only; use actual server prompt_tokens for bins",
                "cases_sha256": digest(cases) if cases else None, "corpus_sha256": digest(corpus) if corpus else None,
                "profiled": cfg.get("profiled", False), "deployment": cfg.get("deployment"),
                "tail_protocol": "nearest-rank; p95 >=60 successes, p99 >=300 successes; exploratory sample-count rule, not confidence intervals",
                "output_protocol": "natural completion unless body_extra explicitly forces ignore_eos"}
    write_json(root / "manifest.json", manifest)
    if cases: write_json(root / "cases.json", cases)
    if corpus: write_json(root / "corpus.json", corpus)
    futures = set()
    rng = random.Random(cfg["seed"])
    arrivals = []
    if cfg.get("rate") is not None:
        offset = 0.
        while len(arrivals) < cfg["requests"] and offset < cfg["duration"]:
            arrivals.append(start + offset)
            offset += rng.expovariate(cfg["rate"])
    admitted = 0
    interrupted = False
    with (root / "attempts.jsonl").open("x", buffering=1) as journal, (root / "scheduled.jsonl").open("x", buffering=1) as scheduled_log, ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool, Telemetry(root, cfg["endpoints"], cfg.get("telemetry_interval", 5), cfg.get("deadline_epoch")):
        if arrivals:
            # Persist the whole bounded offered schedule before admission can lag or stop.
            for idx, scheduled in enumerate(arrivals):
                scheduled_log.write(json.dumps({"attempt_id": root.name + ":" + str(idx), "scheduled_at": start_wall + scheduled - start}) + "\n")
            scheduled_log.flush()
            os.fsync(scheduled_log.fileno())
        def flush(done):
            if not done: return
            for future in done:
                row = future.result()
                journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
        try:
            while admitted < (len(arrivals) if cfg.get("rate") is not None else cfg["requests"]):
                now = time.monotonic()
                absolute_expired = cfg.get("deadline_epoch") is not None and time.time() >= cfg["deadline_epoch"]
                if cfg.get("rate") is None and (now >= start + cfg["duration"] or absolute_expired):
                    break
                done = {x for x in futures if x.done()}
                flush(done); futures -= done
                if cfg.get("rate") is None:
                    if len(futures) >= cfg["concurrency"]:
                        done, _ = wait(futures, timeout=.05, return_when=FIRST_COMPLETED)
                        flush(done); futures -= done
                        continue
                    scheduled = time.monotonic()
                else:
                    scheduled = arrivals[admitted]
                    if now < scheduled and not absolute_expired:
                        time.sleep(min(.01, scheduled - now)); continue
                idx = admitted
                admitted += 1
                attempt = root.name + ":" + str(idx)
                scheduled_epoch = start_wall + scheduled - start
                if cfg.get("rate") is None:
                    scheduled_log.write(json.dumps({"attempt_id": attempt, "scheduled_at": scheduled_epoch}) + "\n")
                    scheduled_log.flush()
                    os.fsync(scheduled_log.fileno())
                admitted_epoch = start_wall + time.monotonic() - start
                expired = cfg.get("rate") is not None and (time.monotonic() >= start + cfg["duration"] or absolute_expired)
                if expired or len(futures) >= cfg["concurrency"]:
                    # Keep arrivals independent of response completion. No hidden driver queue.
                    journal.write(json.dumps({"attempt_id": attempt, "scheduled_at": scheduled_epoch, "admitted_at": admitted_epoch, "send_at": None,
                        "end_at": time.time(), "status": "shed", "error": "client_admission_window_elapsed" if expired else "client_inflight_cap", "usage": None, "retry": 0}) + "\n")
                    continue
                case = cases[idx % len(cases)] if cases else None
                entry = corpus[idx % len(corpus)] if corpus else None
                messages = ([{"role": "user", "content": case["prompt"]}] if case else
                            entry["messages"] if entry else synthetic(idx, cfg["prompt_chars"], cfg["seed"] + int(cfg["run_nonce"][:8], 16)))
                if entry and cfg.get("cache_protocol", "distinct-prefix") == "distinct-prefix":
                    nonce = hashlib.sha256((cfg["run_nonce"] + str(idx)).encode()).hexdigest()[:36]
                    messages = [{**m, "content": nonce + "\n" + m["content"]} for m in messages]
                def worker(base, messages, scheduled_epoch, admitted_epoch, attempt, case, entry):
                    row = request(base, cfg["model"], messages, cfg["output"], cfg["timeout"], scheduled_epoch,
                                  attempt, cfg.get("body_extra"), bool(case or (entry and entry.get("rubric"))), cfg.get("deadline_epoch"))
                    row["admitted_at"] = admitted_epoch
                    if entry: row["corpus_id"] = entry["id"]; row["content_tokens"] = entry.get("content_tokens")
                    if entry and entry.get("rubric"):
                        rubric=entry["rubric"];text=row.get("content","").strip()
                        if rubric["kind"]=="exact_text":row["quality_pass"]=text==rubric["expected"]
                        else:
                            try:
                                value=json.loads(text)
                                row["quality_pass"]=value==rubric["expected"] and type(value) is type(rubric["expected"])
                            except ValueError:row["quality_pass"]=False
                    if case:
                        row["task_id"] = case["id"]
                        text = row.get("content", "").strip()
                        if text.startswith("```json") and text.endswith("```"): text = text[7:-3].strip()
                        try:
                            value = json.loads(text)
                            row["quality_pass"] = value == case["expected"] and type(value) is type(case["expected"])
                        except ValueError: row["quality_pass"] = False
                    return row
                futures.add(pool.submit(worker, cfg["endpoints"][idx % len(cfg["endpoints"])], messages, scheduled_epoch, admitted_epoch, attempt, case, entry))
        except KeyboardInterrupt:
            interrupted = True
        finally:
            # Running requests are drained with per-request deadlines; preserve their outcome.
            while futures:
                done, _ = wait(futures, timeout=.1, return_when=FIRST_COMPLETED)
                flush(done); futures -= done
    ended = time.time()
    write_json(root / "completion.json", {"start_at": start_wall, "end_at": ended, "elapsed_s": ended - start_wall,
               "scheduled": len(arrivals) if cfg.get("rate") is not None else admitted, "interrupted": interrupted, "drain_included": True,
               "admission_started_at":start_wall, "admission_ended_at":min(ended,start_wall+cfg["duration"],cfg.get("deadline_epoch",ended)),
               "absolute_deadline_reached":bool(cfg.get("deadline_epoch") and ended>=cfg["deadline_epoch"]),
               "stop_reason": "interrupted" if interrupted else "request_cap" if admitted >= cfg["requests"] else "admission_duration"})
    result = summarize(root)
    write_json(root / "summary.json", result)
    return result


def distribution(values):
    ordered = sorted(x for x in values if x is not None)
    def percentile(p, minimum):
        return ordered[max(0, math.ceil(len(ordered) * p) - 1)] if len(ordered) >= minimum else None
    return {"n": len(ordered), "p50": percentile(.5, 1), "p95": percentile(.95, 60), "p99": percentile(.99, 300),
            "min": min(ordered) if ordered else None, "max": max(ordered) if ordered else None}


def read_rows(path):
    rows, malformed = [], 0
    for line in Path(path).read_text().splitlines():
        try: rows.append(json.loads(line))
        except ValueError: malformed += 1
    return rows, malformed


def summarize(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    cfg = manifest["config"]
    rows, malformed = read_rows(root / "attempts.jsonl")
    scheduled, bad_scheduled = read_rows(root / "scheduled.jsonl") if (root / "scheduled.jsonl").exists() else ([], 0)
    recorded = {r["attempt_id"] for r in rows}
    unfinished = [r for r in scheduled if r["attempt_id"] not in recorded]
    complete = json.loads((root / "completion.json").read_text()) if (root / "completion.json").exists() else None
    start = min((r["scheduled_at"] for r in rows), default=0)
    end = max((r.get("end_at") or start for r in rows), default=start)
    elapsed = complete["elapsed_s"] if complete else end - start
    good = [r for r in rows if r["status"] == "success"]
    def delta(row, a, b):
        return row[a] - row[b] if row.get(a) is not None and row.get(b) is not None else None
    tpot = [(r["last_generated_at"] - r["first_generated_at"]) / (r["usage"]["completion_tokens"] - 1)
            for r in good if r["usage"]["completion_tokens"] > 1 and r.get("last_generated_at") is not None]
    qualified = [r for r in good if (delta(r, "first_generated_at", "scheduled_at") or 0) <= cfg["ttft_slo"]
                 and delta(r, "end_at", "scheduled_at") <= cfg["e2e_slo"] and r.get("quality_pass", True)]
    counts = {status: sum(r["status"] == status for r in rows) for status in ("success", "failed", "shed")}
    input_total = sum(r["usage"]["prompt_tokens"] for r in good)
    output_total = sum(r["usage"]["completion_tokens"] for r in good)
    missing = sum(not isinstance(r.get("usage"), dict) or not number(r.get("usage", {}).get("completion_tokens")) for r in rows if r["status"] != "shed")
    invalid_protocol = [r["attempt_id"] for r in rows if r.get("error") in {"non_sse_response", "response_byte_limit", "unexpected_multiple_choices", "unsupported_content_shape", "response_identity_missing_or_mismatched", "missing_or_invalid_token_usage", "malformed_stream", "output_limit_exceeded"}]
    lag = [max(r.get("send_at") or r.get("admitted_at") or r["scheduled_at"], r.get("admitted_at") or r["scheduled_at"]) - r["scheduled_at"] for r in rows]
    return {"schema": 1, "config_sha256": manifest["config_sha256"], "attempts": len(rows) + len(unfinished), "counts": {**counts, "unfinished": len(unfinished)},
            "complete": complete is not None and not complete["interrupted"], "malformed_journal_lines": malformed,
            "missing_usage_attempts": missing, "elapsed_s_including_drain": elapsed,
            "successful_input_tokens": input_total, "successful_output_tokens": output_total,
            "input_tokens_per_s": input_total / elapsed if elapsed else None,
            "output_tokens_per_s": output_total / elapsed if elapsed else None,
            "requests_per_s": len(good) / elapsed if elapsed else None,
            "goodput_requests_per_s": len(qualified) / elapsed if elapsed else None,
            "goodput_output_tokens_per_s": sum(r["usage"]["completion_tokens"] for r in qualified) / elapsed if elapsed else None,
            "slo": {"ttft_s": cfg["ttft_slo"], "e2e_s": cfg["e2e_slo"], "basis": "scheduled arrival including driver delay"},
            "latency_success_s": distribution([delta(r, "end_at", "scheduled_at") for r in good]),
            "ttft_generated_success_s": distribution([delta(r, "first_generated_at", "scheduled_at") for r in good]),
            "ttft_visible_success_s": distribution([delta(r, "first_content_at", "scheduled_at") for r in good]),
            "tpot_s": distribution(tpot), "chunk_intervals_s": distribution([x for r in good for x in r["chunk_intervals_s"]]),
            "chunk_interval_is_not_token_itl": True, "driver_delay_s": distribution(lag),
            "input_token_distribution": distribution([r["usage"]["prompt_tokens"] for r in good]),
            "output_token_distribution": distribution([r["usage"]["completion_tokens"] for r in good]),
            "quality_diagnostic": {"n": sum("quality_pass" in r for r in rows), "passed": sum(r.get("quality_pass", False) and r["status"] == "success" for r in rows)},
            "valid_capacity_evidence": bool(complete and not complete["interrupted"] and not malformed and not bad_scheduled and not unfinished and not missing and not invalid_protocol and not manifest["profiled"] and not any(x > .1 for x in lag)),
            "invalid_protocol_attempts": invalid_protocol,
            "deployment_attested": bool(manifest.get("deployment")),
            "cache_protocol": cfg.get("cache_protocol", "distinct-prefix"),
            "cache_flush_verified": False,
            "quality_scope": "Small held-fixed coding/instruction diagnostic, not general coding competence or deterministic output equivalence",
            "failed_partial_tokens": "retained in raw usage when observable; excluded from successful throughput, never treated as zero billable cost"}


def archive(root, output, deadline_epoch=None):
    def checkpoint():
        if deadline_epoch is not None and time.time() >= deadline_epoch: raise TimeoutError("Archive absolute deadline reached")
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError("Archive must be a new file outside result directory")
    inventory = []
    for p in sorted(root.rglob("*")):
        checkpoint()
        if p.is_symlink(): raise ValueError("Result archive refuses symlinks")
        if p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""): checkpoint(); h.update(block)
            inventory.append({"path": str(p.relative_to(root)), "size": p.stat().st_size, "sha256": h.hexdigest()})
    with tarfile.open(output, "x:gz") as tar:
        for p in [root,*sorted(root.rglob("*"))]:
            checkpoint(); tar.add(p,arcname=str(Path(root.name)/p.relative_to(root)),recursive=False)
    h = hashlib.sha256()
    with output.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): checkpoint(); h.update(block)
    checkpoint()
    write_json(str(output) + ".manifest.json", {"utc": utc(), "files": inventory, "archive_sha256": h.hexdigest()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for action in ("run", "suite", "quality"):
        q = sub.add_parser(action)
        q.add_argument("--endpoint", action="append", required=True); q.add_argument("--model", required=True)
        q.add_argument("--out", required=True); q.add_argument("--requests", type=int, default=300)
        q.add_argument("--duration", type=float, default=180); q.add_argument("--timeout", type=float, default=120)
        q.add_argument("--concurrency", type=int, default=4); q.add_argument("--rate", type=float)
        q.add_argument("--prompt-chars", type=int, default=4000); q.add_argument("--output", type=int, default=256)
        q.add_argument("--seed", type=int, default=20261010); q.add_argument("--ttft-slo", type=float, required=True)
        q.add_argument("--e2e-slo", type=float, required=True); q.add_argument("--body", default="{}")
        q.add_argument("--ready-timeout", type=float, default=60); q.add_argument("--deployment", action="append")
        q.add_argument("--corpus"); q.add_argument("--profiled", action="store_true")
        q.add_argument("--cache-protocol", choices=("distinct-prefix", "warm-replay"), default="distinct-prefix")
        q.add_argument("--budget-seconds", type=float, default=7200)
        if action == "suite":
            q.add_argument("--concurrencies", default="1,4,16,64"); q.add_argument("--repeats", type=int, default=1)
            q.add_argument("--resume", action="store_true")
        if action == "quality": q.add_argument("--cases", default=str(HERE / "cases.json"))
    q = sub.add_parser("summary"); q.add_argument("root"); q.add_argument("--out")
    q = sub.add_parser("archive"); q.add_argument("root"); q.add_argument("output")
    args = p.parse_args()
    if args.action == "summary":
        result = summarize(args.root)
        if args.out: write_json(args.out, result)
        print(json.dumps(result, indent=2)); return 0
    if args.action == "archive": archive(args.root, args.output); return 0
    cfg = {k: getattr(args, k) for k in ("model", "requests", "duration", "timeout", "concurrency", "rate", "prompt_chars", "output", "seed", "ttft_slo", "e2e_slo", "profiled")}
    cfg["endpoints"] = [endpoint(x) for x in args.endpoint]
    cfg["body_extra"] = json.loads(args.body)
    cfg["cache_protocol"] = args.cache_protocol
    cfg["deployment"] = [json.loads(Path(x).read_text()) for x in args.deployment or []]
    if any(x.get("profiled") for x in cfg["deployment"]) and not args.profiled:
        raise ValueError("A profiled deployment cannot produce scored capacity evidence")
    if cfg["deployment"] and any(x["resolved"]["config"]["served_model"] != args.model for x in cfg["deployment"]):
        raise ValueError("Deployment/model mismatch")
    validate(cfg)
    cases = json.loads(Path(args.cases).read_text()) if args.action == "quality" else None
    corpus = json.loads(Path(args.corpus).read_text())["entries"] if args.corpus else None
    concurrencies = [int(x) for x in args.concurrencies.split(",")] if args.action == "suite" else [args.concurrency]
    repeats = args.repeats if args.action == "suite" else 1
    if not 1 <= repeats <= 3 or len(concurrencies) > 8 or any(not 1 <= x <= 256 for x in concurrencies):
        raise ValueError("Bounded suite requires <=8 concurrency cells and <=3 repeats")
    if (args.duration + args.timeout) * len(concurrencies) * repeats > args.budget_seconds:
        raise ValueError("Suite worst-case duration exceeds declared wall budget")
    wait_ready(cfg["endpoints"], args.model, args.ready_timeout)
    parent = Path(args.out)
    if args.action == "suite": parent.mkdir(parents=True, exist_ok=args.resume)
    for repeat in range(repeats):
        for concurrency in concurrencies:
            cell = {**cfg, "concurrency": concurrency}
            root = parent / f"c{concurrency}-r{repeat + 1}" if args.action == "suite" else parent
            if args.action == "suite" and args.resume:
                root, previous = resume_cell(root, cell, cases, corpus)
                if previous is not None:
                    print(json.dumps({"cell": str(root), "resume": "retained_complete", "counts": previous["counts"]}), flush=True)
                    continue
            result = run_cell(cell, root, cases, corpus)
            print(json.dumps({"cell": str(root), "counts": result["counts"], "output_tokens_per_s": result["output_tokens_per_s"], "valid": result["valid_capacity_evidence"]}), flush=True)
            if result["missing_usage_attempts"] or result["counts"]["failed"] > max(3, result["attempts"] // 10):
                return 1  # pause suite; preserve evidence before diagnosis/recovery
    return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except (ValueError, OSError, KeyError) as exc:
        print("BLOCKED: " + str(exc), file=sys.stderr); sys.exit(2)
