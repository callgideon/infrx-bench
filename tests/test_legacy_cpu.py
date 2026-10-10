"""No GPU, Docker, network outside loopback, installs, or model downloads."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from infrx_bench import harness, node
from infrx_bench import historical_collect as collect
from infrx_bench.corpus import make_entries


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": self.server.model}]}).encode() if self.path == "/v1/models" else b"fixture_active 0\n"
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        self.server.seen += 1
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append(body)
        mode = self.server.mode
        if mode == "http": self.send_response(429); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        def emit(packet):
            self.wfile.write(b"data: " + json.dumps(packet).encode() + b"\n\n"); self.wfile.flush()
        try:
            if mode == "drip":
                for b in b"data: {\"never_completed\": true}":
                    self.wfile.write(bytes([b])); self.wfile.flush(); time.sleep(.01)
                return
            if mode in ("slow", "timeout"): time.sleep(.08 if mode == "slow" else .2)
            emit({"model": body["model"], "choices": [{"delta": {"reasoning_content": "think"}}]})
            time.sleep(.001)
            observed = "wrong/model" if mode == "identity" else body["model"]
            packet = {"model": observed, "choices": [{"delta": {"content": "[0, 6, 0, 7]"}}]}
            if mode == "missing_identity": packet.pop("model")
            emit(packet)
            emit({"model": observed, "choices": [{"delta": {}, "finish_reason": "stop"}],
                **({} if mode == "usage" else {"usage": {"prompt_tokens": 128, "completion_tokens": 8, "completion_tokens_details": {"reasoning_tokens": 2}}})})
            if mode != "incomplete": self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): pass


@contextmanager
def server(mode="ok", model="fixture/model"):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    srv.daemon_threads = True
    srv.mode, srv.seen, srv.bodies = mode, 0, []
    srv.model = model
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    try: yield srv, f"http://127.0.0.1:{srv.server_port}"
    finally: srv.shutdown(); srv.server_close(); t.join()


def config(base, **extra):
    return {"endpoints": [base], "model": "fixture/model", "requests": 4, "duration": .4, "timeout": .4,
            "concurrency": 2, "rate": None, "prompt_chars": 128, "output": 16, "seed": 4,
            "ttft_slo": 1., "e2e_slo": 1., "body_extra": {}, "telemetry_interval": .05, **extra}


class DriverTests(unittest.TestCase):
    def test_readiness_generation_accounting_and_timing(self):
        with server() as (_, base):
            harness.wait_ready([base], "fixture/model", .3)
            row = harness.request(base, "fixture/model", [{"role": "user", "content": "x"}], 16, .4, time.time(), "a", retain=True)
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["usage"]["completion_tokens"], 8)
        self.assertLess(row["first_generated_at"], row["first_content_at"])
        self.assertLessEqual(row["last_generated_at"], row["end_at"])
        self.assertEqual(row["retry"], 0)

    def test_deadline_cancels_trickling_partial_line(self):
        with server("drip") as (_, base):
            started = time.monotonic()
            row = harness.request(base, "fixture/model", [{"role":"user","content":"x"}], 16, .05, time.time(), "drip")
            self.assertEqual(row["error"], "request_timeout")
            self.assertLess(time.monotonic()-started, .2)

    def test_bad_streams_are_preserved_as_failures(self):
        for mode, error in (("identity", "response_identity_missing_or_mismatched"), ("missing_identity", "response_identity_missing_or_mismatched"),
                            ("usage", "missing_or_invalid_token_usage"), ("incomplete", "incomplete_stream"), ("http", "http_429"), ("timeout", "request_timeout")):
            with self.subTest(mode=mode), server(mode) as (_, base):
                row = harness.request(base, "fixture/model", [{"role": "user", "content": "x"}], 16, .05 if mode == "timeout" else .4, time.time(), "a")
                self.assertEqual(row["status"], "failed"); self.assertEqual(row["error"], error)
                self.assertIsNotNone(row["end_at"])

    def test_closed_run_telemetry_artifact_only_summary_archive(self):
        with tempfile.TemporaryDirectory() as temp, server() as (srv, base):
            root = Path(temp) / "run"
            summary = harness.run_cell(config(base), root)
            self.assertEqual(srv.seen, 4)
            self.assertEqual(summary, harness.summarize(root))
            self.assertEqual(summary["counts"]["success"], 4)
            self.assertIsNone(summary["latency_success_s"]["p95"])
            self.assertTrue(summary["chunk_interval_is_not_token_itl"])
            self.assertTrue((root / "telemetry.jsonl").read_text())
            self.assertEqual(len((root / "scheduled.jsonl").read_text().splitlines()), 4)
            harness.archive(root, Path(temp) / "export.tar.gz")
            self.assertTrue((Path(temp) / "export.tar.gz.manifest.json").exists())
            with self.assertRaises(FileExistsError): harness.run_cell(config(base), root)

    def test_poisson_arrivals_shed_without_hidden_queue(self):
        with tempfile.TemporaryDirectory() as temp, server("slow") as (_, base):
            summary = harness.run_cell(config(base, rate=500., concurrency=1, requests=60, duration=.12), Path(temp) / "open")
            self.assertGreater(summary["counts"]["shed"], 0)
            self.assertGreater(summary["counts"]["success"], 0)
            self.assertEqual(summary["counts"]["unfinished"], 0)

    def test_slow_admission_retains_all_offered_arrivals_and_shed_delay(self):
        original = harness.os.fsync
        delayed = False
        def slow_first_fsync(fd):
            nonlocal delayed
            original(fd)
            if not delayed:
                delayed = True
                time.sleep(.15)
        with tempfile.TemporaryDirectory() as temp, server() as (srv, base):
            root = Path(temp)/"lagged"
            with patch.object(harness.os, "fsync", side_effect=slow_first_fsync):
                summary = harness.run_cell(config(base, rate=1000., requests=8, duration=.02), root)
            self.assertEqual(summary["attempts"], 8)
            self.assertEqual(summary["counts"]["shed"], 8)
            self.assertEqual(srv.seen, 0)
            self.assertEqual(len((root/"scheduled.jsonl").read_text().splitlines()), 8)
            self.assertGreater(summary["driver_delay_s"]["min"], .1)
            self.assertFalse(summary["valid_capacity_evidence"])
            rows, _ = harness.read_rows(root/"attempts.jsonl")
            self.assertTrue(all(r["error"] == "client_admission_window_elapsed" for r in rows))

    def test_quality_failure_not_silently_passed(self):
        cases = [{"id": "pass", "prompt": "a", "expected": [0,6,0,7]}, {"id": "fail", "prompt": "b", "expected": [1]}]
        with tempfile.TemporaryDirectory() as temp, server() as (_, base):
            summary = harness.run_cell(config(base), Path(temp) / "quality", cases=cases)
            self.assertEqual(summary["quality_diagnostic"], {"n": 4, "passed": 2})
            self.assertLess(summary["goodput_requests_per_s"], summary["requests_per_s"])

    def test_interrupted_artifact_retains_unfinished_denominator(self):
        with tempfile.TemporaryDirectory() as temp, server() as (_, base):
            root = Path(temp) / "run"; harness.run_cell(config(base), root)
            with (root / "scheduled.jsonl").open("a") as f: f.write(json.dumps({"attempt_id": "lost", "scheduled_at": time.time()}) + "\n")
            (root / "completion.json").unlink()
            summary = harness.summarize(root)
            self.assertEqual(summary["attempts"], 5)
            self.assertEqual(summary["counts"]["unfinished"], 1)
            self.assertFalse(summary["valid_capacity_evidence"])

    def test_caps_and_identity_cannot_be_overridden(self):
        for changes in ({"concurrency": 257}, {"rate": float("inf")}, {"duration": float("nan")}, {"body_extra": {"max_tokens": 100000}}, {"e2e_slo": 0}):
            with self.assertRaises(ValueError): harness.validate(config("http://localhost", **changes))
        with self.assertRaises(ValueError): harness.endpoint("http://secret:password@localhost")

    def test_reservation_window_and_export_reserve(self):
        collect.check_window(101, 100, 4000)
        for now in (99, 2201, 4000):
            with self.assertRaises(ValueError): collect.check_window(now, 100, 4000)

    def test_cli_suite_two_configs_and_budget_guard(self):
        with tempfile.TemporaryDirectory() as temp, server() as (srv, base):
            cmd = [sys.executable, str(harness.HERE / "harness.py"), "suite", "--endpoint", base, "--model", "fixture/model", "--out", str(Path(temp)/"suite"), "--requests", "2", "--duration", ".1", "--timeout", ".2", "--concurrencies", "1,2", "--ttft-slo", "1", "--e2e-slo", "1"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(srv.seen, 4)
            result = subprocess.run(cmd + ["--budget-seconds", ".01"], capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 2); self.assertIn("wall budget", result.stderr)

    def test_suite_retry_is_retained_and_changed_corpus_rejected(self):
        with tempfile.TemporaryDirectory() as temp, server() as (srv, base):
            root = Path(temp)/"suite"
            corpus = Path(temp)/"corpus.json"
            corpus.write_text(json.dumps({"entries":[{"id":"a", "messages":[{"role":"user","content":"original"}]}]}))
            cmd = [sys.executable, str(harness.HERE/"harness.py"), "suite", "--endpoint", base, "--model", "fixture/model", "--out", str(root), "--requests", "2", "--duration", ".1", "--timeout", ".2", "--concurrencies", "1", "--ttft-slo", "1", "--e2e-slo", "1", "--corpus", str(corpus)]
            self.assertEqual(subprocess.run(cmd, capture_output=True, text=True, timeout=5).returncode, 0)
            (root/"c1-r1"/"completion.json").unlink()
            retry = subprocess.run(cmd+["--resume"], capture_output=True, text=True, timeout=5)
            self.assertEqual(retry.returncode, 0, retry.stderr)
            self.assertEqual(len(list(root.glob("c1-r1-retry-*"))), 1)
            before = srv.seen
            retained = subprocess.run(cmd+["--resume"], capture_output=True, text=True, timeout=5)
            self.assertEqual(retained.returncode, 0, retained.stderr)
            self.assertEqual(srv.seen, before)
            corpus.write_text(corpus.read_text().replace("original", "changed"))
            rejected = subprocess.run(cmd+["--resume"], capture_output=True, text=True, timeout=5)
            self.assertEqual(rejected.returncode, 2)
            self.assertIn("corpus", rejected.stderr)
            self.assertEqual(srv.seen, before)

    def test_resume_rejects_cases_and_driver_changes(self):
        with tempfile.TemporaryDirectory() as temp, server() as (_, base):
            root = Path(temp)/"cell"
            cfg = config(base)
            cases = [{"id":"a", "prompt":"a", "expected":[0,6,0,7]}]
            harness.run_cell(cfg, root, cases=cases)
            with self.assertRaisesRegex(ValueError, "cases"):
                harness.resume_cell(root, cfg, cases=[{**cases[0], "expected":[1]}])
            with patch.object(harness, "driver_hash", return_value="changed"):
                with self.assertRaisesRegex(ValueError, "driver"):
                    harness.resume_cell(root, cfg, cases=cases)

    def test_profiled_results_never_capacity_evidence(self):
        with tempfile.TemporaryDirectory() as temp, server() as (_, base):
            summary = harness.run_cell(config(base, profiled=True), Path(temp)/"profile")
            self.assertFalse(summary["valid_capacity_evidence"])

    def test_small_model_queue_and_resume_end_to_end(self):
        selected_arm="qwen-smoke-vllm031"
        model = node.arm(selected_arm)["served_model"]
        with tempfile.TemporaryDirectory() as temp, server(model=model) as (srv, base):
            root=Path(temp)
            deployment=root/"launch.json"
            node.write_json(deployment, {"resolved":{"arm":selected_arm, "config":{"served_model":model}}, "profiled":False})
            cmd=[sys.executable, str(node.HERE/"historical_collect.py"), "--arm", selected_arm, "--endpoint",base, "--deployment",str(deployment), "--out",str(root/"queue"), "--ttft-slo","1", "--e2e-slo","2", "--budget-seconds","600", "--smoke", "--execute"]
            result=subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((root/"queue"/"queue-complete.json").exists())
            (root/"queue"/"quality-before"/"completion.json").unlink()
            retry=subprocess.run(cmd+["--resume"], capture_output=True, text=True, timeout=10)
            self.assertEqual(retry.returncode,0,retry.stderr)
            self.assertEqual(sum(p.is_dir() for p in (root/"queue").glob("quality-before-retry-*")),1)
            before=srv.seen
            result=subprocess.run(cmd+["--resume"], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(srv.seen,before)


class NodeTests(unittest.TestCase):
    def test_reusing_weights_still_stages_selected_runtime_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/"staging.json").write_text("{}")
            item={"config":node.arm("qwen-smoke-vllm031"),"image":node.arm("qwen-smoke-vllm031")["image_digest"]}
            args=SimpleNamespace(arm="qwen-smoke-vllm031",resolved="fixture",weights=str(root),execute=True,timeout=5)
            with patch.object(node,"load_resolved",return_value=item), patch.object(node,"verify_staging",return_value="fixture") as verified, patch.object(node,"run",return_value=SimpleNamespace(returncode=0)) as called, patch.object(node.time,"monotonic",return_value=100):
                node.stage(args)
                verified.assert_called_once()
                called.assert_called_once_with(["docker","pull",item["image"]],timeout=5)

    def test_staging_timeout_failure_and_interrupt_remove_only_owned_helper(self):
        token, cid = "b"*32, "a"*64
        for error in (subprocess.TimeoutExpired("stage", 5), KeyboardInterrupt(), None):
            with self.subTest(error=type(error).__name__), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                cfg = node.arm("qwen-smoke-vllm031")
                item = {"config":cfg, "image":cfg["image_digest"]}
                args = SimpleNamespace(arm="qwen-smoke-vllm031", resolved="fixture", weights=str(root/"weights"), out=str(root/"stage.log"), execute=True, timeout=5, forward_hf_token=False)
                responses = [SimpleNamespace(returncode=0), SimpleNamespace(returncode=0, stdout=cid+" 1 "+token), SimpleNamespace(returncode=0)]
                with patch.object(node,"load_resolved",return_value=item), patch.object(node.shutil,"disk_usage",return_value=SimpleNamespace(free=100*1024**3)), patch.object(node.uuid,"uuid4",return_value=SimpleNamespace(hex=token)), patch.object(node,"run",side_effect=responses) as calls, patch.object(node.subprocess,"run",side_effect=error, return_value=SimpleNamespace(returncode=1)) as stage_run:
                    with self.assertRaises(type(error) if error else ValueError): node.stage(args)
                    command = stage_run.call_args.args[0]
                    self.assertIn("b200-stage-"+token, command)
                    self.assertIn(node.OWNER+".stage="+token, command)
                    self.assertEqual(calls.call_args_list[-1].args[0], ["docker","rm","--force",cid])

    def test_staging_uses_one_deadline_and_does_not_start_when_exhausted(self):
        for times, exhausted in (([100,100,103], False), ([100,100,111], True)):
            with self.subTest(exhausted=exhausted), tempfile.TemporaryDirectory() as temp:
                cfg = node.arm("qwen-smoke-vllm031")
                item = {"config":cfg, "image":cfg["image_digest"]}
                args = SimpleNamespace(arm="qwen-smoke-vllm031", resolved="fixture", weights=str(Path(temp)/"weights"), out=str(Path(temp)/"stage.log"), execute=True, timeout=10, forward_hf_token=False)
                with patch.object(node,"load_resolved",return_value=item), patch.object(node.shutil,"disk_usage",return_value=SimpleNamespace(free=100*1024**3)), patch.object(node.time,"monotonic",side_effect=times), patch.object(node,"run",return_value=SimpleNamespace(returncode=0)) as pull, patch.object(node.subprocess,"run",return_value=SimpleNamespace(returncode=0)) as staged, patch.object(node,"cleanup_staging") as cleaned:
                    if exhausted:
                        with self.assertRaises(subprocess.TimeoutExpired): node.stage(args)
                        staged.assert_not_called()
                    else:
                        node.stage(args)
                        self.assertEqual(staged.call_args.kwargs["timeout"], 7)
                    self.assertEqual(pull.call_args.kwargs["timeout"], 10)
                    cleaned.assert_called_once()

    def test_cleanup_refuses_other_invocation_and_tolerates_already_removed(self):
        for response, error in ((SimpleNamespace(returncode=0, stdout="a"*64+" 1 other"), True),
                                (SimpleNamespace(returncode=1, stderr="No such object: fixture"), False),
                                (SimpleNamespace(returncode=1, stderr="daemon unavailable"), True)):
            with self.subTest(response=response), patch.object(node,"run",return_value=response) as called:
                if error:
                    with self.assertRaises(ValueError): node.cleanup_staging("fixture", "expected")
                else: node.cleanup_staging("fixture", "expected")
                self.assertEqual(called.call_count, 1)

    def test_launch_failure_timeout_and_interrupt_cleanup_with_shared_deadline(self):
        cfg = node.arm("qwen-smoke-vllm031")
        item = {"arm":"qwen-smoke-vllm031", "config":cfg, "image":cfg["image_digest"]}
        for mode in ("creation_timeout", "creation_failure", "interrupt", "ready_timeout", "success"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                args = SimpleNamespace(arm=item["arm"], resolved="fixture", weights=temp, gpus="0", port=18300, name="b200-fixture", profile=None, execute=True, experimental=False, timeout=10, out=str(Path(temp)/"launch"))
                clock = [100.]
                def fake_run(command, timeout=30):
                    if command[0] == "nvidia-smi":
                        return SimpleNamespace(returncode=0, stdout="0, fixture-uuid, NVIDIA L40S" if "--query-gpu=index,uuid,name" in command else "")
                    if "--detach" in command:
                        self.assertEqual(timeout, 10)
                        self.assertTrue(any(x.startswith(node.OWNER+".launch=") for x in command))
                        clock[0] += 3
                        if mode == "creation_timeout": raise subprocess.TimeoutExpired(command, timeout)
                        return SimpleNamespace(returncode=1 if mode == "creation_failure" else 0, stdout="a"*64, stderr="fixture failure")
                    return SimpleNamespace(returncode=0, stdout=" ".join(cfg["flags"]), stderr="")
                ready_error = KeyboardInterrupt() if mode == "interrupt" else ValueError("readiness timeout") if mode == "ready_timeout" else None
                with patch.object(node,"load_resolved",return_value=item), patch.object(node,"verify_staging",return_value="fixture"), patch.object(node.time,"monotonic",side_effect=lambda:clock[0]), patch.object(node,"run",side_effect=fake_run), patch.object(node,"cleanup_container") as cleanup, patch.object(harness,"wait_ready",side_effect=ready_error) as ready:
                    if mode == "success": node.launch(args)
                    else:
                        expected = subprocess.TimeoutExpired if mode == "creation_timeout" else KeyboardInterrupt if mode == "interrupt" else ValueError
                        with self.assertRaises(expected): node.launch(args)
                    if mode == "success": cleanup.assert_not_called()
                    else:
                        cleanup.assert_called_once()
                        self.assertEqual(cleanup.call_args.args[0], args.name)
                        self.assertEqual(cleanup.call_args.args[2], "launch")
                    if mode not in ("creation_timeout", "creation_failure"):
                        self.assertEqual(ready.call_args.args[2], 7)

    def test_unsupported_cells_block_and_inheritance(self):
        for name in ("fp8", "tensorrt-llm"):
            with self.assertRaisesRegex(ValueError, "BLOCKED"): node.arm(name)
        self.assertIn("DSPARK", node.arm("native-latency")["flags"])
        self.assertNotIn("--ep-size", node.arm("nvfp4-b200")["flags"])

    def test_small_model_runtime_alternatives_keep_identity_and_pins(self):
        original=node.arm("qwen-smoke")
        modern=node.arm("qwen-smoke-vllm031")
        preview=node.arm("qwen-smoke-sglang")
        self.assertNotEqual(original["image_digest"],modern["image_digest"])
        self.assertEqual(modern["revision"],original["revision"])
        self.assertEqual(modern["served_model"],preview["served_model"])
        self.assertEqual(preview["engine"],"sglang")
        self.assertIn("--context-length",preview["flags"])
        self.assertNotIn("--max-model-len",preview["flags"])

    def test_owned_launch_is_immutable_local_and_gpu_scoped(self):
        item = {"arm": "native-throughput", "config": node.arm("native-throughput"), "image": "lmsysorg/sglang@sha256:" + "a"*64}
        cmd = node.launch_command(item, Path("/tmp/weights"), "0,1,2,3", 18300, "b200-fixture")
        self.assertIn("--pull=never", cmd); self.assertIn("127.0.0.1:18300:30000", cmd)
        self.assertIn('"device=0,1,2,3"', cmd); self.assertIn("/tmp/weights:/model:ro", cmd)
        self.assertNotIn("--privileged", cmd)
        for gpus in ("0", "0,0,1,2", "0,1,2,3,4,5,6,7"):
            with self.assertRaises(ValueError): node.launch_command(item, Path("/tmp/weights"), gpus, 18300, "b200-fixture")

    def test_staging_requires_matching_pinned_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cfg = node.arm("qwen-smoke")
            p = root/"config.json"; p.write_text("{}")
            weight = root/"fixture.safetensors"; weight.write_bytes(b"fixture; not a real model")
            receipt = {"repo": cfg["repo"], "revision": cfg["revision"], "verified": True, "files": [{"path": x.name, "size": x.stat().st_size, "mtime_ns": x.stat().st_mtime_ns, "sha256": node.sha(x)} for x in (p, weight)]}
            node.write_json(root/"staging.json", receipt)
            self.assertEqual(len(node.verify_staging(root, cfg)), 64)
            p.write_text("{\"changed\":true}")
            with self.assertRaises(ValueError): node.verify_staging(root, cfg)

    def test_corpus_uses_tokenizer_not_character_guess(self):
        entries = make_entries(lambda s: list(s), 1024, 2, 9)
        self.assertEqual(len(entries), 2)
        self.assertLessEqual(abs(entries[0]["content_tokens"]-1024), 6)
        self.assertNotEqual(entries[0]["prompt_sha256"], entries[1]["prompt_sha256"])


if __name__ == "__main__": unittest.main()
