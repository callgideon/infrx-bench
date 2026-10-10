import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import sys
import unittest
from unittest import mock

from infrx_bench import assets, budget, datasets

DATA = b"bounded fixture bytes\n" * 20


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
        self.send_response(206 if start and self.server.range_supported else 200)
        if start and self.server.range_supported:
            self.send_header("Content-Range", f"bytes {start}-{len(DATA)-1}/{len(DATA)}")
        else:
            start = 0
        self.send_header("Content-Length", str(len(DATA)-start)); self.end_headers()
        self.wfile.write(DATA[start:])


class Assets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()/"download"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.range_supported = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.manifest = {"schema": assets.SCHEMA, "repo":"fixture/public", "revision":"a"*40,
                         "access":"public-ungated", "license":"fixture-only",
                         "files":[{"path":"asset.bin", "bytes":len(DATA), "sha256":hashlib.sha256(DATA).hexdigest(),
                                   "url": f"http://127.0.0.1:{self.server.server_port}/asset"}]}

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(); self.tmp.cleanup()

    def pull(self, **kwargs):
        return assets.download(self.manifest, self.root, 30, 100000, license_reviewed=True, fixture=True, **kwargs)

    def test_download_is_checksum_bound_and_idempotent(self):
        first = self.pull(); second = self.pull()
        self.assertEqual(first, second); self.assertTrue(first["verified"])
        self.assertEqual((self.root/"asset.bin").read_bytes(), DATA)

    def test_resume_requires_exact_range(self):
        self.root.mkdir(); (self.root/"asset.bin.part").write_bytes(DATA[:25])
        self.assertTrue(self.pull()["verified"])

    def test_server_ignoring_range_preserves_partial(self):
        self.root.mkdir(); (self.root/"asset.bin.part").write_bytes(DATA[:25]); self.server.range_supported = False
        with self.assertRaisesRegex(ValueError, "honor"): self.pull()
        self.assertEqual((self.root/"asset.bin.part").read_bytes(), DATA[:25])

    def test_checksum_failure_has_no_receipt(self):
        self.manifest["files"][0]["sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "checksum"): self.pull()
        self.assertFalse((self.root/"staging.json").exists()); self.assertTrue((self.root/"asset.bin.part").exists())

    def test_changed_manifest_is_not_resumed(self):
        self.pull(); self.manifest["license"] = "changed"
        with self.assertRaisesRegex(ValueError,"identical"): self.pull()

    def test_deadline_never_resets_on_resume(self):
        self.pull(); path = self.root/"download-state.json"; state = json.loads(path.read_text()); state["expires_at"] = time.time()-1; path.write_text(json.dumps(state))
        with self.assertRaises(TimeoutError): self.pull()
        self.assertEqual(json.loads(path.read_text())["expires_at"], state["expires_at"])

    def test_existing_file_never_overwritten(self):
        self.root.mkdir(); (self.root/"asset.bin").write_bytes(b"operator content")
        with self.assertRaisesRegex(ValueError,"Existing"): self.pull()
        self.assertEqual((self.root/"asset.bin").read_bytes(), b"operator content")

    def test_paths_credentials_and_control_files_refused(self):
        for path in ("../escape", "/absolute", "a/../b", "a\\b", "staging.json", "download-state.json", "download.lock", "file.part"):
            self.manifest["files"][0]["path"] = path
            with self.assertRaises(ValueError): assets.validate(self.manifest, fixture=True)
        self.manifest["files"][0]["path"] = "asset.bin"
        self.manifest["files"][0]["url"] = "https://user:password@huggingface.co/a"
        with self.assertRaises(ValueError): assets.validate(self.manifest)

    def test_symlink_and_budget_and_license_fail_before_transfer(self):
        self.root.symlink_to(Path(self.tmp.name))
        with self.assertRaisesRegex(ValueError,"symlink"): self.pull()
        self.root.unlink()
        with self.assertRaises(ValueError): assets.download(self.manifest,self.root,30,1,license_reviewed=True,fixture=True)
        with self.assertRaises(ValueError): assets.download(self.manifest,self.root,30,10000,fixture=True)
        self.assertFalse(self.root.exists())

    def test_disk_gate_blocks_before_creation(self):
        with mock.patch("shutil.disk_usage", return_value=mock.Mock(free=0)):
            with self.assertRaises(ValueError): self.pull()
        self.assertFalse(self.root.exists())

    def test_watchdog_kills_only_its_child(self):
        start = time.monotonic()
        original = budget.subprocess.Popen
        def child(_args, **kwargs):
            return original([sys.executable,"-c","import threading; threading.Event().wait(10)"], **kwargs)
        with mock.patch("infrx_bench.budget.subprocess.Popen", side_effect=child):
            with self.assertRaisesRegex(TimeoutError,"watchdog"):
                budget.supervised("fixture-stall",{},time.time()+.15)
        self.assertLess(time.monotonic()-start,2)


class Datasets(unittest.TestCase):
    def test_frozen_task_import_is_seeded_and_retains_joint_rubrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp).resolve()/"dataset.json"
            path.write_text(json.dumps({"entries":[{"id":str(i),"messages":[{"role":"user","content":f"Return {i}."}],"rubric":{"kind":"exact_text","expected":str(i)}} for i in range(20)]}))
            raw = path.read_bytes(); rev="b"*40
            manifest={"schema":assets.SCHEMA,"repo":"fixture/tasks","revision":rev,"license":"fixture-only","access":"public-ungated","kind":"dataset","split":"test","format":"infrx-tasks-json",
                      "files":[{"path":"dataset.json","bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"url":f"https://raw.githubusercontent.com/fixture/tasks/{rev}/dataset.json"}]}
            first = datasets.import_tasks(manifest,path,5,42)
            self.assertEqual(first,datasets.import_tasks(manifest,path,5,42)); self.assertNotEqual(first,datasets.import_tasks(manifest,path,5,43))
            self.assertTrue(all(task["rubric"] for task in first["entries"]))
            path.write_text("{}");
            with self.assertRaisesRegex(ValueError,"bytes differ"): datasets.import_tasks(manifest,path,5,42)
