"""Persisted UTC deadlines, checked I/O and a watchdog for client processes only."""
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import harness


def check(epoch, reserve=0):
    remaining = epoch - time.time() - reserve
    if remaining <= 0: raise TimeoutError("Absolute run deadline reached")
    return remaining


def window(cfg, root, started):
    path = Path(root)/"run-window.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if time.time() < saved["started_at"] - 1:
            raise ValueError("Host clock moved behind original run start; reconcile clock before resume")
    else:
        expires = started + cfg["budget_seconds"]
        if cfg.get("end_utc"):
            expires = min(expires, dt.datetime.fromisoformat(cfg["end_utc"].replace("Z", "+00:00")).timestamp())
        reserve = cfg.get("export_reserve_seconds", 0)
        saved = {"schema":"infrx-window/2", "started_at":started, "expires_at":expires,
                 "collection_expires_at":expires-reserve, "export_reserve_seconds":reserve}
        harness.write_json(path, saved)
    check(saved["expires_at"])
    return saved


def supervised(module, payload, epoch):
    """Kill only our isolated client process group if any I/O/drain stalls."""
    remaining = check(epoch)
    process = subprocess.Popen([sys.executable, "-m", module], stdin=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        process.communicate(json.dumps(payload), timeout=remaining)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        # No server, GPU service or external process belongs to this group.
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)
        raise TimeoutError("Client watchdog stopped work at the durable absolute deadline")
    if process.returncode:
        raise ValueError("Supervised client paused; inspect preserved evidence")
