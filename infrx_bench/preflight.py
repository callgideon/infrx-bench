"""Read-only device inventory. Does not install, launch, download or stop anything."""
import json
import platform
import shutil
import subprocess
import urllib.request
from pathlib import Path

from . import harness


def inventory(endpoint=None, model=None):
    report = {"schema":"infrx-preflight/1", "utc":harness.utc(), "system":platform.system(),
        "architecture":platform.machine(), "python":platform.python_version(), "tools":{},
        "gpu_qualification":"NOT_RUN", "paid_actions":False}
    for name, path in {"device_model":"/proc/device-tree/model", "l4t_release":"/etc/nv_tegra_release", "os_release":"/etc/os-release"}.items():
        source = Path(path)
        if source.is_file(): report[name] = source.read_bytes()[:4096].decode(errors="replace").strip("\x00\n")
    commands = {"nvidia-smi":["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,driver_version", "--format=csv,noheader"],
        "gpu_processes":["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
        "docker":["docker", "version", "--format", "{{.Server.Version}}"],
        "nsys":["nsys", "--version"], "ncu":["ncu", "--version"]}
    for label, command in commands.items():
        if not shutil.which(command[0]): report["tools"][label] = {"status":"unavailable"}; continue
        try:
            r = subprocess.run(command, capture_output=True, text=True, timeout=10)
            report["tools"][label] = {"exit_code":r.returncode, "stdout":r.stdout[:32768], "stderr":r.stderr[:4096]}
        except (OSError, subprocess.TimeoutExpired) as e: report["tools"][label] = {"error":type(e).__name__}
    if endpoint:
        base = harness.endpoint(endpoint)
        try:
            req = urllib.request.Request(base+"/v1/models", headers=harness.headers())
            with urllib.request.urlopen(req, timeout=5) as r: data = json.loads(r.read(1024*1024))
            ids = [x.get("id") for x in data.get("data", [])]
            report["endpoint"] = {"models":ids, "requested_model_present":model in ids if model else None,
                "generation":"NOT_RUN", "weights_identity":"requires_local_attestation"}
        except (OSError, ValueError) as e: report["endpoint"] = {"error":type(e).__name__}
    return report
