#!/usr/bin/env python3
"""Owned-container staging/launch/recovery. Defaults to a plan; never provisions."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
OWNER = "callgideon.b200.readiness"


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def arm(name, matrix=None):
    data = json.loads(Path(matrix or HERE / "matrix.json").read_text())
    item = data["arms"][name]
    if "inherits" in item:
        item = {**data["arms"][item["inherits"]], **item}
    if item["status"] == "blocked":
        raise ValueError(f"BLOCKED {name}: {item['reason']}")
    return item


def run(argv, timeout=30):
    # No shell or environment dumps; callers use bounded public arguments.
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def public_json(url):
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read(4 * 1024 * 1024))


def resolve(args):
    """Read small metadata only. Digest is resolved before any image pull."""
    cfg = arm(args.arm)
    tag = cfg["image_tag"]
    if cfg.get("image_digest"):
        digest = cfg["image_digest"]
    else:
        repo, image_tag = tag.split(":", 1)
        auth = public_json("https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + repo + ":pull")
        req = urllib.request.Request("https://registry-1.docker.io/v2/" + repo + "/manifests/" + image_tag,
            headers={"Authorization": "Bearer " + auth["token"], "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read(4 * 1024 * 1024)
            value = r.headers.get("Docker-Content-Digest")
        if not value or value != "sha256:" + hashlib.sha256(body).hexdigest():
            raise ValueError("Registry digest absent or mismatched")
        digest = repo + "@" + value
    if not re.fullmatch(r"[a-z0-9./_-]+@sha256:[0-9a-f]{64}", digest):
        raise ValueError("An immutable image digest is required")
    info = public_json("https://huggingface.co/api/models/" + cfg["repo"] + "/revision/" + cfg["revision"])
    if info.get("id") != cfg["repo"] or info.get("sha") != cfg["revision"] or info.get("disabled"):
        raise ValueError("HF identity/revision mismatch or disabled model")
    if info.get("gated"):
        raise ValueError("Gated weights: access must be resolved by the operator; this tool never retrieves credentials")
    result = {"schema": 1, "arm": args.arm, "resolved_utc": utc(), "config": cfg, "image": digest,
              "hf_revision": info["sha"], "license": info.get("cardData", {}).get("license"),
              "hf_source": "https://huggingface.co/api/models/" + cfg["repo"],
              "matrix_sha256": sha(HERE / "matrix.json")}
    if result["license"] != cfg["license"]:
        raise ValueError("License mismatch; inspect upstream terms before staging")
    write_json(args.out, result)


def load_resolved(path, name):
    item = json.loads(Path(path).read_text())
    cfg = arm(name)
    if item["arm"] != name or item["matrix_sha256"] != sha(HERE / "matrix.json") or item["config"] != cfg:
        raise ValueError("Resolved manifest belongs to another arm/matrix")
    if item["hf_revision"] != cfg["revision"] or not re.fullmatch(r"[a-z0-9./_-]+@sha256:[0-9a-f]{64}", item["image"]):
        raise ValueError("Missing immutable pins")
    return item


def preflight(args):
    cfg = arm(args.arm)
    root = Path(args.work).resolve()
    root.mkdir(parents=True, exist_ok=True)
    facts = {"utc": utc(), "arm": args.arm, "free_disk_gib": shutil.disk_usage(root).free / 1024**3,
             "required_free_disk_gib": cfg["minimum_free_disk_gib"], "tools": {}, "checks": {}}
    for name, cmd in {"docker": ["docker", "version", "--format", "{{.Server.Version}}"],
                      "nvidia-smi": ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader"],
                      "topology": ["nvidia-smi", "topo", "-m"],
                      "nsys": ["nsys", "--version"], "ncu": ["ncu", "--version"]}.items():
        try:
            result = run(cmd)
            facts["tools"][name] = {"exit": result.returncode, "output": result.stdout[:16384]}
        except (OSError, subprocess.TimeoutExpired) as exc:
            facts["tools"][name] = {"exit": None, "error": type(exc).__name__}
    facts["checks"]["disk"] = facts["free_disk_gib"] >= cfg["minimum_free_disk_gib"]
    facts["checks"]["docker"] = facts["tools"]["docker"]["exit"] == 0
    facts["checks"]["gpu"] = facts["tools"]["nvidia-smi"]["exit"] == 0
    facts["host_memory"] = Path("/proc/meminfo").read_text() if Path("/proc/meminfo").exists() else None
    facts["status"] = "inventory_only"  # does not establish free GPU ownership or CUDA compatibility
    write_json(args.out, facts)
    print(json.dumps(facts, indent=2))


def stage(args):
    item = load_resolved(args.resolved, args.arm)
    cfg = item["config"]
    print(json.dumps({"arm": args.arm, "repo": cfg["repo"], "revision": cfg["revision"], "image": item["image"],
                      "weights": str(Path(args.weights).resolve()), "download": "explicit execute only"}, indent=2))
    if not args.execute:
        return
    if args.timeout <= 0:
        raise ValueError("Staging timeout must be positive")
    deadline = time.monotonic() + args.timeout

    def remaining(command):
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise subprocess.TimeoutExpired(command, args.timeout)
        return seconds

    root = Path(args.weights).resolve()
    root.mkdir(parents=True, exist_ok=True)
    reuse = (root / "staging.json").exists()
    if reuse:
        verify_staging(root, cfg)
    if not reuse and shutil.disk_usage(root).free < cfg["minimum_free_disk_gib"] * 1024**3:
        raise ValueError("Insufficient free staging disk; allow both checkpoints plus results separately")
    # Dependencies come from the selected image. No host installation.
    pull_cmd = ["docker", "pull", item["image"]]
    pulled = run(pull_cmd, timeout=remaining(pull_cmd))
    if pulled.returncode:
        raise ValueError("Image pull failed; no launch attempted")
    if reuse:
        print("Existing pinned staging receipt verified; exact runtime image pulled; weight download skipped")
        return
    script = r'''
import hashlib, json, os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo, revision = os.environ['STAGE_REPO'], os.environ['STAGE_REVISION']
info = HfApi().model_info(repo, revision=revision, files_metadata=True)
assert info.sha == revision
patterns = ['*.safetensors','*.json','*.jinja','*.py','*.txt','*.model','*.tiktoken','LICENSE','README.md','encoding/**','inference/**']
snapshot_download(repo, revision=revision, local_dir='/weights', allow_patterns=patterns)
files=[]
for entry in info.siblings:
    p=Path('/weights')/entry.rfilename
    if not p.is_file(): continue
    assert entry.size is None or p.stat().st_size == entry.size
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''): h.update(chunk)
    digest=h.hexdigest()
    if entry.lfs: assert digest == entry.lfs.sha256, entry.rfilename
    files.append({'path':entry.rfilename,'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'sha256':digest})
assert any(x['path'].endswith('.safetensors') for x in files)
assert any(x['path']=='config.json' for x in files)
index=Path('/weights/model.safetensors.index.json')
if index.exists():
    needed=set(json.loads(index.read_text())['weight_map'].values())
    assert needed <= {x['path'] for x in files}
receipt={'repo':repo,'revision':revision,'verified':True,'files':files}
with Path('/weights/staging.json').open('x') as f: json.dump(receipt,f,indent=2)
'''
    token = uuid.uuid4().hex
    name = "b200-stage-" + token
    cmd = ["docker", "run", "--rm", "--pull=never", "--name", name,
           "--label", OWNER + "=1", "--label", OWNER + ".stage=" + token,
           "--entrypoint", "python", "-v", str(root) + ":/weights",
           "-e", "STAGE_REPO=" + cfg["repo"], "-e", "STAGE_REVISION=" + cfg["revision"]]
    # Public weights require no token. Forward an existing authorized token only if explicitly supplied.
    if args.forward_hf_token:
        if not os.environ.get("HF_TOKEN"):
            raise ValueError("HF_TOKEN absent")
        cmd += ["-e", "HF_TOKEN"]
    cmd += [item["image"], "-c", script]
    try:
        with Path(args.out).open("x") as log:
            completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=remaining(cmd))
    finally:
        cleanup_staging(name, token)
    if completed.returncode:
        raise ValueError("Staging failed; inspect restricted local staging log; no launch attempted")


def cleanup_staging(name, token):
    """Remove only this invocation's labelled helper, including after CLI timeout."""
    cleanup_container(name, token, "stage")


def cleanup_container(name, token, purpose, log_path=None):
    fmt = '{{.Id}} {{index .Config.Labels "' + OWNER + '"}} {{index .Config.Labels "' + OWNER + '.' + purpose + '"}}'
    inspected = run(["docker", "inspect", "--format", fmt, name], timeout=10)
    if inspected.returncode:
        if "No such object" in inspected.stderr or "No such container" in inspected.stderr:
            return  # --rm already removed it, or the run never created it.
        raise ValueError("Cannot verify owned container; cleanup blocked")
    fields = inspected.stdout.strip().split()
    if len(fields) != 3 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]) or fields[1:] != ["1", token]:
        raise ValueError("Container ownership mismatch; cleanup refused")
    try:
        if log_path:
            try:
                logs = run(["docker", "logs", "--tail", "200", fields[0]], timeout=10)
                Path(log_path).write_text((logs.stdout + logs.stderr)[:65536])
            except (OSError, subprocess.TimeoutExpired):
                pass
    finally:
        removed = run(["docker", "rm", "--force", fields[0]], timeout=30)
    if removed.returncode and "No such container" not in removed.stderr and "No such object" not in removed.stderr:
        raise ValueError("Owned container cleanup failed")


def verify_staging(root, cfg):
    root = Path(root).resolve()
    receipt = json.loads((root / "staging.json").read_text())
    if not receipt.get("verified") or receipt["repo"] != cfg["repo"] or receipt["revision"] != cfg["revision"]:
        raise ValueError("Staged checkpoint lacks matching verified receipt")
    paths = {x["path"] for x in receipt["files"]}
    if "config.json" not in paths or not any(x.endswith(".safetensors") for x in paths):
        raise ValueError("Receipt must contain model config and weight shards")
    index = root / "model.safetensors.index.json"
    if index.exists() and not set(json.loads(index.read_text())["weight_map"].values()) <= paths:
        raise ValueError("Weight index references unstaged shards")
    for entry in receipt["files"]:
        p = (root / entry["path"]).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            raise ValueError("Invalid or missing staged file")
        stat = p.stat()
        if stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]:
            raise ValueError("Staged file changed; restage/reverify")
        if not p.name.endswith(".safetensors") and sha(p) != entry["sha256"]:
            raise ValueError("Staged metadata checksum mismatch")
    return sha(root / "staging.json")


def launch_command(item, weights, gpus, port, name, profile=None, ownership_token=None):
    cfg = item["config"]
    if not re.fullmatch(r"[0-9]+(,[0-9]+)*", gpus) or len(set(gpus.split(","))) != cfg["gpus"]:
        raise ValueError("Select exactly the arm's distinct GPU count")
    if not 1024 <= port <= 65535 or port in (8000, 8100, 8110):
        raise ValueError("Use a task-local port, outside existing app/engine ports")
    if not re.fullmatch(r"b200-[a-z0-9-]{1,48}", name):
        raise ValueError("Owned container name must begin b200-")
    cmd = ["docker", "run", "--detach", "--pull=never", "--name", name, "--label", OWNER + "=1",
           "--label", OWNER + ".arm=" + item["arm"], "--gpus", '"device=' + gpus + '"', "--ipc=host", "--shm-size", "32g",
           "-p", f"127.0.0.1:{port}:30000", "-v", str(weights) + ":/model:ro", "-e", "HF_HUB_OFFLINE=1"]
    if ownership_token:
        cmd += ["--label", OWNER + ".launch=" + ownership_token]
    engine = (["sglang", "serve", "--model-path", "/model"] if cfg["engine"] == "sglang"
              else ["vllm", "serve", "/model"])
    engine += ["--served-model-name", cfg["served_model"], "--host", "0.0.0.0", "--port", "30000"] + cfg["flags"]
    if profile:
        cmd += ["-v", str(profile) + ":/profiles"]
        # CUDA processes are children of this launch; use system-wide process tree collection from nsys.
        engine = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--delay=600", "--duration=60",
                  "--force-overwrite=false", "--output=/profiles/target", *engine]
    return cmd + ["--entrypoint", engine[0], item["image"], *engine[1:]]


def owned(name):
    r = run(["docker", "inspect", "--format", '{{index .Config.Labels "' + OWNER + '"}}', name])
    if r.returncode or r.stdout.strip() != "1":
        raise ValueError("Container is absent or not owned by this package")


def launch(args):
    item = load_resolved(args.resolved, args.arm)
    if item["config"]["status"] == "experimental_hardware_gate" and not args.experimental:
        raise ValueError("Experimental arm needs explicit --experimental and separate smoke/quality qualification")
    root = Path(args.weights).resolve()
    token = uuid.uuid4().hex
    cmd = launch_command(item, root, args.gpus, args.port, args.name, Path(args.profile).resolve() if args.profile else None, token)
    print(json.dumps({"command": cmd, "executed": args.execute}, indent=2))
    if not args.execute:
        return
    if args.timeout <= 0:
        raise ValueError("Launch timeout must be positive")
    deadline = time.monotonic() + args.timeout

    def remaining(command):
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise subprocess.TimeoutExpired(command, args.timeout)
        return seconds

    def bounded_run(command, cap=30):
        return run(command, timeout=min(cap, remaining(command)))

    receipt_sha = verify_staging(root, item["config"])
    # Reject occupied GPUs and ports instead of stopping any host workload.
    apps = bounded_run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
    inventory = bounded_run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"])
    if apps.returncode or inventory.returncode:
        raise ValueError("GPU process inventory failed")
    selected = set(args.gpus.split(","))
    available = {}
    for line in inventory.stdout.splitlines():
        idx, gpu_uuid, gpu_name = [x.strip() for x in line.split(",", 2)]
        available[idx] = gpu_uuid
        if idx in selected and item["config"]["hardware"] == "B200" and "B200" not in gpu_name:
            raise ValueError("Selected GPU is not B200")
    if not selected <= available.keys() or any(available[x] in apps.stdout for x in selected):
        raise ValueError("Selected GPU absent or occupied; coordinate ownership before launch")
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", args.port))
    if args.profile:
        Path(args.profile).mkdir(parents=True, exist_ok=False)
        ns = bounded_run(["docker", "run", "--rm", "--entrypoint", "nsys", item["image"], "--version"])
        if ns.returncode: raise ValueError("Nsight Systems absent from selected image; profile arm is blocked, no installation attempted")
    # Save actual CLI flags/version before launch, with no GPU allocation.
    help_cmd = ["docker", "run", "--rm", "--pull=never", "--entrypoint", "sglang" if item["config"]["engine"] == "sglang" else "vllm", item["image"], "serve", "--help"]
    help_result = bounded_run(help_cmd, cap=180)
    result_dir = Path(args.out)
    result_dir.mkdir(parents=True, exist_ok=False)
    (result_dir / "engine-help.txt").write_text(help_result.stdout)
    (result_dir / "engine-help-stderr.txt").write_text(help_result.stderr)
    if help_result.returncode or any(flag not in help_result.stdout for flag in item["config"]["flags"] if flag.startswith("--")):
        raise ValueError("Pinned image CLI lacks required flags; preserve help and skip arm rather than editing flags silently")
    versions = bounded_run(["docker", "run", "--rm", "--entrypoint", "python", item["image"], "-c",
        "import importlib.metadata as m,json,torch; print(json.dumps({'packages':{p:m.version(p) for p in ['torch','" + item["config"]["engine"] + "']},'cuda':torch.version.cuda,'nccl':torch.cuda.nccl.version()}))"], cap=90)
    (result_dir / "engine-versions.txt").write_text(versions.stdout + versions.stderr)
    write_json(result_dir / "launch.json", {"utc": utc(), "resolved": item, "command": cmd,
               "staging_sha256": receipt_sha, "profiled": bool(args.profile), "gpus": args.gpus})
    try:
        result = bounded_run(cmd, cap=180)
        if result.returncode:
            (result_dir / "launch-error.txt").write_text(result.stderr[:16384])
            raise ValueError("Launch failed; saved diagnostic")
        (result_dir / "container-id.txt").write_text(result.stdout.strip() + "\n")
        from infrx_bench.harness import wait_ready
        wait_ready([f"http://127.0.0.1:{args.port}"], item["config"]["served_model"], remaining(cmd))
        logs = bounded_run(["docker", "logs", "--timestamps", "--tail", "1000", args.name])
        (result_dir / "startup.log").write_text(logs.stdout + logs.stderr)
        write_json(result_dir / "readiness.json", {"utc": utc(), "status": "identity_ready_generation_not_yet_proven"})
    except BaseException:
        cleanup_container(args.name, token, "launch", result_dir / "readiness-error.log")
        raise


def stop(args):
    owned(args.name)
    result = run(["docker", "stop", "--time", "30", args.name], timeout=45)
    if result.returncode:
        raise ValueError("Owned container stop failed")
    if args.remove:
        result = run(["docker", "rm", args.name])
        if result.returncode:
            raise ValueError("Owned stopped container removal failed")


def capture(args):
    owned(args.name)
    result = run(["docker", "logs", "--timestamps", "--tail", "5000", args.name], timeout=60)
    with Path(args.out).open("x") as f: f.write((result.stdout + result.stderr)[:8 * 1024 * 1024])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    for action in ("resolve", "preflight", "stage", "launch"):
        q = sub.add_parser(action)
        q.add_argument("--arm", required=True, choices=json.loads((HERE / "matrix.json").read_text())["arms"])
        q.add_argument("--out", required=True)
        if action == "preflight": q.add_argument("--work", required=True)
        if action in ("stage", "launch"):
            q.add_argument("--resolved", required=True); q.add_argument("--weights", required=True)
            q.add_argument("--execute", action="store_true"); q.add_argument("--timeout", type=int, default=3600)
        if action == "stage": q.add_argument("--forward-hf-token", action="store_true")
        if action == "launch":
            q.add_argument("--gpus", required=True); q.add_argument("--port", type=int, default=18300)
            q.add_argument("--name", required=True); q.add_argument("--experimental", action="store_true")
            q.add_argument("--profile", help="Separate profile output directory; nsys must already exist inside image. Starts a 60s capture after 600s warmup.")
    q = sub.add_parser("stop"); q.add_argument("--name", required=True); q.add_argument("--remove", action="store_true")
    q = sub.add_parser("capture"); q.add_argument("--name", required=True); q.add_argument("--out", required=True)
    args = p.parse_args()
    try:
        globals()[args.action](args)
    except (ValueError, OSError, subprocess.TimeoutExpired, KeyError) as exc:
        print("BLOCKED: " + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
