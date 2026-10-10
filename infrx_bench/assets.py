"""Pinned, checksum-bound public assets. No download runs as an import side effect."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
import urllib.parse
import urllib.request

from . import budget, export, harness, identity

SCHEMA = "infrx-assets/1"
MAX_METADATA = 16 * 1024 * 1024


def safe_path(name):
    path = PurePosixPath(name)
    if not isinstance(name, str) or not name or path.is_absolute() or ".." in path.parts or str(path) != name or "\\" in name:
        raise ValueError("Asset path must be a canonical relative POSIX path")
    if any(part.startswith(".") for part in path.parts):
        raise ValueError("Hidden asset paths are not supported")
    return path


def validate(manifest, fixture=False):
    if manifest.get("schema") != SCHEMA or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", "")):
        raise ValueError("Exact immutable revision and asset schema required")
    if not manifest.get("license") or manifest.get("access") != "public-ungated":
        raise ValueError("Known license and public ungated access required; gated access is manual")
    files = manifest.get("files", [])
    if not 1 <= len(files) <= 10000:
        raise ValueError("Bounded nonempty asset inventory required")
    paths = set()
    for item in files:
        safe_path(item["path"])
        if item["path"] in paths:
            raise ValueError("Duplicate asset path")
        paths.add(item["path"])
        if item["path"].split("/")[0] in ("download-state.json", "download.lock", "staging.json") or item["path"].endswith(".part"):
            raise ValueError("Asset path conflicts with download control files")
        if not isinstance(item["bytes"], int) or isinstance(item["bytes"], bool) or not 0 <= item["bytes"] <= 1024**4:
            raise ValueError("Asset size must be bounded integer bytes")
        if not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise ValueError("Every asset requires SHA-256")
        url = urllib.parse.urlsplit(item["url"])
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Public manifest URLs cannot carry credentials or query tokens")
        if fixture and url.scheme == "http" and url.hostname == "127.0.0.1":
            continue
        if url.scheme != "https" or url.netloc not in ("huggingface.co", "raw.githubusercontent.com"):
            raise ValueError("Asset source must be an approved official public host")
        if manifest["revision"] not in url.path.split("/"):
            raise ValueError("URL must bind the exact revision")
    for path in paths:
        if any(str(parent) in paths for parent in PurePosixPath(path).parents if str(parent) != "."):
            raise ValueError("Asset paths cannot conflict with parent directories")
    return sum(item["bytes"] for item in files)


def bounded_get(url, deadline, maximum=MAX_METADATA):
    with urllib.request.urlopen(url, timeout=min(20, budget.check(deadline))) as response:
        chunks = []
        count = 0
        while True:
            budget.check(deadline)
            chunk = response.read(min(65536, maximum + 1 - count))
            if not chunk:
                break
            count += len(chunk)
            if count > maximum:
                raise ValueError("Metadata byte cap exceeded")
            chunks.append(chunk)
    return b"".join(chunks)


def hf_manifest(repo, revision, deadline, tokenizer_only=False):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Canonical public repository and exact revision required")
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    raw = bounded_get(url, deadline)
    metadata = json.loads(raw)
    if metadata.get("sha") != revision or metadata.get("gated") not in (False, None) or metadata.get("private"):
        raise ValueError("Revision mismatch or gated/private repository; access must be resolved manually")
    license_name = metadata.get("cardData", {}).get("license")
    if not license_name:
        raise ValueError("Model card license missing; do not invent licensing")
    files = []
    for sibling in metadata["siblings"]:
        name = sibling["rfilename"]
        if name.startswith("."):
            continue
        if tokenizer_only and not ("token" in name or "template" in name or name in ("config.json", "LICENSE", "LICENSE.txt", "README.md", "merges.txt", "vocab.json")):
            continue
        safe_path(name)
        size = sibling["size"]
        file_url = f"https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name, safe='/')}"
        sha = sibling.get("lfs", {}).get("sha256")
        if sha is None:
            if size > MAX_METADATA:
                raise ValueError("Non-LFS metadata exceeds 16 MiB; explicit checksum review required")
            data = bounded_get(file_url, deadline, maximum=MAX_METADATA)
            if len(data) != size:
                raise ValueError("Metadata size changed")
            # Git blobId is SHA-1 with a header, not a file SHA-256.
            sha = hashlib.sha256(data).hexdigest()
        files.append({"path": name, "bytes": size, "sha256": sha, "url": file_url})
    result = {"schema": SCHEMA, "repo": repo, "revision": revision, "access": "public-ungated",
              "license": license_name, "license_review_required": True, "metadata_sha256": hashlib.sha256(raw).hexdigest(),
              "metadata_url": url, "scope": "tokenizer-metadata" if tokenizer_only else "complete-checkpoint",
              "created_utc": harness.utc(), "files": files}
    result["total_bytes"] = validate(result)
    return result


def guarded(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Download paths cannot traverse symlinks")
    return path


def preflight(manifest, root, fixture=False):
    total = validate(manifest, fixture)
    root = guarded(root)
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise ValueError("Download root parent must be a directory")
    needed = 0
    for item in manifest["files"]:
        final = guarded(root/item["path"])
        partial = guarded(str(final)+".part")
        for path in (final, partial):
            if path.exists() and not path.is_file():
                raise ValueError("Asset destination is not a regular file")
        needed += 0 if final.exists() else max(0, item["bytes"] - (partial.stat().st_size if partial.exists() else 0))
    free = shutil.disk_usage(ancestor).free
    return {"total_bytes": total, "additional_bytes": needed, "free_bytes": free,
            "disk_gate_passed": free >= needed + 64 * 1024 * 1024,
            "license": manifest["license"], "license_review_required": True,
            "access": manifest["access"], "network_access_checked": False}


def download(manifest, root, seconds, maximum_bytes, license_reviewed=False, fixture=False):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not 0 < seconds <= 48*3600:
        raise ValueError("Positive bounded seconds required")
    if not isinstance(maximum_bytes, int) or isinstance(maximum_bytes, bool) or maximum_bytes < 1:
        raise ValueError("Positive integer total-byte budget required")
    report = preflight(manifest, root, fixture)
    if not license_reviewed or not report["disk_gate_passed"] or report["total_bytes"] > maximum_bytes:
        raise ValueError("Explicit license review, sufficient disk and total-byte budget required")
    root = guarded(root)
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = harness.digest(manifest)
    state_path = root/"download-state.json"
    lock = root/"download.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    try:
        if state_path.exists():
            state = identity.read_json(state_path)["data"]
            if state["manifest_sha256"] != fingerprint or state["maximum_bytes"] != maximum_bytes:
                raise ValueError("Resume must use identical manifest and byte budget")
            deadline = state["expires_at"]
            if time.time() < state["started_at"] - 1:
                raise ValueError("Clock moved behind original download start")
        else:
            started = time.time()
            deadline = started + seconds
            state = {"manifest_sha256": fingerprint, "maximum_bytes": maximum_bytes,
                     "started_at": started, "expires_at": deadline}
            with state_path.open("x") as stream:
                json.dump(state, stream, indent=2)
        budget.check(deadline)
        records = []
        for item in manifest["files"]:
            target = guarded(root/item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = guarded(str(target)+".part")
            if target.exists():
                if target.stat().st_size != item["bytes"] or export.sha(target, deadline) != item["sha256"]:
                    raise ValueError("Existing asset differs; preserve it and use a fresh root")
            else:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > item["bytes"]:
                    raise ValueError("Partial file exceeds expected size")
                if offset < item["bytes"]:
                    headers = {"Range": f"bytes={offset}-"} if offset else {}
                    request = urllib.request.Request(item["url"], headers=headers)
                    with urllib.request.urlopen(request, timeout=min(20, budget.check(deadline))) as response:
                        if offset and (response.status != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{item['bytes']-1}/{item['bytes']}"):
                            raise ValueError("Server did not honor exact resume range")
                        if not offset and response.status != 200:
                            raise ValueError("Unexpected full-download response status")
                        with partial.open("ab" if partial.exists() else "xb") as stream:
                            count = offset
                            while True:
                                budget.check(deadline)
                                chunk = response.read(min(1024*1024, item["bytes"] + 1 - count))
                                if not chunk:
                                    break
                                count += len(chunk)
                                if count > item["bytes"]:
                                    raise ValueError("Download exceeds manifest size")
                                stream.write(chunk)
                            stream.flush()
                            os.fsync(stream.fileno())
                elif not partial.exists():
                    partial.touch(exist_ok=False)
                if partial.stat().st_size != item["bytes"] or export.sha(partial, deadline) != item["sha256"]:
                    raise ValueError("Asset checksum/size failed; partial evidence retained")
                if target.exists():
                    raise ValueError("Destination appeared during transfer")
                partial.rename(target)
            records.append({"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]})
        receipt = {"schema": "infrx-staging/1", "repo": manifest["repo"], "revision": manifest["revision"],
                   "manifest_sha256": fingerprint, "license": manifest["license"], "verified": True,
                   "utc": harness.utc(), "files": records}
        budget.check(deadline)
        receipt_path = root/"staging.json"
        if receipt_path.exists():
            previous = identity.read_json(receipt_path)["data"]
            if any(previous.get(k) != receipt[k] for k in ("repo", "revision", "manifest_sha256", "files", "verified")):
                raise ValueError("Retained staging receipt differs")
            return previous
        with receipt_path.open("x") as stream:
            json.dump(receipt, stream, indent=2)
        return receipt
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    meta = sub.add_parser("manifest")
    meta.add_argument("--repo", required=True); meta.add_argument("--revision", required=True)
    meta.add_argument("--out", required=True); meta.add_argument("--tokenizer-only", action="store_true")
    meta.add_argument("--seconds", type=float, default=120)
    for name in ("preflight", "download"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", required=True); command.add_argument("--root", required=True)
        if name == "download":
            command.add_argument("--execute", action="store_true", required=True)
            command.add_argument("--license-reviewed", action="store_true", required=True)
            command.add_argument("--maximum-bytes", type=int, required=True)
            command.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    try:
        if args.action == "preflight":
            print(json.dumps(preflight(identity.read_json(args.manifest)["data"], args.root), indent=2))
        else:
            if not 0 < args.seconds <= 48*3600:
                raise ValueError("Positive bounded seconds required")
            deadline = time.time() + args.seconds
            payload = vars(args)
            if args.action == "download":
                payload["manifest_data"] = identity.read_json(args.manifest)["data"]
                state_path = guarded(args.root)/"download-state.json"
                if state_path.exists():
                    deadline = min(deadline, identity.read_json(state_path)["data"]["expires_at"])
            budget.supervised("infrx_bench.asset_worker", payload, deadline)
        return 0
    except (ValueError, OSError, KeyError, TimeoutError) as error:
        print("BLOCKED/PAUSED: "+str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
