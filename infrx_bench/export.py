"""Verified archive copies; failed transfers are errors, never completion markers."""
import hashlib
from pathlib import Path
import shutil
import os
from . import budget

from . import harness


def sha(path,deadline_epoch=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""):
            if deadline_epoch is not None: budget.check(deadline_epoch)
            h.update(chunk)
    return h.hexdigest()


def verify(source, destination,deadline_epoch=None):
    source, destination = Path(source), Path(destination)
    if not source.is_file() or not destination.is_file(): raise ValueError("Both archive files must exist")
    left, right = sha(source,deadline_epoch), sha(destination,deadline_epoch)
    if left != right: raise ValueError("Export hash mismatch; durable export NOT confirmed")
    return {"schema":"infrx-export/1", "verified_utc":harness.utc(), "sha256":left,
        "bytes":source.stat().st_size, "destination_verified":True,"destination":str(destination.resolve()),
        "durability":"Byte identity confirmed; destination retention is an operator-selected property."}


def copy(source, destination,deadline_epoch=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists(): raise ValueError("Export destination must be a new file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        for chunk in iter(lambda:src.read(1024*1024),b""):
            if deadline_epoch is not None: budget.check(deadline_epoch)
            dst.write(chunk)
        dst.flush(); os.fsync(dst.fileno())
    receipt = verify(source, destination,deadline_epoch)
    harness.write_json(str(destination)+".receipt.json", receipt)
    return receipt
