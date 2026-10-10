"""Isolated client worker; stdin carries private config without shell interpolation."""
import json
import os
from pathlib import Path
import sys

from . import budget, cli, export, harness


def main():
    job=json.load(sys.stdin)
    operation=job["operation"]
    if operation=="collect":
        if job["config"]["purpose"]=="cpu_mock":os.environ.pop("BENCH_API_KEY",None)
        cli.execute(job["config"],job["out"],mock_token=job.get("mock_token"),prepared=True)
        return
    budget.check(job["deadline"])
    source=Path(job["source"]);destination=Path(job["destination"])
    if operation=="archive":
        if not source.is_dir():raise ValueError("Archive source must be a directory")
        harness.archive(source,destination,job["deadline"])
        result={"archive":str(destination),"sha256":export.sha(destination,job["deadline"])}
    elif operation=="export":result=export.copy(source,destination,job["deadline"])
    elif operation=="verify-export":result=export.verify(source,destination,job["deadline"])
    else:raise ValueError("Unsupported worker operation")
    budget.check(job["deadline"]);print(json.dumps(result,indent=2))


if __name__=="__main__":
    try:main()
    except (ValueError,OSError,KeyError,TimeoutError) as error:
        print("BLOCKED/PAUSED: "+str(error),file=sys.stderr);raise SystemExit(2)
