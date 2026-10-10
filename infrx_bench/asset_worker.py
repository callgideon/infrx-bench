"""Watchdog-isolated public asset preparation worker."""
import json
import sys
import time
from pathlib import Path
from . import assets


def main():
    job = json.load(sys.stdin)
    if job["action"] == "manifest":
        result = assets.hf_manifest(job["repo"], job["revision"], time.time()+job["seconds"], job["tokenizer_only"])
        path = Path(job["out"])
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps({"manifest": str(path), "total_bytes": result["total_bytes"], "files":len(result["files"])}))
    else:
        print(json.dumps(assets.download(job["manifest_data"], job["root"], job["seconds"], job["maximum_bytes"], job["license_reviewed"]), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TimeoutError) as error:
        print("BLOCKED/PAUSED: "+str(error), file=sys.stderr)
        raise SystemExit(2)
