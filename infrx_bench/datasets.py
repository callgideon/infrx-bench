"""Deterministic local task import and optional offline tokenizer preparation."""
import argparse
import copy
import json
from pathlib import Path
import random
import re

from . import assets, export, harness, identity


def import_tasks(manifest, source, count, seed):
    assets.validate(manifest)
    if manifest.get("kind") != "dataset" or not manifest.get("split"):
        raise ValueError("Dataset split and provenance required")
    if len(manifest["files"]) != 1:
        raise ValueError("Task importer accepts one frozen source file")
    path = assets.guarded(source)
    expected = manifest["files"][0]
    if not path.is_file() or path.stat().st_size > identity.MAX_INPUT_BYTES:
        raise ValueError("Bounded regular local dataset required")
    if path.stat().st_size != expected["bytes"] or export.sha(path) != expected["sha256"]:
        raise ValueError("Dataset bytes differ from pinned manifest")
    raw = path.read_text()
    if manifest.get("format") == "gsm8k-jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        tasks = []
        for index, row in enumerate(rows):
            parts = row["answer"].rsplit("####", 1)
            answer = parts[-1].strip().replace(",", "")
            if len(parts) != 2 or not re.fullmatch(r"-?\d+(?:\.\d+)?", answer):
                raise ValueError("GSM8K final numeric answer extraction failed")
            tasks.append({"id": f"gsm8k-{manifest['split']}-{index}",
                          "messages": [{"role": "user", "content": row["question"]+"\nReturn only the final numeric answer, with no explanation or commas."}],
                          "rubric": {"kind": "exact_text", "expected": answer}})
        transformation = "Final answer after ####, strip commas; explicit numeric-only instruction; exact-text instruction-following score, not official GSM8K evaluation"
    elif manifest.get("format") == "infrx-tasks-json":
        document = json.loads(raw)
        tasks = document["entries"]
        transformation = "Frozen exact-text/JSON tasks copied without changing prompts or rubrics"
    else:
        raise ValueError("Unsupported dataset format; never execute generated code or untrusted rubrics")
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= min(len(tasks), 10000):
        raise ValueError("Bounded task count required")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Integer sampling seed required")
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("Duplicate source task IDs")
    selected = [copy.deepcopy(tasks[i]) for i in sorted(random.Random(seed).sample(range(len(tasks)), count))]
    for task in selected:
        if task.get("rubric", {}).get("kind") not in ("exact_text", "exact_json") or "expected" not in task["rubric"]:
            raise ValueError("Frozen supported rubric required")
        if not task.get("messages") or any(message.get("role") not in ("system", "user", "assistant") or not isinstance(message.get("content"), str) or len(message["content"]) > 256000 for message in task["messages"]):
            raise ValueError("Bounded text task messages required")
    return {"schema": "infrx-taskset/1", "seed": seed, "entries": selected,
            "provenance": {"repo": manifest["repo"], "revision": manifest["revision"], "split": manifest["split"],
                           "license": manifest["license"], "source_sha256": expected["sha256"],
                           "manifest_sha256": harness.digest(manifest), "transformation": transformation,
                           "selected_ids": [task["id"] for task in selected]},
            "tokenizer_qualification": "NOT_TOKENIZED"}


def tokenize(tasks, tokenizer_root, maximum_context, output_reserve):
    root = assets.guarded(tokenizer_root)
    receipt = identity.read_json(root/"staging.json")["data"]
    if receipt.get("verified") is not True:
        raise ValueError("Verified tokenizer receipt required")
    for item in receipt["files"]:
        assets.safe_path(item["path"])
        if export.sha(assets.guarded(root/item["path"])) != item["sha256"]:
            raise ValueError("Tokenizer bytes changed")
    if not 1 <= output_reserve < maximum_context <= 1048576:
        raise ValueError("Context must reserve bounded output tokens")
    # Operator supplies an already installed tokenizer dependency in an audited runtime.
    # No package installation, remote code trust or network fallback is permitted here.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    result = copy.deepcopy(tasks)
    for entry in result["entries"]:
        content_tokens = sum(len(tokenizer.encode(message["content"], add_special_tokens=False)) for message in entry["messages"])
        chat_tokens = len(tokenizer.apply_chat_template(entry["messages"], tokenize=True, add_generation_prompt=True))
        if chat_tokens + output_reserve > maximum_context:
            raise ValueError("Task plus chat template/output reserve exceeds configured context")
        entry.update({"content_tokens": content_tokens, "chat_template_prompt_tokens": chat_tokens,
                      "prompt_sha256": harness.digest(entry["messages"])})
    result.update({"tokenizer_repo": receipt["repo"], "tokenizer_revision": receipt["revision"],
                   "tokenizer_metadata_sha256": {item["path"]:item["sha256"] for item in receipt["files"] if "token" in item["path"] or "template" in item["path"] or item["path"] == "config.json"},
                   "server_usage_is_authoritative": True, "tokenizer_qualification": "OFFLINE_LOCAL_CHAT_TEMPLATE",
                   "configured_context_max_tokens": maximum_context, "output_reserve_tokens": output_reserve})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    command = sub.add_parser("import")
    command.add_argument("--manifest", required=True); command.add_argument("--source", required=True)
    command.add_argument("--count", type=int, required=True); command.add_argument("--seed", type=int, required=True)
    command.add_argument("--out", required=True)
    command = sub.add_parser("tokenize")
    command.add_argument("--tasks", required=True); command.add_argument("--tokenizer-root", required=True)
    command.add_argument("--maximum-context", type=int, required=True); command.add_argument("--output-reserve", type=int, required=True)
    command.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        if args.action == "import":
            result = import_tasks(identity.read_json(args.manifest)["data"], args.source, args.count, args.seed)
        else:
            result = tokenize(identity.read_json(args.tasks)["data"], args.tokenizer_root, args.maximum_context, args.output_reserve)
        with Path(args.out).open("x") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps({"out": args.out, "entries": len(result["entries"])}))
        return 0
    except (ValueError, OSError, KeyError, ImportError) as error:
        print("BLOCKED: "+str(error)); return 2


if __name__ == "__main__":
    raise SystemExit(main())
