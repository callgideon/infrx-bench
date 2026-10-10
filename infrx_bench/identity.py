"""Freeze every external input before collection; resume compares byte identities."""
import copy
import hashlib
import json
from pathlib import Path

from . import config, harness

MAX_INPUT_BYTES = 64 * 1024 * 1024


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("Input must be a bounded regular JSON file, without symlinks")
    raw = path.read_bytes()
    if len(raw) > MAX_INPUT_BYTES:
        raise ValueError("Input byte cap exceeded")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes":len(raw),
            "data":json.loads(raw), "source":str(path.resolve())}


def freeze(cfg):
    cfg = copy.deepcopy(cfg)
    inputs = {"cases":read_json(cfg.get("canary_cases_file", Path(__file__).parent/"cases.json"))}
    for cell in cfg["cells"]:
        if cell.get("corpus_file"):
            inputs["corpus:"+cell["id"]] = read_json(cell["corpus_file"])
        baseline = cell.get("baseline_canary_file") or cfg.get("baseline_canary_summary")
        if baseline:
            inputs["baseline:"+cell["id"]] = read_json(baseline)
    if cfg.get("qualification_file"):
        inputs["qualification"] = read_json(cfg["qualification_file"])
        for artifact in inputs["qualification"]["data"].get("artifacts", []):
            name = artifact["kind"]
            if "qualification:"+name in inputs: raise ValueError("Duplicate qualification artifact")
            document = read_json(artifact["path"])
            if document["sha256"] != artifact["sha256"]:
                raise ValueError("Qualification artifact checksum mismatch")
            inputs["qualification:"+name] = document
    if sum(document["bytes"] for document in inputs.values())>128*1024*1024:
        raise ValueError("Aggregate frozen input cap exceeded")
    content = {"config":cfg, "inputs":inputs, "bundle_sha256":config.bundle_hash()}
    return {**content, "fingerprint":harness.digest(content)}


def entries(frozen, cell):
    key = "corpus:"+cell.get("source_cell_id", cell["id"])
    if key in frozen["inputs"]:
        document = frozen["inputs"][key]["data"]
        cfg = frozen["config"]
        if cfg["purpose"] != "cpu_mock":
            identity = cfg["identity"]
            if (document.get("tokenizer_repo"), document.get("tokenizer_revision")) != (identity["repo"], identity["revision"]):
                raise ValueError("Corpus tokenizer identity must match checkpoint")
            if not document.get("tokenizer_metadata_sha256") or document.get("server_usage_is_authoritative") is not True:
                raise ValueError("Tokenizer metadata and actual server-token accounting provenance required")
        result = document["entries"]
    elif cell.get("prompts"):
        result = [{"id":f"prompt-{i}", "messages":[{"role":"user", "content":s}]} for i,s in enumerate(cell["prompts"])]
    else:
        return None
    if not isinstance(result, list) or not 1 <= len(result) <= 10000:
        raise ValueError("Bound corpus entries")
    for entry in result:
        if not entry.get("id") or not entry.get("messages"):
            raise ValueError("Corpus entries need identity and messages")
        for message in entry["messages"]:
            if message.get("role") not in ("system", "user", "assistant") or not isinstance(message.get("content"), str):
                raise ValueError("Text-only bounded corpus messages required")
            if len(message["content"]) > 1048576: raise ValueError("Corpus message cap exceeded")
        rubric=entry.get("rubric")
        if cell.get("evidence_role")=="capacity" and not rubric:
            raise ValueError("Capacity corpus needs task correctness rubrics on served requests")
        if rubric and (rubric.get("kind") not in ("exact_text","exact_json") or "expected" not in rubric):
            raise ValueError("Supported frozen exact-text/JSON rubric required")
    return result
