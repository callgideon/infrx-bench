#!/usr/bin/env python3
"""Run inside the staged runtime image; no network or tokenizer installation."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from infrx_bench.harness import synthetic, digest, write_json


def make_entries(encode, target, count, seed):
    entries = []
    for idx in range(count):
        lo, hi = 128, min(1048576, target * 12)
        chosen = None
        for _ in range(19):
            chars = (lo + hi) // 2
            messages = synthetic(idx, chars, seed)
            tokens = len(encode(messages[0]["content"]))
            chosen = {"id": f"tokens{target}-{idx}", "messages": messages, "content_tokens": tokens,
                      "target_content_tokens": target, "prompt_sha256": digest(messages)}
            if abs(tokens - target) <= max(2, target * .005): break
            if tokens < target: lo = chars + 1
            else: hi = chars - 1
            if hi < lo: break
        if abs(chosen["content_tokens"] - target) > max(2, target * .01):
            raise ValueError("Tokenizer cannot produce target bin within 1%; no character-to-token guess substituted")
        entries.append(chosen)
    return entries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", required=True); p.add_argument("--revision", required=True)
    p.add_argument("--target", type=int, required=True, choices=(128, 1024, 2048, 8192, 32768, 65536, 131072))
    p.add_argument("--count", type=int, default=300); p.add_argument("--seed", type=int, default=20261010)
    p.add_argument("--out", required=True); p.add_argument("--engine", choices=("sglang", "vllm"), default="sglang")
    args = p.parse_args()
    if not 1 <= args.count <= 1000: raise ValueError("Corpus cap is 1000 prompts")
    if args.count*args.target*12>32*1024*1024:raise ValueError("Reduce count for long context: conservative synthetic corpus byte cap is 32 MiB")
    if not re.fullmatch(r"[0-9a-f]{40}",args.revision):raise ValueError("Exact tokenizer revision required")
    root = Path(args.tokenizer)
    receipt = json.loads((root / "staging.json").read_text())
    if receipt["revision"] != args.revision or not receipt["verified"]: raise ValueError("Tokenizer receipt mismatch")
    from . import assets, export
    assets.guarded(root)
    for item in receipt["files"]:
        assets.safe_path(item["path"])
        if export.sha(assets.guarded(root/item["path"]))!=item["sha256"]:raise ValueError("Tokenizer bytes changed")
    if args.engine == "sglang":
        from sglang.benchmark.utils import get_tokenizer
        tokenizer = get_tokenizer(str(root))
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    entries = make_entries(lambda s: tokenizer.encode(s, add_special_tokens=False), args.target, args.count, args.seed)
    if Path(args.out).exists():raise ValueError("Corpus output already exists; preserve it")
    write_json(args.out, {"schema": 1, "tokenizer_repo": receipt["repo"], "tokenizer_revision": args.revision,
        "tokenizer_metadata_sha256": {x["path"]: x["sha256"] for x in receipt["files"] if "token" in x["path"] or "template" in x["path"] or x["path"] == "config.json"},
        "content_tokens_exclude_chat_template_overhead": True, "server_usage_is_authoritative": True,
        "seed": args.seed, "entries": entries})


if __name__ == "__main__": main()
