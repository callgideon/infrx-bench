#!/usr/bin/env python3
"""Checkpointed, bounded collection queue. Plan-only unless --execute is supplied."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time
import uuid

if __package__:
    from . import harness, node
else:
    import harness
    import node


def check_window(now, start, end, reserve=1800):
    if not start <= now < end - reserve:
        raise ValueError("Outside confirmed collection window/export reserve")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=("native-throughput", "native-latency", "nvfp4-b200", "qwen-smoke", "qwen-smoke-vllm031", "qwen-smoke-sglang"))
    p.add_argument("--endpoint", action="append", required=True); p.add_argument("--deployment", action="append", required=True)
    p.add_argument("--corpus", action="append", help="TARGET=corpus.json, e.g. 1024=/data/corpus-1024.json")
    p.add_argument("--out", required=True); p.add_argument("--ttft-slo", type=float, required=True); p.add_argument("--e2e-slo", type=float, required=True)
    p.add_argument("--budget-seconds", type=float, default=28800); p.add_argument("--execute", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Small-model rehearsal outside B200 reservation, <=600s total")
    p.add_argument("--resume", action="store_true"); p.add_argument("--repeats", type=int, default=1, choices=(1,3))
    p.add_argument("--body", help="Capacity request body; DeepSeek defaults to no thinking plus forced length, Qwen smoke disables thinking")
    args = p.parse_args()
    args.body = args.body or ('{"chat_template_kwargs":{"enable_thinking":false}}' if args.smoke else '{"reasoning_effort":"none","ignore_eos":true}')
    cfg = node.arm(args.arm)
    deployment = [json.loads(Path(x).read_text()) for x in args.deployment]
    if any(x["resolved"]["arm"] != args.arm or x.get("profiled") for x in deployment):
        raise ValueError("Each deployment must attest this unprofiled arm")
    if args.smoke != args.arm.startswith("qwen-smoke"):
        raise ValueError("Only qwen-smoke uses the out-of-window smoke queue")
    matrix = json.loads((node.HERE / "matrix.json").read_text())
    start = dt.datetime.fromisoformat(matrix["reservation"]["start"]).timestamp()
    end = dt.datetime.fromisoformat(matrix["reservation"]["end"]).timestamp()
    if args.smoke and args.budget_seconds > 600:
        raise ValueError("Small-model rehearsal budget must be <=600 seconds")
    corpus = {}
    for item in args.corpus or []:
        target, path = item.split("=", 1)
        data = json.loads(Path(path).read_text())
        if data["tokenizer_revision"] != cfg["revision"] or data["tokenizer_repo"] != cfg["repo"]:
            raise ValueError("Corpus tokenizer identity differs from deployment")
        corpus[int(target)] = data["entries"]
    shapes = [(128, 16)] if args.smoke else [(1024,256), (8192,1024), (32768,256)]
    if not args.smoke and any(target not in corpus for target, _ in shapes):
        raise ValueError("Prepare all three tokenized corpora before the B200 collection queue")
    concurrencies = [1,2] if args.smoke else ([1,4,16] if args.arm == "nvfp4-b200" else [1,4,16,64] if args.arm == "native-latency" else [1,4,16,64,256])
    duration, timeout, requests = (1., 5., 4) if args.smoke else (180., 120., 300)
    # Screen the shapes; retain repetitions at the observed knee later, not every cross-product.
    plans = [{"id": f"shape{target}-c{c}", "shape": target, "output": output, "concurrency": c, "rate": None}
             for target, output in shapes for c in concurrencies]
    plan = {"arm": args.arm, "model": cfg["served_model"], "window": matrix["reservation"], "cells": plans,
            "screen_worst_case_s": len(plans)*(duration+timeout), "followup": "four Poisson rates at 0.5/0.8/1.0/1.2 observed best request capacity per shape; optional three retained-knee repeats",
            "budget_s": args.budget_seconds, "remaining_billed_time_includes_idle": True}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute: return 0
    if not args.smoke: check_window(time.time(), start, end)
    root = Path(args.out); root.mkdir(parents=True, exist_ok=args.resume)
    quality_cases = json.loads((harness.HERE/"cases.json").read_text())
    queue_cfg = {"arm": args.arm, "endpoints": args.endpoint, "deployment": deployment, "corpus_hashes": {str(k):harness.digest(v) for k,v in corpus.items()},
                 "cases_sha256": harness.digest(quality_cases), "driver_sha256": harness.driver_hash(),
                 "slo": [args.ttft_slo, args.e2e_slo], "body": json.loads(args.body), "repeats": args.repeats, "smoke": args.smoke}
    if args.resume and (root/"queue.json").exists():
        old = json.loads((root/"queue.json").read_text())
        if old["queue_config"] != queue_cfg: raise ValueError("Resume requires the same frozen queue inputs")
    else: harness.write_json(root/"queue.json", {"utc":harness.utc(), "queue_config":queue_cfg, "plan":plan})
    deadline = min(time.monotonic() + args.budget_seconds, time.monotonic() + end - time.time() - 1800) if not args.smoke else time.monotonic()+args.budget_seconds
    base = {"endpoints": [harness.endpoint(x) for x in args.endpoint], "model":cfg["served_model"], "deployment":deployment,
            "requests":requests, "duration":duration, "timeout":timeout, "prompt_chars":512, "seed":20261010,
            "ttft_slo":args.ttft_slo, "e2e_slo":args.e2e_slo, "body_extra":json.loads(args.body), "cache_protocol":"distinct-prefix"}
    harness.wait_ready(base["endpoints"], base["model"], 60)
    def cell(item, cases=None):
        dest = root/item["id"]
        run = {**base, "output":item["output"], "concurrency":item["concurrency"], "rate":item["rate"]}
        if cases:
            run["requests"] = max(requests, len(cases))
            run["body_extra"] = {"chat_template_kwargs":{"enable_thinking":False}} if args.smoke else {"reasoning_effort":"none"}
        entries = corpus.get(item["shape"]) if not cases else None
        if args.resume:
            dest, previous = harness.resume_cell(dest, run, cases, entries)
            if previous is not None:
                if previous["missing_usage_attempts"] or previous["invalid_protocol_attempts"] or previous["counts"]["failed"] > max(3, previous["attempts"]//10):
                    raise ValueError("Retained failure requires diagnosis and a new queue; resume cannot erase it")
                return previous
        if time.monotonic() + duration + timeout > deadline:
            raise ValueError("Queue budget/export reserve reached; existing artifacts retained")
        summary = harness.run_cell(run, dest, cases=cases, corpus=entries)
        harness.archive(dest, root/(dest.name+".tar.gz"))
        print(json.dumps({"cell":dest.name, "counts":summary["counts"], "output_tokens_per_s":summary["output_tokens_per_s"]}), flush=True)
        if summary["missing_usage_attempts"] or summary["invalid_protocol_attempts"] or summary["counts"]["failed"] > max(3, summary["attempts"]//10):
            raise ValueError("Accounting/protocol/failure stop: preserve artifacts and re-smoke before continuing")
        return summary
    # Natural-completion quality diagnostic is separate from ignore-EOS capacity.
    cell({"id":"quality-before", "shape":128, "output":16 if args.smoke else 256, "concurrency":1, "rate":None},
         quality_cases)
    best = {}
    for item in plans:
        summary = cell(item)
        if summary["valid_capacity_evidence"] and summary["requests_per_s"] and summary["counts"]["success"] >= (1 if args.smoke else 20):
            old = best.get(item["shape"])
            if old is None or summary["goodput_requests_per_s"] > old[1]["goodput_requests_per_s"]: best[item["shape"]] = (item, summary)
    for target, (knee, summary) in best.items():
        for factor in (.5,.8,1.,1.2):
            item = {**knee, "id": f"shape{target}-arrival{factor}", "rate": min(1000., factor*summary["requests_per_s"])}
            cell(item)
        for repeat in range(1, args.repeats+1):
            cell({**knee, "id":f"shape{target}-retained-r{repeat}"})
    cell({"id":"quality-after", "shape":128, "output":16 if args.smoke else 256, "concurrency":1, "rate":None},
         quality_cases)
    completion_path = root/("resume-complete-"+uuid.uuid4().hex[:8]+".json") if (root/"queue-complete.json").exists() else root/"queue-complete.json"
    harness.write_json(completion_path, {"utc":harness.utc(), "status":"collection_complete_quality_review_pending", "best_screen_cells":{str(k):v[0]["id"] for k,v in best.items()}})
    return 0


if __name__ == "__main__":
    try: sys.exit(main())
    except (ValueError, OSError, KeyError) as exc:
        print("BLOCKED/PAUSED: "+str(exc), file=sys.stderr); sys.exit(2)
