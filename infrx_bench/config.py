"""Versioned experiment inputs; proposed matrices are never implicit defaults."""
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import copy

from . import harness

SCHEMA = "infrx-bench/2"


def bundle_hash():
    h = hashlib.sha256()
    for f in sorted([*Path(__file__).parent.glob("*.py"), *Path(__file__).parent.glob("*.json")]):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def finite(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")


def load(path, execution=False):
    path = Path(path)
    cfg = json.loads(path.read_text())
    validate(cfg, execution)
    return cfg


def validate(cfg, execution=False):
    if cfg.get("schema") != SCHEMA:
        raise ValueError("Unsupported experiment schema")
    if cfg.get("purpose") not in ("cpu_mock", "local_rehearsal", "b200_research"):
        raise ValueError("Explicit experiment purpose required")
    from . import policy
    policy.candidate(cfg)
    if cfg.get("arm_role") not in ("baseline", "optimized", "proxy", "mock"):
        raise ValueError("Mandatory explicit arm role")
    if not cfg.get("canary"):
        raise ValueError("Every arm needs per-configuration pre/post canaries")
    identity = cfg["identity"]
    if not identity.get("served_model"):
        raise ValueError("served_model required")
    endpoints = cfg["endpoints"]
    if not isinstance(endpoints, list) or not 1 <= len(endpoints) <= 8:
        raise ValueError("Select 1..8 endpoint origins")
    for e in endpoints: harness.endpoint(e)
    finite(cfg["budget_seconds"], "budget_seconds", .01, 48 * 3600)
    finite(cfg.get("export_reserve_seconds", 0), "export_reserve_seconds", 0, 7200)
    if cfg.get("export_reserve_seconds",0) >= cfg["budget_seconds"]:
        raise ValueError("Export reserve must leave a collection window")
    if cfg.get("end_utc"):
        end = dt.datetime.fromisoformat(cfg["end_utc"].replace("Z", "+00:00"))
        if end.tzinfo is None: raise ValueError("end_utc requires timezone")
    cells = cfg["cells"]
    if not isinstance(cells, list) or not 1 <= len(cells) <= 100:
        raise ValueError("Select 1..100 explicit cells")
    cells = expand(cfg)
    ids = set()
    worst = 0
    for cell in cells:
        cid = cell["id"]
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", cid) or cid in ids:
            raise ValueError("Cell IDs must be unique safe names")
        ids.add(cid)
        if cell.get("kind") not in ("load", "quality", "sessions"):
            raise ValueError("Unsupported cell kind")
        finite(cell["duration_seconds"], "duration_seconds", .001, 3600)
        finite(cell["timeout_seconds"], "timeout_seconds", .001, 600)
        finite(cell["output_tokens"], "output_tokens", 1, 8192)
        finite(cell["concurrency"], "concurrency", 1, 256)
        finite(cell["requests"], "requests", 1, 10000)
        if any(not isinstance(cell[k], int) for k in ("output_tokens", "concurrency", "requests")):
            raise ValueError("Integer output/concurrency/request caps required")
        for key in ("ttft_seconds", "e2e_seconds"):
            finite(cell["slo"][key], key, .001, 7200)
        finite(cell.get("max_error_fraction", 0), "max_error_fraction", 0, 1)
        if "minimum_slo_fraction" not in cell: raise ValueError("Explicit nonzero SLO acceptance fraction required")
        finite(cell["minimum_slo_fraction"], "minimum_slo_fraction", .01, 1)
        if cell.get("required_capacity_tail", "p99") not in ("p95", "p99"):
            raise ValueError("Capacity tail must be p95 or p99")
        if cell.get("arrival_rate") is not None:
            finite(cell["arrival_rate"], "arrival_rate", .001, 1000)
        if cell.get("cache_protocol", "cold-distinct") not in ("cold-distinct", "warm-replay", "session-prefix", "tenant-isolation"):
            raise ValueError("Unknown cache protocol")
        if cell.get("routing", "round_robin") not in ("round_robin", "session_affinity", "prefix_affinity"):
            raise ValueError("Unknown routing policy")
        if cell.get("queue_policy", "fcfs") not in ("fcfs", "prefix_group"):
            raise ValueError("Unknown queue policy")
        finite(cell.get("batch_window_seconds", 0), "batch_window_seconds", 0, 60)
        finite(cell.get("telemetry_interval_seconds", 5), "telemetry_interval_seconds", .01, 60)
        finite(cell.get("max_queue_wait_seconds", 5), "max_queue_wait_seconds", 0, 600)
        finite(cell.get("minimum_stability_seconds", 60), "minimum_stability_seconds", .01, 3600)
        finite(cell.get("maximum_queue_slope_per_second", 0), "maximum_queue_slope_per_second", 0, 10000)
        finite(cell.get("maximum_final_queue", 0), "maximum_final_queue", 0, 100000)
        if cell.get("kind") == "sessions":
            trace = cell.get("sessions", [])
            if not trace or len(trace) > cell["requests"]:
                raise ValueError("Sessions require bounded inline traces")
            session_ids = set()
            total_turns = 0
            for session in trace:
                if session["id"] in session_ids: raise ValueError("Duplicate session ID")
                session_ids.add(session["id"])
                finite(session.get("arrival_seconds", 0), "arrival_seconds", 0, cell["duration_seconds"])
                turns = session["turns"]
                if not 1 <= len(turns) <= 100: raise ValueError("Bound each session to 1..100 turns")
                total_turns += len(turns)
                for turn in turns:
                    if not isinstance(turn["content"], str) or not turn["content"] or len(turn["content"]) > 256000:
                        raise ValueError("Turn content must be bounded nonempty text")
                    finite(turn.get("think_seconds", 0), "think_seconds", 0, 3600)
            if total_turns > cell["requests"]: raise ValueError("Session turns exceed request cap")
        if cell.get("kind") == "quality":
            if "minimum_quality_fraction" not in cell: raise ValueError("Explicit positive quality floor required")
            finite(cell["minimum_quality_fraction"], "minimum_quality_fraction", .01, 1)
        if not isinstance(cell.get("thinking_mode"),str) or not cell["thinking_mode"]:
            raise ValueError("Explicit qualified thinking mode required")
        sampling=cell.get("sampling")
        if not isinstance(sampling,dict) or "temperature" not in sampling or set(sampling)-{"temperature","top_p","seed","frequency_penalty","presence_penalty"}:
            raise ValueError("Explicit bounded sampling controls required")
        finite(sampling["temperature"], "temperature", 0, 2)
        if "top_p" in sampling: finite(sampling["top_p"],"top_p",.001,1)
        if "seed" in sampling and (not isinstance(sampling["seed"],int) or isinstance(sampling["seed"],bool)):
            raise ValueError("Integer sampling seed required")
        for key in ("frequency_penalty","presence_penalty"):
            if key in sampling: finite(sampling[key],key,-2,2)
        body = cell.get("body", {})
        if not isinstance(body, dict) or {"model", "messages", "stream", "max_tokens", "stream_options", "cache_salt", *sampling} & body.keys():
            raise ValueError("Body cannot override identity, streaming, bounds or isolation")
        if cell.get("cache_protocol") == "tenant-isolation" and not cfg.get("capabilities", {}).get("cache_salt_verified"):
            raise ValueError("Tenant isolation requires actual cache_salt capability evidence")
        cost = cell["duration_seconds"] + cell["timeout_seconds"] + 3
        if cell.get("cache_protocol") == "warm-replay": cost *= 2
        worst += cost
    if worst > cfg["budget_seconds"]-cfg.get("export_reserve_seconds",0):
        raise ValueError("Declared worst-case cells exceed total wall budget")
    if not execution: return
    if cfg["purpose"] == "cpu_mock": return
    if identity.get("repo") == "deepseek-ai/DeepSeek-V4.1-Flash" and cfg["purpose"] != "b200_research":
        raise ValueError("Full DeepSeek cannot be a small-host proxy rehearsal")
    if not re.fullmatch(r"[0-9a-f]{40}", identity.get("revision", "")):
        raise ValueError("Execution requires an exact model revision")
    runtime = cfg["runtime"]
    if runtime.get("engine") not in ("vllm", "sglang", "external_openai"):
        raise ValueError("Unsupported runtime; never infer V4.1 support from V4")
    image = runtime.get("image_digest", "")
    source = runtime.get("source_commit", "")
    if not (re.fullmatch(r".+@sha256:[0-9a-f]{64}", image) or re.fullmatch(r"[0-9a-f]{40}", source)):
        raise ValueError("Execution requires immutable image or runtime source identity")
    reviewed = dt.datetime.fromisoformat(runtime["reviewed_utc"].replace("Z", "+00:00"))
    if reviewed.tzinfo is None: raise ValueError("Runtime review requires timezone")
    age = (dt.datetime.now(dt.timezone.utc) - reviewed).total_seconds()
    if not 0 <= age <= runtime.get("max_review_age_days", 7) * 86400:
        raise ValueError("Runtime support review stale or future-dated")
    if not runtime.get("support_evidence"):
        raise ValueError("Exact model/runtime/hardware support evidence required")
    if cfg["purpose"] == "b200_research":
        if cfg.get("parameters_reconciled") is not True or not cfg.get("end_utc"):
            raise ValueError("Reconciled parameters and reservation end_utc required before B200 collection")
        if not cfg.get("workload_provenance"):
            raise ValueError("Representative workload provenance required")
        if not cfg.get("export_destination") or not cfg.get("export_retention_evidence"):
            raise ValueError("Operator-selected durable export destination and retention evidence required")
        for cell in cfg["cells"]:
            if cell["kind"] == "load" and not cell.get("corpus_file"):
                raise ValueError("B200 load requires an explicit tokenizer-backed corpus file")
    if not cfg.get("qualification_file"):
        raise ValueError("Real collection requires bound runtime/hardware/weights/smoke evidence")
    if cfg["arm_role"] == "optimized":
        for cell in cfg["cells"]:
            if not (cell.get("baseline_canary_file") or cfg.get("baseline_canary_summary")):
                raise ValueError("Each optimized configuration needs a matched retained baseline")
        finite(cfg.get("maximum_quality_degradation"),"maximum_quality_degradation",0,1)


def driver_config(cfg, cell):
    return {
        "endpoints": cfg["endpoints"], "model": cfg["identity"]["served_model"],
        "requests": cell["requests"], "duration": cell["duration_seconds"],
        "timeout": cell["timeout_seconds"], "output": cell["output_tokens"],
        "concurrency": cell["concurrency"], "rate": cell.get("arrival_rate"),
        "seed": cfg.get("seed", 20261010), "prompt_chars": cell.get("prompt_chars", 512),
        "ttft_slo": cell["slo"]["ttft_seconds"], "e2e_slo": cell["slo"]["e2e_seconds"],
        "body_extra": {**cell.get("body", {}), **cell["sampling"]}, "profiled": cell.get("profiled", False),
        "cache_protocol": "distinct-prefix" if cell.get("cache_protocol", "cold-distinct") == "cold-distinct" else "warm-replay",
        "deployment": cfg.get("deployment_manifest"),
        "telemetry_interval": cell.get("telemetry_interval_seconds", 5),
        "package_sha256": bundle_hash(), "experiment_identity": cfg["identity"],
    }


def contract(cell):
    return {"thinking_mode":cell["thinking_mode"], "sampling":cell["sampling"],
            "thinking_body":{k:v for k,v in cell.get("body",{}).items() if k != "ignore_eos"},
            "output_tokens":cell["output_tokens"], "configured_context_max_tokens":cell.get("configured_context_max_tokens")}


def expand(cfg):
    """Bracket every configuration/repetition with canaries derived from its controls."""
    result=[]
    for source in cfg["cells"]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}",source["id"]):
            raise ValueError("Source cell ID must leave room for canary/repetition suffixes")
        repetitions=source.get("repetitions",1)
        if not isinstance(repetitions,int) or isinstance(repetitions,bool) or not 1<=repetitions<=5:
            raise ValueError("Bound repetitions to 1..5")
        if source.get("evidence_role","pilot") not in ("pilot","capacity"):
            raise ValueError("Select pilot or capacity evidence role")
        if source.get("evidence_role")=="capacity":
            if source["kind"]!="load" or repetitions<3 or source.get("arrival_rate") is None or source.get("profiled"):
                raise ValueError("Capacity requires >=3 unprofiled open-loop repetitions")
            if source.get("minimum_task_quality_fraction") is None:
                raise ValueError("Capacity requires an explicit served-workload task quality floor")
            finite(source["minimum_task_quality_fraction"],"minimum_task_quality_fraction",.01,1)
            if not source.get("queue_metric") or source["duration_seconds"]<source.get("minimum_stability_seconds",60)+source.get("telemetry_interval_seconds",5):
                raise ValueError("Capacity needs a qualified queue metric and sufficient admission window")
        for repeat in range(repetitions):
            cell=copy.deepcopy(source)
            cell["source_cell_id"]=source["id"]
            cell["id"]=source["id"]+(f"-r{repeat+1}" if repetitions>1 else "")
            for side in ("before","after"):
                canary={**copy.deepcopy(cfg["canary"]),"id":cell["id"]+"-canary-"+side,"kind":"quality",
                        "source_cell_id":source["id"], "for_cell":cell["id"],"side":side,
                        "output_tokens":cell["output_tokens"],"thinking_mode":cell["thinking_mode"],
                        "sampling":copy.deepcopy(cell["sampling"]),
                        "body":{k:v for k,v in cell.get("body",{}).items() if k != "ignore_eos"},
                        "configured_context_max_tokens":cell.get("configured_context_max_tokens"),
                        "contract":contract(cell)}
                if side=="before": result.extend([canary,cell])
                else: result.append(canary)
    return result
