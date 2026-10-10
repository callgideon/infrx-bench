"""Bounded multi-turn session scheduler with visible queue and batch delays."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from . import harness


def route(policy, endpoints, session, turn, sequence):
    if policy == "round_robin": return endpoints[sequence % len(endpoints)]
    key = session["id"] if policy == "session_affinity" else session.get("prefix_group", session["id"])
    index = int(hashlib.sha256(str(key).encode()).hexdigest()[:16], 16) % len(endpoints)
    return endpoints[index]


def select_ready(ready, policy, now, max_wait):
    oldest = min(ready, key=lambda j: (j["eligible"], j["session"]["id"]))
    if policy == "fcfs" or now - oldest["eligible"] >= max_wait: return oldest
    counts = {}
    for job in ready:
        key = job["session"].get("prefix_group", job["session"]["id"])
        counts[key] = counts.get(key, 0) + 1
    group = max(counts, key=lambda k: (counts[k], str(k)))
    return min((j for j in ready if j["session"].get("prefix_group", j["session"]["id"]) == group), key=lambda j: j["eligible"])


def run(driver_cfg, cell, root):
    harness.validate(driver_cfg)
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    start_wall, start = time.time(), time.monotonic()
    nonce = uuid.uuid4().hex
    cfg = {**driver_cfg, "session_trace": cell["sessions"], "queue_policy": cell.get("queue_policy", "fcfs"),
           "routing": cell.get("routing", "round_robin"), "batch_window_seconds": cell.get("batch_window_seconds", 0),
           "session_cache_protocol": cell.get("cache_protocol", "session-prefix"), "run_nonce": nonce}
    harness.write_json(root/"manifest.json", {"schema": 1, "id": root.name, "utc": harness.utc(),
        "config": cfg, "config_sha256": harness.digest(cfg), "driver_sha256": harness.driver_hash(),
        "profiled": cfg.get("profiled", False), "deployment": cfg.get("deployment"),
        "session_semantics": "Think time begins after prior response; only eligible turns are offered. Concurrency bounds active requests, not users.",
        "cache_flush_verified": False, "token_itl_observed": False})
    jobs = []
    for session in cell["sessions"]:
        prefix = cell.get("system_prompt", "You are a careful assistant.")
        if cell.get("cache_protocol") == "cold-distinct": prefix = f"Session {nonce}:{session['id']}\n" + prefix
        jobs.append({"session": session, "turn": 0, "eligible": start + session.get("arrival_seconds", 0),
            "messages": [{"role": "system", "content": prefix}], "offered": False, "first_send": None,
            "completed_turns": 0, "status": "pending"})
    futures = {}
    results = []
    interrupted = False
    sequence = 0
    with (root/"attempts.jsonl").open("x", buffering=1) as attempts, (root/"scheduled.jsonl").open("x", buffering=1) as scheduled, ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool, harness.Telemetry(root, cfg["endpoints"], cfg.get("telemetry_interval", 5),cfg.get("deadline_epoch")):
        def durable(f, row):
            f.write(json.dumps(row, allow_nan=False) + "\n"); f.flush(); os.fsync(f.fileno())
        def offer(job):
            if job["offered"]: return
            job["attempt_id"] = f"{root.name}:{job['session']['id']}:{job['turn']}"
            durable(scheduled, {"attempt_id":job["attempt_id"], "scheduled_at":start_wall+job["eligible"]-start,
                "session_id":job["session"]["id"], "turn":job["turn"]})
            job["offered"] = True
        for job in jobs: offer(job)
        try:
            while jobs or futures:
                done = [f for f in futures if f.done()]
                for f in done:
                    job = futures.pop(f)
                    row = f.result()
                    row.update({"session_id":job["session"]["id"], "tenant_id":job["session"].get("tenant", "default"),
                        "prefix_group":job["session"].get("prefix_group"), "turn":job["turn"], "routing":cfg["routing"],
                        "queue_policy":cfg["queue_policy"], "queue_delay_s": row["send_at"]-row["scheduled_at"],
                        "admitted_at":row["send_at"]})
                    # Tool execution belongs to a frozen trace/adapter, never this benchmark.
                    if row.get("tool_calls"):
                        row["status"] = "failed"; row["error"] = "tool_result_adapter_required"
                    durable(attempts, row)
                    if row["status"] != "success":
                        job["status"] = "failed"; job["ended_at"] = row["end_at"]; results.append(job); continue
                    job["completed_turns"] += 1
                    job["messages"].append({"role":"assistant", "content":row.get("content", "")})
                    job["turn"] += 1
                    if job["turn"] >= len(job["session"]["turns"]):
                        job["status"] = "success"; job["ended_at"] = row["end_at"]; results.append(job)
                    else:
                        job["eligible"] = time.monotonic() + job["session"]["turns"][job["turn"]].get("think_seconds", 0)
                        job["offered"] = False; offer(job); jobs.append(job)
                now = time.monotonic()
                if now >= start + cfg["duration"] or (cfg.get("deadline_epoch") and time.time()>=cfg["deadline_epoch"]):
                    for job in jobs:
                        offer(job)
                        durable(attempts, {"attempt_id":job["attempt_id"], "scheduled_at":start_wall+job["eligible"]-start,
                            "admitted_at":time.time(), "send_at":None, "end_at":time.time(), "status":"shed",
                            "error":"cell_admission_deadline", "usage":None, "session_id":job["session"]["id"], "turn":job["turn"]})
                        job["status"] = "deadline"; job["ended_at"] = time.time(); results.append(job)
                    jobs.clear()
                    if not futures: break
                    wait(futures, timeout=.02, return_when=FIRST_COMPLETED); continue
                ready = [j for j in jobs if j["eligible"] <= now]
                while ready and len(futures) < cfg["concurrency"]:
                    oldest = min(j["eligible"] for j in ready)
                    if now < oldest + cfg["batch_window_seconds"]: break
                    job = select_ready(ready, cfg["queue_policy"], now, cell.get("max_queue_wait_seconds", 5))
                    ready.remove(job); jobs.remove(job)
                    turn = job["session"]["turns"][job["turn"]]
                    job["messages"].append({"role":"user", "content":turn["content"]})
                    body = dict(cfg["body_extra"])
                    if cell.get("cache_protocol") == "tenant-isolation": body["cache_salt"] = "infrx-tenant:"+str(job["session"].get("tenant", "default"))
                    base = route(cfg["routing"], cfg["endpoints"], job["session"], job["turn"], sequence); sequence += 1
                    if job["first_send"] is None: job["first_send"] = time.time()
                    f = pool.submit(harness.request, base, cfg["model"], list(job["messages"]), cfg["output"], cfg["timeout"],
                        start_wall+job["eligible"]-start, job["attempt_id"], body, True,cfg.get("deadline_epoch"))
                    futures[f] = job
                if futures: wait(futures, timeout=.01, return_when=FIRST_COMPLETED)
                else: time.sleep(.005)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            # Preserve every in-flight outcome even after a local interrupt.
            for f, job in list(futures.items()):
                row = f.result()
                row.update({"session_id":job["session"]["id"], "turn":job["turn"], "admitted_at":row["send_at"]})
                durable(attempts, row)
    harness.write_json(root/"completion.json", {"start_at":start_wall, "end_at":time.time(),
        "elapsed_s":time.time()-start_wall, "interrupted":interrupted, "drain_included":True,
        "admission_started_at":start_wall,"admission_ended_at":min(time.time(),start_wall+cfg["duration"],cfg.get("deadline_epoch",time.time())),
        "absolute_deadline_reached":bool(cfg.get("deadline_epoch") and time.time()>=cfg["deadline_epoch"]),
        "stop_reason":"interrupted" if interrupted else "all_sessions_terminal", "scheduled":sequence})
    summary = harness.summarize(root)
    session_rows = [{"id":j["session"]["id"], "status":j["status"], "completed_turns":j["completed_turns"],
        "planned_turns":len(j["session"]["turns"]), "arrival_at":start_wall+j["session"].get("arrival_seconds", 0),
        "end_at":j.get("ended_at"), "first_send_at":j["first_send"]} for j in results]
    harness.write_json(root/"sessions.json", session_rows)
    summary["sessions"] = {"planned":len(cell["sessions"]), "completed":sum(j["status"] == "success" for j in results),
        "failed_or_deadline":sum(j["status"] != "success" for j in results),
        "unoffered_dependent_turns":sum(len(j["session"]["turns"])-j["completed_turns"]-1 for j in results if j["status"] != "success"),
        "completion_latency_s":harness.distribution([j["end_at"]-j["arrival_at"] for j in session_rows if j["status"] == "success"])}
    summary["cache_protocol"] = cell.get("cache_protocol", "session-prefix")
    summary["routing"] = cfg["routing"]; summary["queue_policy"] = cfg["queue_policy"]
    summary["batch_window_seconds"] = cfg["batch_window_seconds"]
    summary["quality_scope"] = "Session completion and transport checks; task correctness requires a frozen rubric."
    harness.write_json(root/"summary.json", summary)
    return summary
