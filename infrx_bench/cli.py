"""Config-driven collection; runtime/hardware evidence and final canaries fail closed."""
import argparse
import json
from pathlib import Path
import re
import sys
import time
import uuid

from . import analysis, budget, config, export, fixture, harness, identity, policy, preflight, sessions


def queue_stability(root, cell):
    metric=cell.get("queue_metric")
    if cell.get("arrival_rate") is None or not metric:
        return {"status":"unknown","reason":"Qualified open-loop queue metric required"}
    completion=json.loads((Path(root)/"completion.json").read_text())
    beginning=completion.get("admission_started_at",completion["start_at"])
    end=completion.get("admission_ended_at",completion["end_at"])
    points=[]
    for line in (Path(root)/"telemetry.jsonl").read_text().splitlines():
        row=json.loads(line)
        if not beginning<=row["at"]<=end: continue
        values=[]
        for raw in row["metrics"].values():
            if not isinstance(raw,str):continue
            for entry in raw.splitlines():
                if re.match(r"^"+re.escape(metric)+r"(?:\{|\s)",entry):
                    try:values.append(float(entry.split()[-1]))
                    except ValueError:pass
        if values:points.append((row["at"],sum(values)))
    if len(points)<5 or points[-1][0]-points[0][0]<cell.get("minimum_stability_seconds",60):
        return {"status":"unknown","reason":"Insufficient admission-period queue samples","samples":len(points)}
    xs=[x-points[0][0] for x,y in points];ys=[y for x,y in points]
    mx=sum(xs)/len(xs);my=sum(ys)/len(ys)
    denominator=sum((x-mx)**2 for x in xs)
    if not denominator:return {"status":"unknown","reason":"Queue timestamps did not advance"}
    slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/denominator
    stable=slope<=cell.get("maximum_queue_slope_per_second",0) and ys[-1]<=cell.get("maximum_final_queue",0)
    return {"status":"stable" if stable else "growing_or_not_drained","metric":metric,
        "samples":len(points),"slope_per_second":slope,"final_queue":ys[-1],
        "scope":"Admission-period observations only; no drain samples; finite-trial evidence."}


def gate(summary, cell):
    counts=summary["counts"];n=summary["attempts"]
    bad=sum(counts.get(key,0) for key in ("failed","shed","unfinished"))
    accepted=(summary.get("goodput_requests_per_s") or 0)*summary["elapsed_s_including_drain"]
    passed=bool(n and summary["complete"] and not summary["missing_usage_attempts"] and not summary["invalid_protocol_attempts"]
        and not summary["malformed_journal_lines"] and bad/n<=cell.get("max_error_fraction",0)
        and accepted/n+1e-12>=cell["minimum_slo_fraction"])
    if cell["kind"]=="quality":
        quality=summary["quality_diagnostic"]
        passed &= bool(quality["n"] and quality["passed"]/quality["n"]>=cell["minimum_quality_fraction"])
    if summary.get("sessions"):
        passed &= summary["sessions"]["completed"]==summary["sessions"]["planned"]
    if cell.get("evidence_role")=="capacity":
        passed &= bool(summary.get("tail_latency_qualified") and summary.get("valid_capacity_evidence")
                       and summary.get("queue_stability",{}).get("status")=="stable")
        quality=summary["quality_diagnostic"]
        passed &= bool(quality["n"]==n-counts.get("shed",0) and quality["n"]
                       and quality["passed"]/quality["n"]>=cell["minimum_task_quality_fraction"])
    return bool(passed)


def prepare(cfg,out,resume=False,mock_token=None):
    started=time.time();config.validate(cfg,execution=True)
    if cfg["purpose"]=="cpu_mock":fixture.verify(cfg["endpoints"],mock_token)
    frozen=identity.freeze(cfg)
    qualification=policy.qualification(frozen)
    for cell in cfg["cells"]:identity.entries(frozen,cell)
    out=Path(out)
    if out.exists() and not resume:raise ValueError("New output directory or explicit --resume required")
    out.mkdir(parents=True,exist_ok=True)
    marker=out/"experiment.json"
    if marker.exists():
        previous=json.loads(marker.read_text())
        if previous["fingerprint"]!=frozen["fingerprint"]:
            raise ValueError("Frozen input bytes or code changed; preserve run and choose a new directory")
        if export.sha(out/"frozen-inputs.json")!=previous["frozen_sha256"] or export.sha(out/"run-window.json")!=previous["window_sha256"]:
            raise ValueError("Frozen snapshot/window changed; resume refused")
        window=budget.window(cfg,out,started)
    else:
        if any(out.iterdir()):raise ValueError("Unrecognized nonempty output directory; preserve it")
        window=budget.window(cfg,out,started)
        harness.write_json(out/"frozen-inputs.json",frozen)
        harness.write_json(marker,{"schema":config.SCHEMA,"fingerprint":frozen["fingerprint"],
            "frozen_sha256":export.sha(out/"frozen-inputs.json"),"window_sha256":export.sha(out/"run-window.json"),
            "qualification":qualification,"utc":harness.utc()})
    return frozen,window


def _retained(out,cell,fingerprint,deadline):
    base=out/cell["id"]
    candidates=([base] if base.exists() else [])+sorted(out.glob(base.name+"-retry-*"))
    found=None
    for candidate in candidates:
        receipt=candidate/"cell-receipt.json"
        if not receipt.exists():continue
        old=json.loads(receipt.read_text())
        if old["fingerprint"]!=fingerprint or old["cell"]!=cell or not old["passed"]:
            raise ValueError("Failed or mismatched retained cell requires diagnosis")
        for name,digest in old["artifact_hashes"].items():
            if export.sha(candidate/name,deadline)!=digest:raise ValueError("Retained artifact changed")
        found=(candidate,json.loads((candidate/"summary.json").read_text()))
    if found:return found
    if candidates:base=out/(base.name+"-retry-"+uuid.uuid4().hex[:8])
    return base,None


def execute(cfg,out,resume=False,mock_token=None,prepared=False):
    frozen,window=prepare(cfg,out,resume or prepared,mock_token)
    out=Path(out);deadline=window["collection_expires_at"]
    results={};roots={};expanded=config.expand(cfg)
    try:
        budget.check(deadline)
        if (out/"experiment-complete.json").exists():
            analysis.verify_final(out)
            return {key:json.loads((out/path).read_text()) for key,path in json.loads((out/"experiment-complete.json").read_text())["summaries"].items()}
        harness.wait_ready(cfg["endpoints"],cfg["identity"]["served_model"],min(10,budget.check(deadline)))
        for cell in expanded:
            root,summary=_retained(out,cell,frozen["fingerprint"],deadline)
            if summary is not None:
                results[cell["id"]]=summary;roots[cell["id"]]=root;continue
            cost=cell["duration_seconds"]+cell["timeout_seconds"]+3
            if cell.get("cache_protocol")=="warm-replay":cost*=2
            budget.check(deadline,cost)
            driver=config.driver_config(cfg,cell);driver["deadline_epoch"]=deadline
            cases=frozen["inputs"]["cases"]["data"] if cell["kind"]=="quality" else None
            if cases:driver["requests"]=max(driver["requests"],len(cases))
            entries=identity.entries(frozen,cell) if not cases else None
            if cell.get("cache_protocol")=="warm-replay":
                if not entries:raise ValueError("Warm replay requires exact frozen prompts")
                prime_root=out/(root.name+"-unscored-prime-"+uuid.uuid4().hex[:8])
                prime=harness.run_cell(driver,prime_root,corpus=entries)
                if not gate(prime,{**cell,"kind":"load","evidence_role":"pilot"}):raise ValueError("Unscored priming failed")
                budget.check(deadline)
            summary=sessions.run(driver,cell,root) if cell["kind"]=="sessions" else harness.run_cell(driver,root,cases=cases,corpus=entries)
            summary.update({"experiment_purpose":cfg["purpose"],"profiled":cell.get("profiled",False),
                            "source_cell_id":cell["source_cell_id"],"evidence_role":cell.get("evidence_role","pilot"),
                            "experiment_fingerprint":frozen["fingerprint"],"configuration_contract":config.contract(cell),
                            "experiment_qualified":False,"qualification_scope":"FINAL_RECEIPT_REQUIRED"})
            summary["queue_stability"]=queue_stability(root,cell)
            tail=cell.get("required_capacity_tail","p99")
            latency=summary["latency_success_s"].get(tail);ttft=summary["ttft_generated_success_s"].get(tail)
            summary["tail_latency_qualified"]=bool(latency is not None and ttft is not None and latency<=cell["slo"]["e2e_seconds"] and ttft<=cell["slo"]["ttft_seconds"])
            summary["cell_gate_passed"]=gate(summary,cell)
            if cases:
                rows,_=harness.read_rows(root/"attempts.jsonl")
                summary["canary_case_ids"]=sorted({row["task_id"] for row in rows if row.get("task_id")})
                summary["cell_gate_passed"] &= summary["canary_case_ids"]==sorted(case["id"] for case in cases)
                summary["canary_identity"]=harness.digest({"cases":cases,"contract":config.contract(cell)})
                baseline=frozen["inputs"].get("baseline:"+cell["source_cell_id"])
                if baseline:
                    if baseline["data"].get("cell_gate_passed") is not True:raise ValueError("Baseline canary is not qualified")
                    comparison=analysis.quality_comparison(baseline["data"],summary,cfg["maximum_quality_degradation"])
                    summary["baseline_comparison"]=comparison;summary["cell_gate_passed"] &= comparison["pass"]
            (root/"summary.json").rename(root/"driver-summary.json")
            harness.write_json(root/"summary.json",summary)
            hashes={path.name:export.sha(path,deadline) for path in root.iterdir() if path.is_file()}
            harness.write_json(root/"cell-receipt.json",{"fingerprint":frozen["fingerprint"],"cell":cell,"passed":summary["cell_gate_passed"],"artifact_hashes":hashes})
            results[cell["id"]]=summary;roots[cell["id"]]=root
            print(json.dumps({"cell":root.name,"counts":summary["counts"],"gate_passed":summary["cell_gate_passed"]}),flush=True)
            if not summary["cell_gate_passed"]:raise ValueError("Quality/SLO/accounting/rejection gate failed; preserve evidence")
        budget.check(deadline)
        pairs={cell["id"]:{"before":cell["id"]+"-canary-before","after":cell["id"]+"-canary-after"} for cell in expanded if cell["kind"]!="quality"}
        groups={}
        for cell in cfg["cells"]:
            if cell.get("evidence_role")!="capacity":continue
            members=[key for key,value in results.items() if value["source_cell_id"]==cell["id"] and value["evidence_role"]=="capacity"]
            if len(members)<3:raise ValueError("Capacity requires three completed matched repetitions")
            groups[cell["id"]]={"members":members,"minimum_goodput_requests_per_s":min(results[key]["goodput_requests_per_s"] for key in members)}
        preliminary=out/("collection-complete-"+uuid.uuid4().hex[:8]+".json")
        harness.write_json(preliminary,{"status":"ALL_CONFIGURATIONS_AND_FINAL_CANARIES_PASSED","fingerprint":frozen["fingerprint"],"pairs":pairs,"utc":harness.utc()})
        hard_end=window["expires_at"];budget.check(hard_end)
        archive_path=out.parent/(out.name+"-"+uuid.uuid4().hex[:8]+".tar.gz")
        harness.archive(out,archive_path,hard_end)
        exported=None
        if cfg.get("export_destination"):
            destination=Path(cfg["export_destination"])/archive_path.name
            exported=export.copy(archive_path,destination,hard_end)
        elif cfg["purpose"]=="b200_research":raise ValueError("Durable export missing; qualification refused")
        hashes={str(path.relative_to(out)):export.sha(path,hard_end) for path in out.rglob("*") if path.is_file()}
        budget.check(hard_end)
        harness.write_json(out/"experiment-complete.json",{"schema":"infrx-final/2","status":"CPU_MOCK_PASS" if cfg["purpose"]=="cpu_mock" else "LOCAL_PROXY_PASS" if cfg["purpose"]=="local_rehearsal" else "B200_FINITE_TRIAL_QUALIFIED",
            "purpose":cfg["purpose"],"fingerprint":frozen["fingerprint"],"pairs":pairs,"capacity_groups":groups,
            "summaries":{key:str((roots[key]/"summary.json").relative_to(out)) for key in results},
            "artifact_hashes":hashes,"archive_sha256":export.sha(archive_path,hard_end),"export":exported,
            "expires_at":hard_end,"completed_at":time.time(),"utc":harness.utc()})
        return results
    except BaseException as error:
        if time.time()<window["expires_at"]:
            harness.write_json(out/("paused-"+str(time.time_ns())+".json"),{"status":"PAUSED","error_type":type(error).__name__,"reason":str(error),"utc":harness.utc()})
        raise


def supervised_execute(cfg,out,resume=False,mock_token=None):
    _,window=prepare(cfg,out,resume,mock_token)
    budget.supervised("infrx_bench.worker",{"operation":"collect","config":cfg,"out":str(out),"mock_token":mock_token},window["expires_at"])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="action",required=True)
    for name in ("plan","run"):
        item=sub.add_parser(name);item.add_argument("--config",required=True)
        if name=="run":item.add_argument("--out",required=True);item.add_argument("--execute",action="store_true");item.add_argument("--resume",action="store_true")
    item=sub.add_parser("rehearse");item.add_argument("--out",required=True);item.add_argument("--config",default=str(Path(__file__).parent.parent/"configs/cpu-mock.json"))
    item=sub.add_parser("preflight");item.add_argument("--endpoint");item.add_argument("--model");item.add_argument("--out")
    item=sub.add_parser("cost");item.add_argument("--summary",required=True);item.add_argument("--run");item.add_argument("--node-hourly",type=float,required=True);item.add_argument("--requests-per-user-minute",type=float,required=True);item.add_argument("--users",type=int);item.add_argument("--allocated-gpus",type=int,default=8);item.add_argument("--other-hourly",type=float,default=0)
    item=sub.add_parser("reservation-cost");item.add_argument("--hours",type=float,required=True);item.add_argument("--node-hourly",type=float,required=True);item.add_argument("--accepted-tasks",type=int,required=True);item.add_argument("--active-user-hours",type=float,required=True);item.add_argument("--other-hourly",type=float,default=0)
    item=sub.add_parser("compare-quality");item.add_argument("baseline");item.add_argument("candidate");item.add_argument("--max-degradation",type=float,required=True)
    for name in ("archive","export","verify-export"):
        item=sub.add_parser(name);item.add_argument("source");item.add_argument("destination");item.add_argument("--deadline-utc",required=True)
    item=sub.add_parser("sweep");item.add_argument("--config",required=True);item.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    try:
        if args.action in ("plan","run"):
            cfg=config.load(args.config)
            if args.action=="plan" or not args.execute:
                print(json.dumps({"execution":False,"config":cfg,"expanded_cells":config.expand(cfg),"bundle_sha256":config.bundle_hash()},indent=2));return 0
            if cfg["purpose"]=="cpu_mock":raise ValueError("Controlled CPU fixture runs use rehearse, not supplied endpoints")
            supervised_execute(cfg,args.out,args.resume)
        elif args.action=="rehearse":
            cfg=config.load(args.config)
            if cfg["purpose"]!="cpu_mock":raise ValueError("rehearse only accepts the fixed CPU fixture arm")
            with fixture.server() as (server,base):supervised_execute({**cfg,"endpoints":[base]},args.out,mock_token=server.token)
        elif args.action=="preflight":
            result=preflight.inventory(args.endpoint,args.model)
            if args.out:harness.write_json(args.out,result)
            print(json.dumps(result,indent=2))
        elif args.action=="cost":
            summary=json.loads(Path(args.summary).read_text());qualification=analysis.qualify_summary(args.run,args.summary) if args.run else None
            print(json.dumps(analysis.capacity_cost(summary,args.node_hourly,args.requests_per_user_minute,args.users,args.allocated_gpus,args.other_hourly,qualification),indent=2))
        elif args.action=="reservation-cost":print(json.dumps(analysis.reservation_cost(args.hours,args.node_hourly,args.accepted_tasks,args.active_user_hours,args.other_hourly),indent=2))
        elif args.action=="compare-quality":print(json.dumps(analysis.quality_comparison(json.loads(Path(args.baseline).read_text()),json.loads(Path(args.candidate).read_text()),args.max_degradation),indent=2))
        elif args.action=="sweep":
            from . import sweeps
            print(json.dumps(sweeps.write(config.load(args.config),args.out),indent=2))
        else:
            import datetime as dt
            deadline=dt.datetime.fromisoformat(args.deadline_utc.replace("Z","+00:00"))
            if deadline.tzinfo is None:raise ValueError("Deadline requires timezone")
            budget.supervised("infrx_bench.worker",{"operation":args.action,"source":args.source,"destination":args.destination,"deadline":deadline.timestamp()},deadline.timestamp())
        return 0
    except (ValueError,OSError,KeyError,TimeoutError) as error:
        print("BLOCKED/PAUSED: "+str(error),file=sys.stderr);return 2


if __name__=="__main__":raise SystemExit(main())
