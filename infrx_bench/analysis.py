"""Offline capacity and cost scenarios. User prices/rates are mandatory inputs."""
import math
import json
from pathlib import Path
from . import export


def reservation_cost(hours, node_hourly, accepted_tasks, active_user_hours, other_hourly=0):
    for value in (hours, node_hourly, active_user_hours):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Positive reservation hours, price and active user hours required")
    if not isinstance(accepted_tasks, int) or isinstance(accepted_tasks, bool) or accepted_tasks < 0:
        raise ValueError("Nonnegative integer accepted tasks required")
    if not math.isfinite(other_hourly) or other_hourly < 0:
        raise ValueError("Nonnegative overhead required")
    total = hours * (node_hourly + other_hourly)
    return {"schema":"infrx-reservation-cost/1", "reservation_hours":hours,
        "whole_node_cost_usd":total, "accepted_tasks":accepted_tasks,
        "usd_per_accepted_task":total/accepted_tasks if accepted_tasks else None,
        "active_user_hours":active_user_hours, "usd_per_active_user_hour":total/active_user_hours,
        "scope":"Input denominators must include only quality/SLO accepted tasks and actual active user hours. Entire paid reservation charged, including idle, setup and failed work."}


def capacity_cost(summary, node_hourly, requests_per_user_minute, users=None, allocated_gpus=8, other_hourly=0, qualification=None):
    for v in (node_hourly, requests_per_user_minute):
        if not math.isfinite(v) or v <= 0: raise ValueError("Positive finite node price and user request rate required")
    if users is not None and (not isinstance(users, int) or isinstance(users, bool) or users <= 0):
        raise ValueError("Requested users must be a positive integer")
    if not math.isfinite(other_hourly) or other_hourly < 0 or not 1 <= allocated_gpus <= 8:
        raise ValueError("Invalid overhead or hypothetical GPU allocation")
    elapsed = summary["elapsed_s_including_drain"]
    goodput = summary.get("goodput_requests_per_s") or 0
    success = summary["counts"]["success"]
    accepted = goodput * elapsed
    conservative=qualification["minimum_goodput_requests_per_s"] if qualification else goodput
    capacity = math.floor(conservative * 60 / requests_per_user_minute)
    full_cost = (node_hourly+other_hourly) * elapsed/3600
    output = summary.get("successful_output_tokens", 0)
    qualified = bool(qualification and qualification.get("verified_final") and summary.get("experiment_purpose")=="b200_research")
    return {"schema":"infrx-cost/1", "node_hourly_usd":node_hourly, "other_hourly_usd":other_hourly,
        "whole_reservation_gpus":8, "hypothetical_allocated_gpus":allocated_gpus,
        "hypothetical_fractional_hourly_usd":node_hourly*allocated_gpus/8+other_hourly,
        "actual_full_node_cost_for_observed_cell_usd":full_cost,
        "successful_output_usd_per_million":full_cost/output*1e6 if output else None,
        "usd_per_slo_accepted_request":full_cost/accepted if accepted else None,
        "requests_per_user_minute":requests_per_user_minute,
        "capacity_users_scenario":capacity, "sustainable_users_qualified":capacity if qualified else None,
        "requested_users":users, "requested_load_requests_per_s":users*requests_per_user_minute/60 if users is not None else None,
        "qualification":"MATCHED_REPEATED_FINITE_TRIAL_QUALIFIED" if qualified else "SCENARIO_ONLY_NOT_SUSTAINABILITY_PROOF",
        "capacity_goodput_requests_per_s":conservative,"capacity_scope":"Finite repeated trials, not indefinite stability or a confidence interval",
        "successful_requests":success, "slo_accepted_requests":accepted,
        "completed_sessions":summary.get("sessions",{}).get("completed"),
        "usd_per_completed_session_transport":full_cost/summary["sessions"]["completed"] if summary.get("sessions",{}).get("completed") else None,
        "constraints":"User rates/distributions must match this cell. Offload, idle, loading, failed work, export and reservation gaps must enter a whole-window ledger.",
        "fractional_allocation_is_hypothetical":True}


def quality_comparison(baseline, candidate, max_degradation=0):
    if not math.isfinite(max_degradation) or not 0 <= max_degradation <= 1:
        raise ValueError("Quality degradation bound must be in [0, 1]")
    if baseline.get("canary_identity") != candidate.get("canary_identity"):
        raise ValueError("Canary prompts, thinking, sampling and output budgets must match")
    if not baseline.get("canary_identity"):raise ValueError("Canary identity required")
    a, b = baseline["quality_diagnostic"], candidate["quality_diagnostic"]
    if not a["n"] or not b["n"]: raise ValueError("Nonempty canary results required")
    if a["n"] != b["n"] or baseline.get("canary_case_ids") != candidate.get("canary_case_ids"):
        raise ValueError("Canary coverage must match")
    af, bf = a["passed"]/a["n"], b["passed"]/b["n"]
    return {"baseline_fraction":af, "candidate_fraction":bf, "degradation":af-bf,
        "maximum_degradation":max_degradation, "pass":af-bf <= max_degradation,
        "scope":"Small deterministic regression canary; not a general intelligence score."}


def verify_final(root):
    root=Path(root).resolve();receipt=json.loads((root/"experiment-complete.json").read_text())
    if receipt.get("schema")!="infrx-final/2" or receipt["completed_at"]>=receipt["expires_at"]:
        raise ValueError("No valid completion before the durable deadline")
    marker=json.loads((root/"experiment.json").read_text())
    if receipt["fingerprint"]!=marker["fingerprint"]:raise ValueError("Final identity mismatch")
    for name,digest in receipt["artifact_hashes"].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or export.sha(path)!=digest:raise ValueError("Final evidence changed")
    if not receipt.get("pairs"):raise ValueError("Final per-configuration canary pairs required")
    for cell,pair in receipt["pairs"].items():
        scored=json.loads((root/receipt["summaries"][cell]).read_text())
        for name in (cell,pair["before"],pair["after"]):
            summary=json.loads((root/receipt["summaries"][name]).read_text())
            if summary.get("cell_gate_passed") is not True or summary.get("experiment_fingerprint")!=receipt["fingerprint"]:
                raise ValueError("Final quality/SLO cell is unqualified")
        before=json.loads((root/receipt["summaries"][pair["before"]]).read_text())
        after=json.loads((root/receipt["summaries"][pair["after"]]).read_text())
        if before["canary_identity"]!=after["canary_identity"]:raise ValueError("Canary contract changed")
        if before["configuration_contract"]!=scored["configuration_contract"] or after["configuration_contract"]!=scored["configuration_contract"]:
            raise ValueError("Canary thinking/sampling/output/context differs from scored configuration")
    return receipt


def qualify_summary(root,summary_path):
    receipt=verify_final(root)
    if receipt.get("purpose")!="b200_research" or receipt.get("status")!="B200_FINITE_TRIAL_QUALIFIED" or not receipt.get("export",{}).get("destination_verified"):
        raise ValueError("Real B200 final canaries and verified export are required for capacity")
    if export.sha(receipt["export"]["destination"])!=receipt["archive_sha256"]:
        raise ValueError("Exported evidence is missing or changed")
    root=Path(root).resolve();path=Path(summary_path).resolve()
    if not path.is_relative_to(root):raise ValueError("Summary is outside this experiment")
    relative=str(path.relative_to(root))
    names=[name for name,p in receipt["summaries"].items() if p==relative]
    if len(names)!=1:raise ValueError("Summary is not a qualified scored cell")
    group=next((group for group in receipt["capacity_groups"].values() if names[0] in group["members"]),None)
    if not group or len(group["members"])<3:raise ValueError("Three matched capacity repetitions required")
    observations=[]
    for name in group["members"]:
        summary=json.loads((root/receipt["summaries"][name]).read_text())
        if (summary.get("evidence_role")!="capacity" or not summary.get("tail_latency_qualified")
            or not summary.get("valid_capacity_evidence") or summary.get("profiled")
            or summary.get("queue_stability",{}).get("status")!="stable"):
            raise ValueError("Capacity latency/queue/protocol/profile gate failed")
        observations.append(summary["goodput_requests_per_s"])
    return {"verified_final":True,"minimum_goodput_requests_per_s":min(observations),"repetitions":len(observations)}
