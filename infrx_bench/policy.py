"""Supported candidate identities and required evidence; no manufactured qualification."""
from . import harness

NATIVE = ("deepseek-ai/DeepSeek-V4.1-Flash", "2cba9e42aa026125f3ed06c6d98c1db82f7ca027")
NVFP4 = ("nvidia/DeepSeek-V4.1-Flash-NVFP4", "3431dde3247c13b5957f682b1e3c6fcae2566079")
PROXY = ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc")
CANDIDATES = {"native-sglang-baseline":(NATIVE,"sglang","baseline"),
              "native-sglang-session-cache":(NATIVE,"sglang","optimized"),
              "native-vllm-baseline":(NATIVE,"vllm","baseline"),
              "native-vllm-text-only":(NATIVE,"vllm","optimized"),
              "nvfp4-candidate":(NVFP4,"sglang","optimized"),
              "qwen-proxy":(PROXY,None,"proxy")}


def candidate(cfg):
    if cfg.get("arm") == "cpu-fixture":
        if cfg.get("purpose") != "cpu_mock" or cfg.get("arm_role") != "mock" or cfg["runtime"].get("engine") != "fixture":
            raise ValueError("CPU fixture cannot relabel a real arm")
        if cfg["identity"] != {"repo":"fixture/model","revision":"fixture-only","served_model":"fixture/model","precision":"CPU fixture"}:
            raise ValueError("CPU fixture identity is fixed")
        return
    if cfg.get("purpose") == "cpu_mock" or cfg.get("arm") not in CANDIDATES:
        raise ValueError("Unsupported/blocked arm; purpose labels do not grant support")
    model, engine, role = CANDIDATES[cfg["arm"]]
    if (cfg["identity"]["repo"], cfg["identity"]["revision"]) != model or cfg.get("arm_role") != role:
        raise ValueError("Arm role and exact checkpoint must match supported candidate")
    expected_precision="native" if model==NATIVE else "nvfp4" if model==NVFP4 else "bf16-proxy"
    if cfg["identity"].get("precision")!=expected_precision:
        raise ValueError("Precision label cannot relabel a native/proxy checkpoint")
    if engine and cfg["runtime"].get("engine") != engine: raise ValueError("Unsupported engine/checkpoint combination")
    if model != PROXY and cfg["purpose"] != "b200_research": raise ValueError("DeepSeek candidates require B200 qualification")
    if model == PROXY and cfg["purpose"] != "local_rehearsal": raise ValueError("Proxy is functional evidence only")


def qualification(frozen):
    cfg = frozen["config"]
    candidate(cfg)
    if cfg["purpose"] == "cpu_mock": return None
    inputs = frozen["inputs"]
    receipt = inputs.get("qualification",{}).get("data",{})
    if receipt.get("schema") != "infrx-qualification/2" or receipt.get("status") != "QUALIFIED_WITH_EVIDENCE":
        raise ValueError("Unknown runtime/hardware facts: qualified evidence file required")
    if receipt.get("identity") != cfg["identity"] or receipt.get("runtime") != cfg["runtime"]:
        raise ValueError("Qualified checkpoint/runtime differs from collection")
    required = ("inventory", "weights", "runtime", "smoke")
    if any("qualification:"+name not in inputs for name in required):
        raise ValueError("Missing inventory, weights, runtime inspection or generation-smoke evidence")
    artifacts = {name:inputs["qualification:"+name]["data"] for name in required}
    weights = artifacts["weights"]
    if weights.get("verified") is not True or (weights.get("repo"),weights.get("revision")) != (cfg["identity"]["repo"],cfg["identity"]["revision"]):
        raise ValueError("Exact verified weight evidence required")
    if not weights.get("files") or any(not item.get("sha256") for item in weights["files"]):
        raise ValueError("Initial weight checksum inventory required")
    runtime = artifacts["runtime"]
    if runtime.get("runtime") != cfg["runtime"] or runtime.get("inspection_status") != "VERIFIED":
        raise ValueError("Actual runtime inspection missing or mismatched")
    deployment_contract=cfg.get("deployment_contract")
    if deployment_contract is not None and runtime.get("deployment_contract")!=deployment_contract:
        raise ValueError("Actual topology/offload/scheduler differs from proposed deployment contract")
    modes = runtime.get("request_modes",{})
    for cell in cfg["cells"]:
        thinking = {key:value for key,value in cell.get("body",{}).items() if key != "ignore_eos"}
        if modes.get(cell["thinking_mode"]) != thinking:
            raise ValueError("Thinking semantics not qualified for this exact runtime")
        context = cell.get("configured_context_max_tokens")
        if context is None or context != runtime.get("configured_context_max_tokens"):
            raise ValueError("Configured context maximum must match actual deployment inspection")
        if cell.get("cache_protocol") == "tenant-isolation" and runtime.get("cache_salt_verified") is not True:
            raise ValueError("Tenant salt capability is unverified")
    smoke = artifacts["smoke"]
    if smoke.get("identity") != cfg["identity"] or smoke.get("runtime") != cfg["runtime"] or smoke.get("generation_status") != "PASSED":
        raise ValueError("Exact actual generation smoke required")
    if smoke.get("failed_attempts") != 0 or not smoke.get("successful_output_tokens",0):
        raise ValueError("Generation smoke failed or empty")
    inventory = artifacts["inventory"]
    if inventory.get("status") != "VERIFIED" or not inventory.get("hardware"):
        raise ValueError("Actual device facts remain unknown")
    hardware = inventory["hardware"]
    if cfg["purpose"] == "b200_research":
        if hardware.get("gpu_model") != "NVIDIA B200" or hardware.get("node_gpu_count") != 8 or hardware.get("allocated_gpu_count") not in (2,4,8):
            raise ValueError("Qualified reserved B200 deployment required")
        if deployment_contract is not None and hardware.get("allocated_gpu_count")!=deployment_contract["topology"]["allocated_gpu_count"]:
            raise ValueError("Observed GPU allocation differs from deployment contract")
        if cfg["arm"] == "native-vllm-text-only" and runtime.get("language_model_only_verified") is not True:
            raise ValueError("Text-only runtime flag remains unqualified")
    return {"fingerprint":harness.digest(receipt), "hardware":hardware,
            "scope":"Operator-supplied evidence artifacts verified by byte identity and field consistency; not a remote attestation protocol."}
