"""Review-only native B200 deployment commands. This module never executes a server."""
import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shlex

from . import config, harness, identity, policy

ROOT = Path(__file__).resolve().parents[1]
NO_THINKING = {"reasoning_effort":"none", "chat_template_kwargs":{"thinking":False}}


def contract(spec):
    return {"engine":spec["runtime"]["engine"], "topology":spec["topology"],
            "configured_context_max_tokens":spec["configured_context_max_tokens"],
            "engram_cpu_offload":spec["engram_cpu_offload"],
            "scheduler":spec["scheduler"], "speculative_decoding":"disabled",
            "language_model_only":False, "request_modes":{"none":NO_THINKING}}


def validate(spec):
    if spec.get("schema")!="infrx-deployment/1" or spec.get("status")!="PROPOSED_ON_NODE_QUALIFICATION_REQUIRED":
        raise ValueError("Explicit proposed deployment schema/status required")
    engine = spec["runtime"]["engine"]
    if engine not in ("sglang","vllm") or spec.get("id") != f"native-{engine}-b200-default":
        raise ValueError("Only the two reviewed native default candidates are supported")
    if (spec["identity"].get("repo"),spec["identity"].get("revision"))!=policy.NATIVE or spec["identity"].get("precision")!="native":
        raise ValueError("Exact native checkpoint required")
    if spec["identity"].get("served_model")!=policy.NATIVE[0]:
        raise ValueError("Fixed native served identity required")
    expected_topology={"node_gpu_count":8,"allocated_gpu_count":4 if engine=="sglang" else 2,
                       "tensor_parallel_size":4 if engine=="sglang" else 2,
                       "expert_parallel_size":4 if engine=="sglang" else None,
                       "expert_parallel_enabled":engine=="sglang"}
    if harness.digest(spec.get("topology"))!=harness.digest(expected_topology):
        raise ValueError("Unreviewed topology: only SGLang TP4/EP4 and vLLM TP2 are proposed")
    if spec.get("engram_cpu_offload") is not (engine=="vllm"):
        raise ValueError("Offload must match the reviewed default candidate")
    config.finite(spec["configured_context_max_tokens"],"configured_context_max_tokens",512,1048576)
    if not isinstance(spec["configured_context_max_tokens"],int):
        raise ValueError("Integer context limit required")
    expected_scheduler={"max_running_requests":256} if engine=="sglang" else {"max_num_seqs":128}
    if harness.digest(spec.get("scheduler"))!=harness.digest(expected_scheduler) or harness.digest(spec.get("request_modes"))!=harness.digest({"none":NO_THINKING}):
        raise ValueError("Unknown scheduler or thinking semantics; only the reviewed default is supported")
    if spec.get("speculative_decoding")!="disabled" or spec.get("language_model_only") is not False:
        raise ValueError("Advanced/speculative/text-only changes need a separate proposal")
    manifest_name=spec["runtime_review_file"]
    if manifest_name!=f"manifests/runtimes/{engine}-native-b200.json":
        raise ValueError("Expected immutable runtime review path")
    review=identity.read_json(ROOT/manifest_name)["data"]
    runtime=spec["runtime"]
    if runtime["image_digest"] != review["registry"]["repo"]+"@"+review["registry"]["platform_digest"]:
        raise ValueError("Image must match the reviewed Linux amd64 platform digest")
    if runtime["source_commit"] != review["registry"]["source_commit"]:
        raise ValueError("Runtime source must match image metadata, not the cookbook commit")
    if review["registry"]["architecture"]!="amd64" or review["registry"]["os"]!="linux" or review.get("on_node_qualification")!="NOT_RUN":
        raise ValueError("Runtime review cannot substitute for on-node qualification")
    if runtime["reviewed_utc"]!=review["reviewed_utc"] or runtime["support_evidence"] != [source["url"] for source in review["sources"]]:
        raise ValueError("Support provenance must match the retained review")
    for source in review["sources"]:
        if not re.fullmatch(r"[0-9a-f]{64}",source["sha256"]):
            raise ValueError("Source-byte checksum required")
        if source.get("commit") and source["commit"] not in source["url"]:
            raise ValueError("Support source must bind its exact commit")
    return spec


def validate_benchmark(spec, benchmark):
    validate(spec);config.validate(benchmark)
    if benchmark["purpose"]!="b200_research" or benchmark["arm"]!=f"native-{spec['runtime']['engine']}-baseline":
        raise ValueError("Benchmark arm must match the native deployment")
    if benchmark["identity"]!=spec["identity"] or benchmark["runtime"]!=spec["runtime"]:
        raise ValueError("Benchmark checkpoint/runtime differs from deployment")
    if benchmark.get("deployment_contract")!=contract(spec):
        raise ValueError("Benchmark topology/offload/scheduler contract differs")
    for cell in benchmark["cells"]:
        if (cell.get("configured_context_max_tokens")!=spec["configured_context_max_tokens"]
            or cell.get("thinking_mode")!="none" or cell.get("body")!=NO_THINKING):
            raise ValueError("Benchmark context/thinking must match the deployment; fixed-length stress is separate")
        if cell.get("actual_input_token_target",0)+cell["output_tokens"]>spec["configured_context_max_tokens"]:
            raise ValueError("Prompt plus full output budget exceeds configured context")
    return benchmark


def command(spec, weights, gpus, port, name):
    validate(spec)
    path=PurePosixPath(weights)
    if not path.is_absolute() or str(path)!=weights or ".." in path.parts or any(c in weights for c in (":","\n","\r","\x00")):
        raise ValueError("Canonical absolute Linux weight path required; no volume delimiters")
    if not re.fullmatch(r"[0-7](,[0-7])*",gpus) or len(set(gpus.split(",")))!=spec["topology"]["allocated_gpu_count"] or len(gpus.split(","))!=spec["topology"]["allocated_gpu_count"]:
        raise ValueError("Select the exact distinct GPU count within the eight-GPU node")
    if not isinstance(port,int) or isinstance(port,bool) or not 1024<=port<=65535 or port in (8000,8100,8110):
        raise ValueError("Select a nonreserved task-local port")
    if not re.fullmatch(r"infrx-[a-z0-9-]{1,48}",name):
        raise ValueError("Task-owned container name required")
    engine=spec["runtime"]["engine"];maximum=str(spec["configured_context_max_tokens"])
    docker=["docker","run","--rm","--pull=never","--name",name,"--label","infrx-bench.proposal="+spec["id"],
            "--gpus",'"device='+gpus+'"',"--ipc=host","--shm-size","32g",
            "-p",f"127.0.0.1:{port}:30000","-v",weights+":/model:ro",
            "-e","HF_HUB_OFFLINE=1","-e","TRANSFORMERS_OFFLINE=1"]
    if engine=="sglang":
        docker += ["-e","SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0","-e","SGLANG_DEFAULT_THINKING=false"]
        server=["sglang","serve","--model-path","/model","--trust-remote-code","--tp","4","--ep-size","4",
                "--max-running-requests","256","--context-length",maximum,"--reasoning-parser","auto","--tool-call-parser","auto"]
    else:
        docker += ["-e","VLLM_ENGINE_READY_TIMEOUT_S=3600","-e","VLLM_USE_RUST_FRONTEND=1"]
        attention={"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":True}
        server=["vllm","serve","/model","--tokenizer-mode","deepseek_v41","--tensor-parallel-size","2",
                "--attention-config",json.dumps(attention,separators=(",",":")),"--kv-cache-dtype","fp8",
                "--max-num-seqs","128","--engram-config",'{"cpu_offload":true}',"--max-model-len",maximum,
                "--tool-call-parser","deepseek_v41","--enable-auto-tool-choice","--reasoning-parser","deepseek_v41"]
    server += ["--served-model-name",spec["identity"]["served_model"],"--host","0.0.0.0","--port","30000"]
    return docker+["--entrypoint",server[0],spec["runtime"]["image_digest"],*server[1:]]


def preview(spec,benchmark,weights,gpus,port,name,benchmark_path):
    validate_benchmark(spec,benchmark)
    endpoint=f"http://127.0.0.1:{port}"
    if benchmark["endpoints"]!=[endpoint]:
        raise ValueError("Rendered port must match the benchmark endpoint")
    argv=command(spec,weights,gpus,port,name)
    return {"schema":"infrx-deployment-preview/1","executed":False,"readiness":spec["status"],
            "deployment_sha256":harness.digest(spec),"deployment_contract":contract(spec),
            "docker_argv":argv,"review_shell_command":shlex.join(argv),
            "benchmark_plan_argv":["python3","bench.py","plan","--config",str(benchmark_path)],
            "benchmark_run_after_approval_argv":["python3","bench.py","run","--config",str(benchmark_path),"--out","results/"+spec["id"],"--execute"],
            "smoke_request":{"model":spec["identity"]["served_model"],"messages":[{"role":"user","content":"What is 17*19? Return only the integer."}],"max_tokens":256,"temperature":0,**NO_THINKING},
            "smoke_expected_answer":"323", "qualification":"NOT_RUN",
            "required_checks":spec["required_checks"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment",required=True);parser.add_argument("--benchmark",required=True)
    parser.add_argument("--weights",required=True);parser.add_argument("--gpus",required=True)
    parser.add_argument("--port",type=int,required=True);parser.add_argument("--name",required=True)
    parser.add_argument("--out")
    args=parser.parse_args()
    try:
        result=preview(identity.read_json(args.deployment)["data"],identity.read_json(args.benchmark)["data"],
                       args.weights,args.gpus,args.port,args.name,args.benchmark)
        if args.out:
            with Path(args.out).open("x") as stream:json.dump(result,stream,indent=2)
        else:print(json.dumps(result,indent=2))
        return 0
    except (ValueError,OSError,KeyError) as error:
        print("BLOCKED: "+str(error));return 2


if __name__=="__main__":raise SystemExit(main())
