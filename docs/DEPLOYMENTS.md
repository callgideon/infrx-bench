# Native B200 deployment proposals

Two default candidates now have concrete deployment specifications and matching benchmark files. `infrx_bench.deployment` validates their agreement and prints a Docker argument vector plus a shell-quoted review command. It has no execution switch and never runs Docker, sends generation requests, pulls an image or downloads weights.

| Candidate | Deployment specification | Matching benchmark |
|---|---|---|
| SGLang native TP4/EP4 | configs/deployments/native-sglang-b200.proposed.json | configs/b200-sglang-native.proposed.json |
| vLLM native TP2, Engram CPU offload | configs/deployments/native-vllm-b200.proposed.json | configs/b200-vllm-native.proposed.json |

The [pinned SGLang builder](https://github.com/sgl-project/sglang/blob/6fc8d9da3288e9710a6f9a1f59503cacf8a984f0/docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx) marks its B200 throughput recipe TP4/EP4 with 256 active requests. This proposal keeps Engram on GPUs, uses automatic backends/parsers and disables speculation. Its preview image is pinned to the Linux amd64 manifest of dev-dsv41.

The [pinned vLLM recipe](https://github.com/vllm-project/recipes/blob/8ed3d031cee2f8b7cfc5b701438aaacef6c69a18/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml) and [B200-specific command](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash/hw/b200.json) propose TP2 with Engram CPU offload, native sparse-attention settings, FP8 KV and 128 active sequences. The template explicitly selects the documented Rust frontend and pins the observed nightly amd64 image. Host DRAM, pinned/UVA placement, loader memory and transfer cost require measurement; no expert or KV offload is implied.

These are different operational bundles: allocation, expert parallelism, Engram placement and scheduler caps differ. Do not attribute the resulting difference solely to the engine or charge hypothetical fractional GPU cost for an eight-GPU reservation.

## Preview locally

Paths below are illustrative existing Linux weight locations; preview does not inspect or create them. Native weight receipt/hash verification happens separately. Create the results parent before using --out.

```sh
python3 -m infrx_bench.deployment --deployment configs/deployments/native-sglang-b200.proposed.json --benchmark configs/b200-sglang-native.proposed.json --weights /srv/infrx/native --gpus 0,1,2,3 --port 18300 --name infrx-sglang-native --out results/sglang-launch-review.json
python3 -m infrx_bench.deployment --deployment configs/deployments/native-vllm-b200.proposed.json --benchmark configs/b200-vllm-native.proposed.json --weights /srv/infrx/native --gpus 0,1 --port 18400 --name infrx-vllm-native --out results/vllm-launch-review.json
python3 bench.py plan --config configs/b200-sglang-native.proposed.json
python3 bench.py plan --config configs/b200-vllm-native.proposed.json
```

Previews use exact GPU selection, pull=never, read-only local weights, offline Hugging Face/Transformers settings, a task-owned name and a loopback published port. No privileged container is requested. IPC is shared with the host as in the upstream launch pattern; review this on the authorized node. SGLang's upstream trust-remote-code flag requires inspecting immutable local model/tokenizer code before launch. Backend flags and parser spelling are engine-specific; no automatic alternate backend/runtime is supplied.

## Context and thinking contract

Both proposals explicitly cap configured context at 16,384 tokens rather than inheriting the checkpoint maximum. The benchmark proposes 2,048 input tokens and a 256-token output cap, natural completion and temperature zero. These workload/SLO values remain proposals. Chat-template overhead and actual server prompt usage must fit the context with the full answer reserve.

Both request bodies send reasoning_effort none and chat_template_kwargs thinking false. This avoids different engine defaults and ambiguous tier mappings. The parsed answer/reasoning and no-thinking semantics still need actual verification. Higher thinking modes are not included in these default candidates. Speculation, vision-elision, custom cache/offload features and advanced parallelism remain separate proposals. A text smoke/pilot does not qualify vision.

## Required on-node qualification

Runtime review manifests in manifests/runtimes contain source commits/file SHA-256, recipe snapshot hashes, image index/platform/config digests and registry-observed runtime source commits. Documentation commits and image commits are distinct. Reading registry metadata proves no actual flag availability, load success, memory fit or quality. Inspect the exact installed image and serve help, GPU interconnect/allocation, CPU/NUMA memory, startup backend/configuration, weights and tokenizer before a separately approved bounded generation/canary run.

Complete the existing qualification artifacts from RUNBOOK.md. In addition, the runtime artifact must record deployment_contract exactly as in the benchmark: actual topology, offload, scheduler, configured context and request semantics. Inventory allocation must match four GPUs for SGLang or two for vLLM. The collector rejects mismatches even when operator status labels say VERIFIED. These are consistency checks, not remote attestation.

Only after parameter reconciliation, current support review, exact qualification, reservation end/export completion and explicit GPU/run approval:

```sh
python3 bench.py run --config /path/to/reviewed-sglang-config.json --out results/sglang-approved-pilot --execute
python3 bench.py run --config /path/to/reviewed-vllm-config.json --out results/vllm-approved-pilot --execute
```

The checked-in files intentionally fail real execution while parameters/corpus/qualification/end/export are unresolved. Published preview commands were constructed and checked on CPU; neither engine image was pulled or launched.
