# Reviewed source and research hypotheses

Source audit: B200 readiness branch `a6446df7c087fa010ff37137f2b2375f5710fff0`; newer DeepSeek branch `ea64dc831aefd1da5dd8d101e94618b0ec92f4a0`. Only the bounded driver, staging helpers, tokenizer corpus, ten canary cases and CPU tests were selected. The newer branch informs requirements; destructive node preparation, mutable defaults, fallback orchestration and silent export failures were not imported. Original research remains in its own checkout.

Historical CPU tests and separate Qwen3.5-2B L40S artifacts establish upstream client/small-model work. They do not establish this reconstruction's tests or native DeepSeek B200 performance. See `PROVENANCE.json` for exact source identities and `VALIDATION.md` for fresh results.

## Model/runtime decisions requiring fresh qualification

The native checkpoint under discussion is `deepseek-ai/DeepSeek-V4.1-Flash` revision `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`, with mixed MXFP4 experts, other MXFP8 tensors and BF16 embedding/head tensors. Parent research reports approximately 511 GB / 476 GiB of weights, with a 552B backbone, 196B Engram and about 14B draft. Resident full-weight fit on a 128 GB Thor or 48 GB L40S is not established; small-model rehearsal on those hosts is a separate functional check.

[vLLM's V4.1 recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash) discusses Blackwell configurations including TP2 with Engram CPU offload and TP4 interactive serving. Do not repeat a categorical “TP2 cannot fit” claim from older plans. Measure GPU/CPU memory, offload cost and latency on the exact configuration. Use the documented image/runtime path and current support check.

[SGLang's V4.1 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx) uses its dedicated preview path. Parent research found a verified B200 TP4/EP4 reference, low-latency and throughput settings, and restrictions on unrelated optimization toggles. The sanitized historical image digest is retained in the proposed config; an old immutable digest is not proof of current supported flags or successful execution. Record autodetected kernel backends from actual logs. Do not infer support for DP attention, DeepEP, MegaMoE or deterministic mode.

The separate [NVIDIA NVFP4 export](https://huggingface.co/nvidia) identified by parent research uses revision `3431dde3247c13b5957f682b1e3c6fcae2566079`; approximately 492 GiB is larger than the native weights. Its GB300 evidence does not establish B200 compatibility. An exact model repository and runtime recipe must be supplied before execution. Uniform FP8/NVFP8 and TensorRT-LLM V4.1-Flash engine arms remain blocked/unverified; V4 documentation or a TensorRT kernel backend is insufficient.

## Vision-elision hypothesis

For text-only use, vLLM documents `--language-model-only` to skip vision encoder/aligner loading. If the baseline already uses it, there is no additional optimization from enabling the same flag. Confirm exact-model support and compare actual memory/latency under held-fixed text tasks. Do not apply an unqualified SGLang config override or rewrite checkpoint weights.

Parent inspection of the pinned native index reported dedicated vision tensors: encoder 823,685,120 stored bytes and projector 146,821,120, total 970,506,240 bytes (about 0.904 GiB and 0.190% of 510,286,023,000 stored tensor bytes), plus small delimiter tensors. Stored bytes are not per-GPU VRAM savings: sharding, replication, buffers and allocation differ. Avoid promises of a large memory reduction and never remove shared routing weights.

## Cache/session hypothesis

The source research, including `research/cross-cutting/serving-optimizations.md` and `research/cross-cutting/scaling/02-serving-stack-and-routing.md`, motivates prefix-aware routing and hybrid attention/state cache management. These are hypotheses to test with real engine counters. Independent random requests or a persistent JIT cache do not test multi-turn KV/state reuse. A warm exact-prompt replay is also different from growing sessions.

The new scheduler makes histories, tenant salts, endpoint affinity, queue order and admission windows explicit. It records outcomes without claiming engine cache hits or kernel batches. Extend task/rubric adapters and realistic traces after parameters are reconciled; built-in JSON canaries and synthetic prompts do not substitute for representative reasoning/coding workloads.
