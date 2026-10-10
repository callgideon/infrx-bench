# Experiment catalog

Proposed research, not execution permission. Collector-ready describes the client path; every real deployment still requires exact evidence. Freeze checkpoint/runtime/topology, corpus/seeds, sampling, thinking/output budgets, cache state, telemetry, quality rubric and SLO within comparisons. Report confounders.

| Track | Question and proposed axes | Measurement, readiness and priority |
|---|---|---|
| E01 Reference | Does an exact native deployment fit, generate and pass tasks? 2k input/256 output anchor, natural completion | Identity, fit, actual output/usage and canaries; client ready, runtime pending; first, 4h |
| E02 GPU/offload | Smallest supported topology meeting quality/SLO? 2/4/8 GPUs, documented CPU offload | GPU/CPU memory, offload bandwidth, latency, full-node cost; manual deployment qualification, 4h |
| E03 Engine | Common-control effect versus best operational bundle? Exact SGLang/vLLM builds; matched topology where supported | Joint quality/SLO goodput and memory; runtime pending, 5h. Different TP/EP/offload is a bundle comparison |
| E04 Format | Can a separately exported checkpoint improve accepted-task cost? Native versus qualified NVFP4 | Load size/time, memory, quality and latency; conditional B200 support, shares 4h with E09; unverified NVFP8/TRT-LLM blocked |
| E05 Scheduler | Which admission settings help? Active caps 1/4/16/64; select three qualified token-budget values from 2k/8k/32k | Driver wait, offered/admitted traffic, actual engine counters; driver ready, engine flags manual; 7h with E06 |
| E06 Load | Accepted-load knee? Rates .25/.5/.75/1/1.25 of a measured pilot anchor; refine knee by 10% | Admission-window queues plus rejection/error/quality/tail gates; client ready, corpus/metric qualification pending; 7h screen, 12h confirmation |
| E07 Context | Actual prompt versus configured maximum? Sparse actual 2k/8k/32k/64k/128k; max 16k/64k/128k/256k with answer reserve | Server tokens, memory, TTFT, quality; sparse file generation ready, tokenizer/runtime limits pending; 6h with E08 |
| E08 Cache/session | Does verified reuse help? Reuse 0/50/90%, prefix sharing 0/25/75%, 4–8 turns, tenants 1/8/32 at fixed total load | Cache counters, queue, history correctness/isolation; warm/session client ready, trace mixture/capability pending; 6h |
| E09 Thinking | Documented modes' quality/latency tradeoff? Caps 512/2k/8k, actual reasoning/answer tokens | Matched task success, generated/visible TTFT, total cost; runtime semantics pending; 4h with E04 |
| E10 Quality | Does each configuration preserve correctness? Matched canaries and rubrics on actual served tasks | Coverage, regressions, joint quality/latency; exact-text/JSON scoring ready, broad coding/tool evaluator pending; continuous |
| E11 Cost/users | Accepted task and active user cost? Actual full-node price; cadence 1 task/30/120/300s, burst 1/2/4 scenarios | Full reservation ledger, task denominator/user hours; calculator ready, price/workload evidence pending; 4h with E12 |
| E12 Whole node/recovery | Aggregate replicas/recovery? Qualified partition topologies and explicit recovery events | Aggregate acceptance, recovery loss/time, contention; orchestration/failure injection pending; conditional 4h |

Anchors are 2k input/256 output, 8k/512 and 32k/512. Screens are 2–5 minutes and establish screening evidence only. Confirmation proposes three one-hour repeats and a three-hour winner soak in bounded segments. The driver's 60-success p95/300-success p99 minima are exploratory, not confidence claims. Flat queues must be considered jointly with dropped traffic, errors, quality and latency. Do not double-discount burst/headroom after using a conservative measured capacity estimate.

```sh
python3 bench.py plan --config configs/b200-screen.proposed.json
python3 bench.py sweep --config configs/context-thinking-sweep.proposed.json --out results/context-plans
python3 bench.py rehearse --out results/cpu-check
python3 bench.py preflight --out results/inventory.json
python3 bench.py run --config /path/to/reviewed-qualified-config.json --out results/approved-pilot --execute
```

The first four commands inspect/prepare or use the CPU fixture. The final command needs separate GPU/run approval and completed qualification, and launches no server. Generated sweep files require matching deployments/corpora. GPU/offload search, engine flags, workload-mixture generation, replicas and recovery are manual/pending. Fixed-length ignore_eos is stress only; use natural completion for task cost. Fit, generation and SLO qualification remain separate. See RUNBOOK.md and PLAN48H.md.

Default native paths for E01/E03 now have configs/b200-sglang-native.proposed.json and configs/b200-vllm-native.proposed.json plus matching review-only deployment previews; see DEPLOYMENTS.md. These complete concrete command/config preparation, not on-node engine acceptance. Their different TP/EP/offload/scheduler settings are explicit operational-bundle confounders.
