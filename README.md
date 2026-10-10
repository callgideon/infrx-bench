# infrx-bench

Prepare and collect bounded inference evidence for **DeepSeek-V4.1-Flash on a reserved eight-B200 node**. Compare deployments by latency, task correctness, sustained offered load and whole-node cost. This checkpoint provides a CPU-tested collector and proposed research tracks; GPU fit, runtime support and performance remain unqualified.

Start locally with Python 3.10+ and the standard library. No package installation, model download, server launch or cloud action is automatic.

```sh
python3 -m unittest discover -s tests -v
python3 bench.py rehearse --out results/cpu-check
python3 bench.py plan --config configs/b200-sglang-native.proposed.json
python3 bench.py plan --config configs/b200-vllm-native.proposed.json
python3 bench.py sweep --config configs/context-thinking-sweep.proposed.json --out results/context-plans
```

The rehearsal starts a controlled loopback SSE fixture and collects six pilot configurations, each bracketed by matched canaries: 18 scored cells plus an unscored warm prime. Fixture answers, timings and counters establish client functionality only.

| Start here | Contents |
|---|---|
| [AGENTS.md](AGENTS.md) | Shared agent instructions; [CLAUDE.md](CLAUDE.md) imports them |
| [Runbook](docs/RUNBOOK.md) | Ordered preparation, qualification, collection, resume and export |
| [Native deployment proposals](docs/DEPLOYMENTS.md) | SGLang TP4/EP4 and vLLM TP2/offload command previews with matched benchmark templates |
| [Experiment catalog](docs/EXPERIMENTS.md) | Twelve tracks, hypotheses, axes, controls, readiness and commands |
| [Parameters](docs/PARAMETERS.md) | Workload/SLO contracts and acceptance rules |
| [48-hour proposal](docs/PLAN48H.md) | Priorities, budgets and stop gates |
| [Assets and datasets](docs/ASSETS.md) | Pinned manifests, explicit bounded downloads and offline tokenization |
| [Data contracts](docs/DATA.md) | Raw journals, derived summaries, qualification and artifact inventories |
| [Research](docs/RESEARCH.md) | Primary-source support evidence and model-format distinctions |
| [Validation](VALIDATION.md) | Actual CPU checks and limitations |

Real collection requires `run --execute`, an exact checkpoint and runtime, checksum-bound inventory/weights/runtime/smoke evidence, frozen tokenizer-backed inputs and explicit quality/SLO gates. B200 execution also requires reconciled parameters, a reservation end time and an independently selected export destination. Templates deliberately block execution while these facts remain unknown. Preparation and publication do not authorize paid work or GPU runs.

The collector preserves failed, shed and unfinished outcomes; checks matched canaries before and after every configuration; and keeps the original deadline across resume. Capacity calculations require a verified final receipt, matched open-loop repetitions, usable admission-window queue evidence, task rubrics on the served workload and verified export. A stable queue with dropped traffic is insufficient. Short screens and profiled runs cannot establish sustainable capacity.

`infrx_bench/node.py`, `matrix.json` and `historical_collect.py` preserve historical reference helpers. Their execution paths and older pins need a separate review; use `bench.py` for current collection. See [PROVENANCE.json](PROVENANCE.json) and [NOTICE.md](NOTICE.md).

Results, downloaded assets and private data are ignored by Git. They may contain prompts, outputs and operational inventory; retain them privately and review a separate sharing copy. No DeepSeek benchmark measurements are included.
