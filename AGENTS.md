# Repository instructions

## Goal and current scope

Prepare reproducible inference benchmarks for DeepSeek-V4.1-Flash on a reserved eight-B200 node. Read README.md, VALIDATION.md, docs/RUNBOOK.md and docs/EXPERIMENTS.md before changing collection or interpreting results. AGENTS.md is the canonical shared file; CLAUDE.md imports it for Claude Code. Codex and OpenCode use AGENTS.md directly. Keep shared instructions here.

Current preparation includes CPU-tested transport, scheduling, bounded asset staging, deterministic task import, per-configuration quality gates, resume identity, export verification and proposed experiment plans. Device/runtime qualification and GPU measurements are separate work. No pending device connection is required to improve or publish this preparation.

## Execution boundaries

- Preparation, inspection and tests do not authorize GPU runs, model-weight downloads, launching/stopping services, paid infrastructure, account changes or package installation on remote machines. Obtain task-specific authorization for those actions. Honor any current pause; do not troubleshoot remote devices as an incidental prerequisite.
- Never infer a hostname, price, reservation end, supported thinking flag or model fit. Unknown runtime/hardware facts must remain explicit blockers. Do not fabricate qualification receipts.
- Preserve existing files and other repositories. Do not reset, clean, force-push, format disks, clear caches or kill external services. A watchdog may stop only the collector's own isolated process group.
- Use existing approved authentication. Never put passwords/tokens in prompts, scripts, arguments, manifests, logs or committed files. BENCH_API_KEY is read from the local process environment and is not retained.

## Code and checks

The collection client and CPU tests use Python 3.10+ and the standard library. Keep serving engines, tokenizer libraries and profiling dependencies external. Do not silently install or fall back to a different runtime/checkpoint.

```sh
python3 -m unittest discover -s tests -v
python3 bench.py rehearse --out results/unique-cpu-check
python3 bench.py plan --config configs/b200-screen.proposed.json
python3 -m infrx_bench.contracts --kind config --input configs/b200-screen.proposed.json
```

Use a fresh output path; preserve failed attempts. Check command help when adding options and update documentation with tested examples. Regression tests should exercise failures and evidence gates, not just mirror implementation.

Look first for an available Semgrep MCP server and scan generated code/commands before execution when the enclosing workspace instructions require it. For the initial preparation/publication task, the user explicitly authorized the installed Semgrep CLI as a substitute for the unavailable semgrep_scan tool. This is a task-specific exception, not a waiver for future tasks. Respect applicable enclosing instructions and request a specific substitution if required tooling is unavailable. Record actual scan scope, findings and omissions in VALIDATION.md.

## Evidence rules

- Freeze config, code/data bundle, corpus, canary, baseline and qualification artifact bytes. Any change requires a new experiment identity; never edit raw evidence to pass resume.
- Each changed configuration/repetition requires matched before/after canaries with the same output cap, thinking semantics, sampling and context. Optimized arms additionally require a matched retained baseline and explicit degradation allowance.
- Exact model repo/revision, precision and runtime/hardware evidence determine supported arms. CPU/proxy labels cannot qualify a real model. Block unqualified NVFP8/TRT-LLM candidates rather than relabeling them.
- Offered arrivals are independent of response completion. Record failures, shedding, queue wait and unfinished requests. Evaluate latency from scheduled arrival. Driver grouping is not engine batching; chunk intervals are not token ITL; cache protocols do not establish a hit without counters.
- Distinguish actual input tokens, chat-template overhead and configured context maximum. Server usage is authoritative. Synthetic prompts are stress fixtures, not representative production data.
- Capacity requires qualified unprofiled open-loop repetitions, sufficient tails/queue observations, served-task quality and SLO gates together, final canaries and verified export. Never multiply transport throughput by unrelated offline accuracy to claim accepted-task goodput.
- Charge actual full-node reservation cost, including setup, idle GPUs, failures, retries and export. Fractional GPU prices and user-cadence calculations are labeled scenarios; never guess prices or user rates.
- Keep the saved absolute deadline across resume. Stop admitting work before the collection deadline/export reserve. Interruptions retain evidence and do not become completed experiments.

## Publication

Commit source, proposed configs, public metadata manifests and reviewed documentation only. Results, weights, private workloads, host identifiers, endpoints and credentials remain outside Git. Before an authorized push, fresh-fetch, preserve others' work, run relevant checks, review staged content for secrets/privacy/licensing, push without force and verify the remote SHA and any CI. Publishing preparation does not require future GPU acceptance gates to have passed; document those gates honestly.
