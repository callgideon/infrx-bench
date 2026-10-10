# Parameters and evidence contracts

All numeric settings in real-model examples are proposals. Workload distribution, user cadence, SLOs, repetitions, reservation dates, prices, thinking modes and agreed output budgets remain operator inputs. CPU values are test-only; see EXPERIMENTS.md for proposed ranges/readiness.

## Questions

1. What is the smallest deployment that fits the model, passes a generation/task smoke test, and meets the chosen SLO? These are three separate milestones.
2. How many users can a configuration sustain for the agreed requests per user per minute and actual prompt/output/think-time distribution?
3. What is the whole-node cost per accepted request, completed task/session and active user hour? Can accepted quality and latency reduce this cost compared with the baseline?
4. Do prefix/session affinity, cache isolation, admission grouping and queue ordering change measured cache reuse, queue growth and task completion under the same workload?

Concurrency is a cap on active requests. It is not a user count. Open-loop arrival rates stay independent of response completion; arrivals exceeding the client inflight cap are recorded as shed. Session traces have explicit arrivals and response-dependent think times. Reports must describe which user model produced the offered load.

## Axes

| Axis | Contract |
|---|---|
| Model/precision | Exact repository and revision; native mixed formats, a separately qualified NVFP4 export, and blocked/unverified NVFP8 are distinct arms |
| Engine | Exact vLLM or SGLang image/source and model/hardware support evidence; kernel backend names do not qualify TensorRT-LLM as an engine |
| GPUs | Screen supported 2/4/8-GPU configurations; require fit, smoke, quality and SLO gates for each; do not presume that all fit |
| Context | Actual tokenizer/server prompt tokens and configured maximum are separate fields; a lower maximum need not reduce paged-KV allocation |
| Output/thinking | Keep output cap, thinking mode and sampling matched within a comparison; capture reasoning separately; forced-length `ignore_eos` stress is distinct from natural task completion |
| Cache | `cold-distinct` prepends unique prefixes; `warm-replay` primes exact explicit prompts; neither proves a flush or hit. Session traces preserve history. Tenant salts require verified engine support |
| Routing | Round robin, session affinity and prefix affinity select endpoints deterministically; record actual engine cache/queue counters before attributing a gain |
| Queue/batch | FCFS or bounded-wait prefix grouping and a driver admission window; driver grouping is not a claim about engine batches |
| SLOs | Arrival-to-first generated event and arrival-to-end, including driver wait; visible TTFT is also reported. Chunk intervals are not token ITL |
| Profiling | Separate diagnostic arms; all profiled results are excluded from capacity claims |

The historical tokenizer corpus helper records content-token bins and tokenizer identity. Chat template overhead and cold prefixes change server prompt length: report actual usage rather than treating character counts as tokens. Public example prompts are functional fixtures, not a realistic workload distribution.

## Gates and accounting

Every configuration/repetition has matched pre/post canaries. Optimized arms additionally name a retained baseline canary and a maximum permitted degradation. All canary task IDs must appear. The ten JSON tasks cover instruction following and small coding/reasoning checks; this is a regression canary, not an intelligence benchmark. Broader task correctness needs an agreed frozen rubric and datasets. Session completion alone is not task correctness. Capacity requires rubrics on the actual served workload and a verified final receipt/export, not raw summary flags.

SSE responses must match the served model, terminate properly, preserve bounded output and include valid usage. Failure, shedding, unfinished journals, protocol/accounting errors and canary regressions remain in evidence and stop the queue. Failed cells require diagnosis before another run directory is created. Resume checks the experiment, bundle and artifact hashes and never overwrites interrupted attempts.

Closed-loop trials provide stress evidence. Sustainability requires qualified open-loop queue metrics, adequate duration/repetitions, matched workload and accepted-request SLO goodput. The built-in finite-window slope check is exploratory; retain raw observations and do not extrapolate a short stable trial to indefinite operation. p95 needs at least 60 successes and p99 at least 300 under the driver's exploratory sample-count rule; missing tails do not pass an SLO by default.

## Cost

`cost` requires the real full eight-GPU node hourly price and requests per user per minute; no provider price is guessed. Full-node observed cost is charged regardless of allocated GPU count. A fractional allocation price is explicitly hypothetical. Accepted goodput divided by user cadence is a capacity scenario; mock/proxy/closed-loop/profiled results cannot qualify sustainable B200 users.

Use `reservation-cost` with elapsed reservation hours, accepted tasks and active user hours to account for loading, idle GPUs, failed work, retries, offload, profiling, export and gaps. Cell-level token/request prices alone omit these costs. CPU/memory/offload overhead is a separate nonnegative hourly input. Cost per completed session requires an actual completed-task/session denominator, not every HTTP success.
