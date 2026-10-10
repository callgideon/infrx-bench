# Runbook

## 1. Verify the client locally

```sh
python3 -m unittest discover -s tests -v
python3 bench.py rehearse --out results/cpu-check
python3 bench.py plan --config configs/b200-screen.proposed.json
python3 bench.py sweep --config configs/context-thinking-sweep.proposed.json --out results/context-plans
```

These are CPU checks and file preparation. The controlled loopback fixture qualifies no model, engine, hardware fit, cache optimization or latency. Core commands require Python 3.10+; no installation is automatic. `bench.py --help` and the asset/dataset modules expose their separate command sets.

## 2. Reconcile the proposed experiment

Select tracks from [EXPERIMENTS.md](EXPERIMENTS.md). Freeze workload mixture, actual context bins, answer/reasoning budgets, thinking semantics, sampling, SLO acceptance fraction, quality floors/degradation allowance, user cadence, repetitions, reservation end and export location. The checked-in numeric values are proposals. Realistic coding/data/tool tasks need a frozen rubric; this client never executes model-generated code or tools.

Use the [asset workflow](ASSETS.md) to review pinned model/tokenizer/dataset manifests. Asset preflight checks local capacity and manifest consistency; it does not prove remote download access, fit or engine support. Large downloads require separate explicit execution authorization. Download receipts feed tokenizer preparation and weight evidence.

## 3. Qualify one exact deployment

On an already approved and authenticated executor, collect read-only inventory:

```sh
python3 bench.py preflight --out results/device-inventory.json
```

Inventory output is observation, not a `VERIFIED` qualification receipt. Check memory, GPU count, CPU RAM/offload, device/runtime support, disk and other processes. Use an explicitly approved launch procedure for the exact immutable engine/checkpoint; no server launch is included in `bench.py run`. Requalify every changed GPU topology, engine/precision, context maximum or request mode.

Create a private `infrx-qualification/2` document with exact `identity` and `runtime`, status `QUALIFIED_WITH_EVIDENCE`, and checksum-bound `artifacts` entries `{kind, path, sha256}` for inventory, weights, runtime and smoke. See `policy.py` for the enforced fields. Weight evidence includes repo/revision and verified file hashes. Runtime inspection records the actual runtime, configured context and exact `request_modes` mapping; salt and text-only capabilities need observed evidence. Smoke evidence requires actual successful generation with the same identity/runtime and no failures. Inventory identifies the observed GPU model, full node count and allocated GPUs.

These are operator evidence records checked for byte integrity and consistency, not remote cryptographic attestation. Unknown facts stay blocked; do not manufacture successful statuses from a template. Bind tokenizer-backed corpora to the same checkpoint. Tokenization is offline; server usage remains authoritative.

## 4. Collect a short pilot

```sh
python3 bench.py plan --config /path/to/reviewed-config.json
python3 bench.py run --config /path/to/reviewed-config.json --out results/b200-pilot --execute
```

Run only after explicit GPU/run approval. The reviewed B200 config must set `parameters_reconciled`, an aware `end_utc`, frozen workload provenance, positive acceptance/quality gates, qualification file and export/retention evidence. The proposed files intentionally fail execution until completed. No paid service action is implicit.

Every scored cell is bracketed by matching canaries. Warm replay has a separate unscored prime. Any accounting, quality, SLO or rejection gate failure stops the remaining queue. Compare optimized arms with matched retained baseline canaries; preserve baseline artifacts privately. Do not let different TP/EP/offload bundles masquerade as a pure engine comparison.

## 5. Confirm selected candidates

Short screens rank hypotheses only. A capacity cell must be unprofiled open-loop, use at least three matched repetitions, have adequate admission-window queue observations and tail samples, and score task rubrics on the actual served workload. Rejections and poor quality cannot be hidden behind a flat queue. Three proposed one-hour repeats and a winner soak need manually composed bounded runs: the collector caps each cell at one hour. Allocate export reserve before starting another cell.

## 6. Resume or export

Resume with the original config/code/inputs, `--resume` and unchanged evidence. The saved absolute deadline is never extended. Interrupted cells receive new attempt directories; completed failed cells require diagnosis. Do not remove or edit failed records. The supervised client watchdog stops only its own process group.

Real B200 collection archives and verifies the selected destination before issuing its final qualification receipt. Standalone export tools also require an aware absolute deadline:

```sh
python3 bench.py archive results/b200-pilot results/b200-pilot-review.tar.gz --deadline-utc 2026-10-12T18:00:00Z
python3 bench.py export results/b200-pilot-review.tar.gz /path/to/independent-destination/b200-pilot-review.tar.gz --deadline-utc 2026-10-12T18:00:00Z
python3 bench.py verify-export results/b200-pilot-review.tar.gz /path/to/independent-destination/b200-pilot-review.tar.gz --deadline-utc 2026-10-12T18:00:00Z
```

Dates above are examples; set the real remaining deadline. A second path on the same disk is not disaster recovery. Filesystem copies are supported; remote/object-store retention must be arranged by the operator. The collection archive precedes the final receipt, which references the verified archive; retain that final receipt as well. Raw operational artifacts remain private. Inspect and sanitize a separate sharing copy.

## 7. Analyze cost honestly

Use `bench.py cost --help` and provide actual full-node price and user rate. Pass `--run` with a verified complete experiment to qualify a capacity result; raw summary labels alone are insufficient. `reservation-cost` includes the entire paid window and actual accepted-task/active-user denominators. Session transport completion is not task correctness. Future runtimes or device support remain separate gates and do not block publishing this CPU-tested preparation.
