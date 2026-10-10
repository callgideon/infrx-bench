# Data contracts and retention

schemas/contracts.json documents config, run metadata, attempts, retained outputs, telemetry, quality, cost, pause logs, artifact inventory, final receipt, assets and datasets. contracts.py validates this structural subset without dependencies; it is not general JSON Schema. Config/assets also invoke semantic validators. Structural validity alone cannot qualify capacity or engines.

```sh
python3 -m infrx_bench.contracts --kind config --input configs/b200-screen.proposed.json
python3 -m infrx_bench.contracts --kind attempts --input results/cpu-check/closed-c1/attempts.jsonl --jsonl
python3 -m infrx_bench.contracts --kind final --input results/cpu-check/experiment-complete.json
```

| Artifact | Contract/use |
|---|---|
| experiment.json, frozen-inputs.json, run-window.json | Byte-bound inputs and original deadline; preserve for resume |
| manifest/corpus/cases | Exact cell controls/inputs |
| scheduled.jsonl, attempts.jsonl | Raw arrivals and outcomes, including failed/shed/unfinished requests |
| telemetry.jsonl | Timestamped raw GPU/process/Prometheus observations; missing fields remain missing |
| Canary/task attempts | Retained content/reasoning and quality; other loads retain output hashes |
| driver-summary.json | Preserved original derived driver summary |
| summary.json, cell-receipt.json | Derived gates/integrity; no final qualification from cell success alone |
| collection-complete-*.json | Collection/canaries before export; not final qualification |
| Archive manifest/export receipt | Inventory, archive SHA and verified destination |
| experiment-complete.json | Final pairs, repeat groups, artifact hashes, deadline and archive/export reference |
| paused-*.json | Preserved failure/pause; partial runs remain partial |

Never rewrite raw journals/inputs/receipts. New analyses record source hashes, code/config revision, denominators and units. Resume rejects changed external inputs and artifacts, preserves interrupted attempts and stops on completed failed cells.

The final receipt is written after the collection archive/export; retain it separately with the verified archive. Hashes establish integrity/field consistency, not remote attestation or truthful operator inputs. There is no automatic prompt/host/output redactor: review a separate sharing copy and keep raw/private data out of Git. Whole-window cost includes all paid time and actual jointly quality/SLO accepted served tasks.
