# Assets, datasets and dependencies

Client/tests: Python 3.10+ standard library. Serving engines and offline transformers/engine tokenizer libraries must already be installed in an inspected immutable runtime. Imports/plans install nothing. Missing tokenizer dependencies are explicit errors; no network fallback or automatic remote-code trust.

`manifests/models/` holds exact checkpoint revisions, public access/license metadata, sizes and SHA-256. LFS hashes come from pinned upstream metadata; non-LFS files are hashed from bounded bytes, capped at 16 MiB each. Manifest creation reads metadata/docs/vocabulary only, never weights. Recheck support, access and actual license before use. A manifest is not download permission.

```sh
python3 -m infrx_bench.assets preflight --manifest manifests/datasets/gsm8k-test.json --root data/downloads/gsm8k
python3 -m infrx_bench.assets --help
python3 -m infrx_bench.datasets --help
```

Preflight checks manifest consistency, local disk and missing bytes; it does not probe remote access or prove fit. After explicit download authorization and license review:

```sh
python3 -m infrx_bench.assets download --manifest manifests/datasets/gsm8k-test.json --root data/downloads/gsm8k --execute --license-reviewed --maximum-bytes 1000000 --seconds 300
python3 -m infrx_bench.datasets import --manifest manifests/datasets/gsm8k-test.json --source data/downloads/gsm8k/test.jsonl --count 100 --seed 20261010 --out data/private/gsm8k-tasks.json
```

Create the private output parent first. The pinned test file is 749,738 bytes; rows are not committed. Import samples deterministically and retains version/license/source SHA/selected IDs. Extracting the answer after `####` and adding a numeric-only instruction creates an arithmetic/instruction variant, not official GSM8K evaluation or representative developer traffic. JSON tasksets with explicit exact-text/JSON rubrics are also supported; arbitrary code/rubrics are never executed.

Downloads retain partials and require exact HTTP Range responses on resume, original manifest/byte budget/deadline, size and final SHA-256. Existing mismatched files are preserved. Disk reserve, license review and explicit --execute are mandatory. Symlink paths and credential URLs are rejected; use canonical /private/tmp or /private/var paths on macOS when necessary. Gated/private access is manual. A killed client may leave download.lock: inspect and confirm no active client before removing only that lock; never change state to extend time.

`staging.json` binds verified files to repo/revision. Offline task tokenization verifies those hashes, preserves prompts/rubrics and checks chat-template/output reserve:

```sh
python3 -m infrx_bench.datasets tokenize --tasks data/private/gsm8k-tasks.json --tokenizer-root assets/native-tokenizer --maximum-context 16384 --output-reserve 512 --out data/private/gsm8k-tokenized.json
```

Server usage remains authoritative. `python3 -m infrx_bench.corpus --help` exposes a separate bounded synthetic token-bin generator; use small counts for long contexts. Synthetic stress does not provide task correctness or representative traffic. Tests use bounded local/loopback bytes. Massive transfers, actual engine tokenizer compatibility and representative dataset qualification remain operational prerequisites.
