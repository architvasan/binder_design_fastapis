# bioemu2-service

A FastAPI wrapper around **BioEmu2** (`microsoft/bioemu2`) that samples an
equilibrium conformational ensemble for a single protein sequence and returns
the ensemble topology (mmCIF) plus the sampled trajectory.

Same shape as the other services in this repo: typed `Settings`, an injectable
backend `Protocol` (so the API is tested without a GPU or weights), one model
resident at a time, inference serialized per process. BioEmu2 is **monomer
only**; sampling runs **synchronously** and returns the topology inline. The
binary `.xtc` trajectory is opt-in and returned base64-encoded.

**Status: implemented and tested.** `src/bioemu2_service/server.py` + `tests/`.
ruff, mypy, and the pytest suite pass (validation, parameter forwarding, the
sample→file→response mapping, and the missing-output error path) via a stubbed
`bioemu2` module. The real server boots and serves `/health`, `/v1/sample`,
and `/docs`.

## Install and run

BioEmu2 is **not on PyPI** — install it from source into the same environment,
then install this service:

```bash
# 1. BioEmu2 itself (see github.com/microsoft/bioemu2)
pip install "git+https://github.com/microsoft/bioemu2.git#egg=bioemu2[cuda]"

# 2. this service
cd bioemu2
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
bioemu2-serve                    # http://localhost:8000
```

Preload the model at startup instead of on the first request:

```bash
BIOEMU2_PRELOAD_MODEL=bioemu2 bioemu2-serve
```

## Sample an ensemble

```bash
curl http://localhost:8000/v1/sample -H 'Content-Type: application/json' -d '{
  "model": "bioemu2",
  "sequence": "GYDPETGTWG",
  "num_samples": 10
}'
```

Response (`topology` is mmCIF text; add `"return_trajectory": true` to also get
the base64 `.xtc`):

```json
{
  "object": "sample",
  "model": "bioemu2",
  "format": "mmcif",
  "topology": "data_...\n",
  "num_samples": 10,
  "trajectory_xtc_base64": null,
  "usage": {"sequence_length": 10, "num_samples": 10}
}
```

### Request fields

| Field | Default | Notes |
| --- | --- | --- |
| `model` | — | `bioemu2` |
| `sequence` | — | one protein chain (monomer only) |
| `num_samples` | 10 | conformations to draw (1–1000) |
| `mode` | `equilibrium` | sampling mode |
| `return_trajectory` | false | include base64 `.xtc` in the response |

## HTTP contract

```
POST /v1/sample  -> {object:'sample', topology, num_samples, trajectory_xtc_base64?, usage}
GET  /health
```

## Design notes

- **Monomer only** — BioEmu2 samples a single protein, so the request takes one
  `sequence` (no chain list, unlike the ESMFold2 service).
- **Backend call surface:** the ensemble is produced by
  `EmuSession(System.from_sequence(seq), model=..., mode=...).sample(num_samples=...)`
  and written with `save_frames(..)` to `topology.cif` + `samples.xtc`, which
  the service reads back.
- **⚠ Confirm `System.from_sequence`** against `bioemu2/docs/system.md`. The
  `bioemu2` README demonstrates `System.from_pdb_id(...)` and says systems can
  also be built from sequences, but does not show the exact sequence
  constructor. It is isolated in `BioEmu2Backend.sample` (one line) and is the
  only call that depends on the real package surface; fix it there if the name
  differs. Everything else is covered by the stubbed-module tests.
- **MSA:** BioEmu2 retrieves/embeds via an inlined ColabFold path (AlphaFold2
  weights ~3.5 GB download to `~/.cache/colabfold/` on first use). On an
  offline compute node, pre-warm that cache or supply an A3M — same offline
  concern as the other GPU services.
- **Transport:** topology inline, trajectory opt-in base64. `.xtc` is compact,
  but for very large `num_samples` consider a shared-storage path (archetype B)
  instead of inflating the JSON.

## What copies from the ESMFold2 service vs. what is new

| Box | Action |
| --- | --- |
| `Settings` | copy; env prefix `BIOEMU2_`; drop `hf_token`/`device` |
| `FoldBackend(Protocol)` | → `SampleBackend`; `fold` → `sample` |
| `load`/`unload` | copy; `bioemu2.load_model()` instead of `from_pretrained` |
| request/response models | new: `SampleRequest`/`SampleResponse` (single sequence) |
| `create_app` / lock / routes / `main` | copy ~verbatim; route `/v1/sample` |
| packaging envelope | copy; scripts → `bioemu2-serve`; bioemu2 dep is source-only |
| test strategy | copy: `FakeBackend` + injected fake `bioemu2` module |
