# rfd4-proteina-service

A FastAPI wrapper around **RFD4-Proteina**
([RosettaCommons/RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina),
the `nvidia/RFD4-Proteina` checkpoint) that runs a binder/complex **design
campaign** and returns the produced designs.

Same shape as the other services in this repo: typed `Settings`, an injectable
backend `Protocol` (so the API is tested without a GPU or weights), one model
resident at a time, inference serialized per process. The model is loaded
**in-process** (`rfd4.Proteina.from_pretrained` + `rfd4.generate`); a request
supplies the condition spec and generation settings, and the response returns
design **file paths** on the node's shared storage (a campaign writes many
structure files).

**Status: implemented and tested.** `src/rfd4_proteina_service/server.py` +
`tests/`. ruff, mypy, and the pytest suite pass (validation, override safety,
override construction, inputs-YAML staging, output collection, failure paths)
with `rfd4_proteina` stubbed — so the suite needs no GPU/CUDA/weights. The real
server boots and serves `/health`, `/v1/design`, and `/docs`.

## Install and run

RFD4-Proteina loads in-process, so install it (with a CUDA extra) into the
serving env, then install this service — see `deploy/serve.sh`:

```bash
cd rfd4_proteina
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "rfd4-proteina[all-cuda13]"   # or all-cuda12; the model + CUDA
uv pip install -e ".[dev]"
rfd4-proteina-serve                           # http://localhost:8000
```

Preload the model at startup instead of on the first request:

```bash
RFD4_PRELOAD_MODEL=rfd4-proteina rfd4-proteina-serve
```

## Run a campaign

Provide the condition spec either inline (`inputs_yaml`) or as a path to a YAML
already staged on the node (`inputs_path`) — exactly one. The spec is the repo's
binder/condition format (contig string, `conditions` with `sel(...)`
expressions, hotspots, and an `input` structure path).

```bash
curl http://localhost:8000/v1/design -H 'Content-Type: application/json' -d '{
  "model": "rfd4-proteina",
  "inputs_yaml": "binder:\n  input: /data/5o45_cropped.cif\n  contig: \"50-120, /0, A17-131\"\n",
  "num_replicates": 2,
  "inference_experiment": "quickstart",
  "seed": 0
}'
```

Response (paths are on the node's storage):

```json
{
  "object": "design",
  "model": "rfd4-proteina",
  "run_name": "rfd4-0-1a2b3c4d",
  "output_dir": "./runs/rfd4-0-1a2b3c4d",
  "num_designs": 2,
  "designs": [
    {"id": "design_0", "structure_path": ".../inference_samples/design_0.cif", "format": "cif"}
  ]
}
```

### Request fields

| Field | Default | Notes |
| --- | --- | --- |
| `model` | `rfd4-proteina` | → `nvidia/RFD4-Proteina` |
| `inputs_yaml` | — | condition-spec YAML as text (exactly one of yaml/path) |
| `inputs_path` | — | path to a staged condition-spec YAML |
| `run_name` | auto | unique name; auto = `rfd4-<seed>-<hex>` |
| `num_replicates` | 1 | designs per run (1–100) |
| `inference_experiment` | `quickstart` | `quickstart` skips scoring + compile warm-up |
| `seed` | 0 | RNG seed |
| `extra_overrides` | `[]` | extra Hydra `key=value` tokens (validated; reserved keys blocked) |

`extra_overrides` reaches the same override list as the `rfd4-generate` CLI;
the service-controlled keys (`inputs`, `out_dir`, `num_replicates`, `seed`,
`inference_experiment`) are rejected.

## HTTP contract

```
POST /v1/design  -> {object:'design', run_name, output_dir, num_designs, designs[]}
GET  /health
```

## Design notes & things to confirm

- **The campaign is wrapped as** `rfd4.generate(model, overrides=["inputs=…",
  "out_dir=<run>", "num_replicates=…", "inference_experiment=…", "seed=…"])`,
  mirroring the `rfd4-generate` CLI.
- **⚠ Output layout.** Designs are collected from
  `<out_dir>/inference_samples/*.{cif,pdb}` (with a fallback to any structure
  file under the run dir), per the repo README. Confirm against the repo's
  inference docs; collection is isolated in `ProteinaBackend._collect_designs`.
- **Input schema.** The condition-spec YAML is passed through verbatim rather
  than modeled field-by-field — the repo documents one worked example, not a
  formal schema — so `inputs_yaml`/`inputs_path` keep the full expressiveness
  (contig, conditions, hotspots) without this service guessing the grammar.
- **Blocking vs async.** Campaigns can be long; the synchronous `POST
  /v1/design` under the lock is a first cut. A job API (submit → `job_id` →
  poll) is the clear next step for long runs.
- **Safety.** `run_name`, `inference_experiment`, and every override token are
  charset-validated; nothing from the request is passed through a shell.

## What copies from the other services vs. what is new

| Box | Action |
| --- | --- |
| `Settings` | copy; env prefix `RFD4_`; hf_token/revision, runs_root |
| backend `Protocol` | → `DesignBackend`; `embed`/`fold` → `generate(request)` |
| `load`/`unload` | copy; `rfd4.Proteina.from_pretrained` instead of ESM/Fold |
| request/response models | new: `DesignRequest` / `DesignResponse` / `DesignItem` |
| `create_app` / lock / routes / `main` | copy ~verbatim; route `/v1/design` |
| packaging envelope | copy; scripts → `rfd4-proteina-serve`; model dep is CUDA-only |
| test strategy | copy: `FakeBackend` + injected fake `rfd4_proteina` module |
