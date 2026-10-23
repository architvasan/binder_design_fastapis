# rfd4-proteina-service

A FastAPI wrapper around **RFD4 / Proteina-Complexa** (RosettaCommons
RFD4-Proteina, as ported to Aurora in the `aurora-proteina-complexa` recipe)
that runs a **binder-design campaign** for a configured target and returns the
produced designs.

This is **archetype C** — an async design job — so it differs from the sync
embedding/fold/sample services in this repo:

- it launches the `complexa` Hydra pipeline as a **subprocess per request**
  (Hydra + `main()` carry global state; the recipe runs a fresh process per
  call);
- it **forces `gen_njobs=1` / `eval_njobs=1`** — the pipeline's parallel branch
  resets `ZE_AFFINITY_MASK` and would fight the node's tile pinning;
- it **serializes** campaigns with a lock (one job per assigned tile); and
- it returns design **file paths** on the node's shared storage, not inline
  structures (a campaign produces many large files).

**Status: implemented and tested.** `src/rfd4_proteina_service/server.py` +
`tests/`. ruff, mypy, and the pytest suite pass (validation, override safety,
argv construction, output collection, and the failure paths) with the
subprocess faked — so the suite needs no Aurora/XPU/weights. The real server
boots and serves `/health`, `/v1/design`, and `/docs`.

## How it runs

The server process must run **inside the activated Proteina-Complexa
environment** on a compute node (so `complexa` is on `PATH` and the framework's
checkpoint/cache env vars are set). Use `deploy/aurora_serve.sh`, which sources
the recipe's activation and launches uvicorn on one tile. From your laptop,
`ssh -L 8000:<node>:8000 …` to reach it.

The canonical invocation it wraps (from the recipe):

```bash
complexa design configs/search_binder_local_pipeline.yaml \
  ++run_name=<unique> ++generation.task_name=<target> \
  ++gen_njobs=1 ++eval_njobs=1 ++seed=<n>
```

## Run a campaign

The `task_name` is a target key already configured in the repo's target/config
system (e.g. `02_PDL1`) — PDB, chain selection, hotspots, and binder length
live there. Defining a *new* target means editing repo configs and is out of
scope for a single request.

```bash
curl http://localhost:8000/v1/design -H 'Content-Type: application/json' -d '{
  "task_name": "02_PDL1",
  "pipeline": "binder",
  "seed": 42
}'
```

Response (paths are on the node's storage):

```json
{
  "object": "design",
  "pipeline": "binder",
  "task_name": "02_PDL1",
  "run_name": "02_PDL1-42-1a2b3c4d",
  "output_dir": "/.../runs/02_PDL1-42-1a2b3c4d",
  "returncode": 0,
  "num_designs": 8,
  "designs": [
    {"id": "design_0", "structure_path": ".../inference_samples/design_0.cif", "format": "cif"}
  ],
  "log_tail": "…"
}
```

### Request fields

| Field | Default | Notes |
| --- | --- | --- |
| `task_name` | — | configured target key (`[A-Za-z0-9._-]+`) |
| `pipeline` | `binder` | allowlisted Hydra config |
| `run_name` | auto | unique name; auto = `<task>-<seed>-<hex>` |
| `seed` | 42 | RNG seed |
| `extra_overrides` | `[]` | extra Hydra `key=value` tokens (validated; reserved keys blocked) |

`extra_overrides` is for power users — e.g. a generation sample-count key once
confirmed against the repo configs. Tokens are validated to `+?+?key=value`
with a safe charset, and the service-controlled keys (`run_name`, `seed`,
`gen_njobs`, `eval_njobs`, `generation.task_name`) are rejected.

## HTTP contract

```
POST /v1/design  -> {object:'design', run_name, output_dir, returncode, num_designs, designs[], log_tail}
GET  /health
```

## Design notes & things to confirm on-cluster

- **⚠ Output layout / count key.** The service locates designs by globbing
  `inference_samples/*.{cif,pdb}` under the run dir (with a fallback to any
  structure file), and passes `out_dir=<runs_root>/<run_name>`. Confirm both
  against `configs/search_binder_local_pipeline.yaml`,
  `configs/pipeline/binder/binder_generate.yaml`, and `docs/INFERENCE.md` — the
  recipe says to read those before a campaign. Collection is isolated in
  `ProteinaBackend._collect_designs`; the generation sample-count override goes
  in `extra_overrides` until its key is confirmed.
- **Env (from the recipe's `profile.json`):** repo
  `/lus/flare/.../Proteina-Complexa`, env `/lus/flare/.../envs/proteina-complexa`
  (Python 3.12, Torch **XPU**). Activation exports `CKPT_PATH`, `AF2_DIR`,
  `ESMFOLD_DIR`, `RF3_CKPT_PATH` and the project caches — verify these exist
  before sampling. Generation + ProteinMPNN use Torch XPU; AF2 reward uses
  JAX/OpenXLA (`sycl`); keep intended CPU helpers distinct from a silent CPU
  fallback of the generative model.
- **Blocking vs async.** Campaigns take minutes to hours, so the synchronous
  `POST /v1/design` is a first cut. The clear next step is a job API (submit →
  `job_id` → poll `/v1/design/{job_id}`) so clients don't hold a long request;
  the lock + subprocess structure already isolates one job at a time.
- **Safety.** The pipeline is launched with a fixed argv list and `shell=False`;
  names and overrides are charset-validated, so request values cannot inject
  shell or extra arguments.

## What copies from the other services vs. what is new

| Box | Action |
| --- | --- |
| `Settings` | copy; env prefix `PROTEINA_`; repo/command/runs_root/timeout |
| backend `Protocol` | → `DesignBackend` with `generate(request)` (no resident model) |
| inference seam | new: subprocess launch of the `complexa` pipeline |
| request/response models | new: `DesignRequest` / `DesignResponse` / `DesignItem` |
| `create_app` / lock / routes / `main` | copy ~verbatim; route `/v1/design` |
| packaging envelope | copy; scripts → `rfd4-proteina-serve`; no model dep |
| test strategy | copy idea: `FakeBackend` + a faked `subprocess.run` |
