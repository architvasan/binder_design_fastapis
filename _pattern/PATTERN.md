# The service pattern

Distilled from [`genslm-esm-service`](https://github.com/ramanathanlab/genslm-esm-service)
(`src/genslm_esm_service/server.py`, ~380 lines). Every service in this repo is
this skeleton with a model-specific contract and backend.

## The six layers

| Layer | What it is | Template anchor | Per-model work |
| --- | --- | --- | --- |
| **0 Spine** | an allowlist dict that doubles as per-item metadata | `MODEL_LAYER_COUNTS` (server.py:41) | redefine: model versions, or input entity domain |
| **1 Config** | `Settings(BaseSettings)` from `SERVICE_*` env vars, typed+validated | `Settings` (server.py:57) | same shape, different fields |
| **2 Contract** | pydantic request/response, `extra='forbid'`, layered validators | `EmbeddingsRequest` (server.py:86) | **rewrite** per model |
| **3 Seam** | backend `Protocol` (`load`/`unload`/`embed`) → injectable, testable | `EmbeddingBackend` (server.py:171) | copy; rename the work method |
| **4 Lifecycle+inference** | one model resident; batched inference | `GenslmEmbeddingBackend` (server.py:201) | copy load/unload; **rewrite** the work method |
| **5 Composition** | `create_app` factory, lifespan, `asyncio.Lock`+`to_thread`, routes | `create_app` (server.py:305) | copy; new route(s); maybe async jobs |
| **6 Envelope** | uv, ruff, mypy, pytest, release-please, dependabot, entry point | `pyproject.toml` | copy ~verbatim; swap dep + names |

## What copies ~verbatim (every service)

- `Settings` shape and the preload validator.
- The backend `Protocol` seam and the `FakeBackend` test strategy — this is what
  lets the whole API be tested with no GPU and no weights.
- `create_app(settings, backend)` factory + `lifespan` (preload on start, unload
  on stop).
- One-model-resident `load()` / `unload()` (swap → free old; CUDA→bf16/CPU→fp32;
  `gc.collect()` + `torch.cuda.empty_cache()`).
- `asyncio.Lock` + `asyncio.to_thread` to serialize GPU work while keeping the
  event loop (and `/health`) responsive. **One service process per GPU.**
- The entire packaging / CI / release envelope.

## What each archetype rewrites

### Archetype A — sync vector (ESM-C)
Almost nothing changes. Keep the OpenAI `/v1/embeddings` contract, the blocking
`async with lock: to_thread(embed)` route, vector-shaped response. ESM-C is the
same family the template already serves.

### Archetype B — async fold job (ESMFold2, Chai-1, Boltz-2)
- **Contract:** request is chains/sequences (+ optional ligands, MSA,
  constraints, sampling params); response is **structure file(s) + confidence
  scores**, not vectors. No OpenAI shape.
- **Work method:** `embed` → `fold`. It does file I/O — write a FASTA/YAML
  input, call the model's `run_inference`, read back CIF/PDB text + scores.
- **Concurrency:** runs are minutes. Replace the single blocking POST with a
  **job API**: `POST /v1/fold/jobs → {job_id}` then `GET /v1/fold/jobs/{id}`.
  The `asyncio.Lock` still serializes the GPU; the job store decouples the HTTP
  request from the run.
- **Transport:** decide how structures leave the service — base64 CIF in JSON
  (simple, bloats payload), or write to shared storage and return a path/URL
  (better for large multi-sample outputs). Pick one repo-wide.

### Archetype C — async design job (ProteinMPNN, RFdiffusion)
Same async-job + file-transport machinery as B. Differences:
- **ProteinMPNN:** input is a backbone PDB + fixed/tied positions + sampling
  temperature; output is **sequences** (FASTA) with per-sequence scores. Output
  is small, so it can stay sync for a single design, but batch jobs want the job
  API.
- **RFdiffusion:** input is a contig map + hotspot residues (+ optional input
  PDB); output is **backbone PDB(s)**. Always a job (diffusion is slow).

## The async-job addition (B and C)

Not in the template — the one genuinely new subsystem. Minimum viable:

```
POST /v1/<task>/jobs      -> validate request, enqueue, return {job_id, status:'queued'}
GET  /v1/<task>/jobs/{id} -> {status: queued|running|done|error, result?|error?}
```

A single background worker pulls jobs and runs them under the existing
`inference_lock` (one GPU job at a time). Start with an in-memory job store +
one worker task in `lifespan`; graduate to a real queue only if needed. Keep the
synchronous `POST /v1/<task>` too for quick single-item calls and tests.

## Build order recommendation

1. **esmc** — proves the pattern copies cleanly (archetype A, lowest risk).
2. **chai1** — the worked archetype-B example (contract + fold + job API).
3. **boltz2, esmfold2** — clone chai1's archetype-B scaffold, swap the backend.
4. **proteinmpnn, rfdiffusion** — archetype C on the same job machinery.
