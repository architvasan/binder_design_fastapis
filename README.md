# binder_design_fastapis

FastAPI inference microservices for the models in our binder-design stack, each
built on the **same pattern** proven in
[`genslm-esm-service`](https://github.com/ramanathanlab/genslm-esm-service):
a thin, well-tested HTTP wrapper around a pinned model library, with typed
config, an injectable backend, one-model-at-a-time GPU residency, and a full
packaging / CI / release envelope.

One service per model, one directory per service. The shared skeleton and the
two service archetypes are documented in [`_pattern/`](_pattern/PATTERN.md).

## Services

| Dir | Model | Task | Input → Output | Archetype |
| --- | --- | --- | --- | --- |
| [`esmc/`](esmc/README.md) | ESM-C | representation / LM | sequence → embedding vector | **A** sync vector |
| [`esmfold2/`](esmfold2/README.md) | ESMFold (v1/v2) | single-seq structure prediction | sequence → PDB + pLDDT | **B** async fold |
| [`chai1/`](chai1/README.md) | Chai-1 | co-folding (complex) | chains (+ligand/MSA/constraints) → CIF + scores | **B** async fold |
| [`boltz2/`](boltz2/README.md) | Boltz-2 | co-folding + affinity | chains (+ligand/MSA) → CIF + scores + affinity | **B** async fold |
| [`proteinmpnn/`](proteinmpnn/README.md) | ProteinMPNN | inverse folding | backbone PDB (+fixed pos) → sequences | **C** async design |
| [`rfdiffusion/`](rfdiffusion/README.md) | RFdiffusion | backbone generation | contigs/hotspots (+PDB) → backbone PDB(s) | **C** async design |

## The three archetypes

- **A — sync vector service (ESM-C).** Nearly a copy of `genslm-esm-service`:
  tiny JSON in, a float vector out, one blocking request at a time. OpenAI-style
  `/v1/embeddings` contract applies directly.
- **B — async fold job (ESMFold2, Chai-1, Boltz-2).** Sequence(s) in, a
  **structure file (PDB/CIF) + confidence scores** out. Runs take seconds to
  minutes, so the contract is submit → poll, not a blocking POST. Output is a
  file, so transport is a decision (base64 in JSON, or a returned artifact path).
- **C — async design job (ProteinMPNN, RFdiffusion).** A spec or an input
  structure in, **sequences or backbones** out. Same async-job + file-transport
  machinery as B; different request/response schema.

See [`_pattern/PATTERN.md`](_pattern/PATTERN.md) for what copies verbatim across
all of them and what each archetype rewrites.

## Status

Drafting. Each directory currently holds a **design doc** (`README.md`) — the
intended contract, the backend API it wraps, and what copies vs. what is new.
Runnable code skeletons come next. `esmc/` and `chai1/` are the two worked
archetypes; the other four reference them.

## Conventions (inherited from the template)

- `uv` + committed lockfile; model library pinned by version/commit.
- Typed `Settings(BaseSettings)` from `SERVICE_*` env vars.
- Injectable backend `Protocol` → the full API is tested with a fake backend,
  no GPU and no model download.
- `create_app(settings, backend)` factory; `lifespan` preload/unload; one model
  resident per process; **one service process per GPU**.
- ruff + mypy + pytest (coverage floor) in pre-commit and CI; release-please for
  versioning; Dependabot with lockfile refresh.
