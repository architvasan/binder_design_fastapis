# rfdiffusion-service (design draft)

Archetype **C** — async design job (structure output). De novo backbone
generation: a spec (contig map + hotspots, optional input PDB) in, **backbone
PDB(s)** out. Always a job — diffusion is slow. Reuses the archetype-B job
machinery from [`chai1`](../chai1/README.md).

> The user wrote "RFDiff4". Confirm whether this is RFdiffusion (original),
> RFdiffusion All-Atom, or RFdiffusion2 ("RFDiff2"); the entry point and config
> differ. The service shape below is the same regardless.

## Backend API being wrapped

RFdiffusion `run_inference.py` (Hydra config). Key inputs:
`inference.input_pdb`, `contigmap.contigs` (the contig string),
`ppi.hotspot_res` (target hotspots), `inference.num_designs`,
`diffuser.T` (timesteps), `denoiser.noise_scale`. Output: backbone PDB(s) (no
sequence — hand to `proteinmpnn` next).

## Intended HTTP contract

```jsonc
POST /v1/generate/jobs    -> {job_id}        (always async)
GET  /v1/generate/jobs/{id}
{
  "input_pdb": "<PDB text or path>",       // target
  "contigs": "A1-150/0 50-80",             // e.g. 50-80 res binder vs chain A
  "hotspot_res": ["A45", "A48", "A52"],
  "num_designs": 20,
  "diffuser_T": 50,
  "noise_scale": 1.0,
  "seed": 0
}
```

`GenerateResponse.data[]`: `{index, pdb, meta}` (contig actually realized, etc.).

## What copies vs. what is new

| Box | Action | Template anchor |
| --- | --- | --- |
| `Settings` | copy; `RFD_` prefix; model dir, device, output scratch | server.py:57 |
| `GenerateRequest` | **new**; `extra='forbid'`; validate contig syntax, hotspot ids vs. PDB | server.py:86 |
| `GenerateResponse` | **new**; backbone PDB(s) | server.py:162 |
| `GenerateBackend(Protocol)` | copy; `load`/`unload`/`generate` | server.py:171 |
| `load`/`unload` | copy; load RFdiffusion weights, one resident | server.py:208 / :237 |
| `generate` | **new**: compose Hydra config, run `run_inference.py`, collect PDBs | replaces server.py:246 |
| job API + worker | **required** (runs are minutes-to-long) | see `_pattern/PATTERN.md` |
| `create_app` / lock / `main` / envelope | copy | server.py:305 / :376 |

## Open decisions

- In-process call vs. subprocess (RFdiffusion's Hydra entry point is
  subprocess-friendly; isolates its heavy deps from the FastAPI process).
- Contig-string validation: how strict server-side vs. let RFdiffusion reject.
- Natural downstream chain: `rfdiffusion` → `proteinmpnn` → `chai1`/`esmfold2`.
  Consider a thin orchestration note, but keep each service single-purpose.
