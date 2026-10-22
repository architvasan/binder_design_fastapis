# chai1-service (design draft)

Archetype **B** — async fold job. Worked example for Chai-1 / Boltz-2 / ESMFold2.

Wraps `chai_lab.chai1.run_inference` and exposes co-folding over HTTP. Folds a
protein complex (optionally with ligands, metals, glycans, MSA, and residue
constraints) and returns ranked structures with interface-confidence scores.

## Backend API being wrapped

`chai_lab.chai1.run_inference` (chai-lab repo, `chai_lab/chai1.py:450`):

```python
run_inference(
    fasta_file: Path, *, output_dir: Path,
    use_esm_embeddings=True, use_msa_server=False, msa_server_url=..., msa_directory=None,
    constraint_path: Path | None = None,
    num_trunk_recycles=3, num_diffn_timesteps=200, num_diffn_samples=5,
    seed: int | None = None, device: str | None = None, low_memory=True,
) -> StructureCandidates   # .cif_paths, .ranking_data[SampleRanking], .pae, .sorted()
```

- **Input** is a FASTA whose headers carry entity type: `>protein|name=A`,
  `>ligand|name=L` (sequence = SMILES), `>rna|...`, `>dna|...`. Constraints are a
  separate CSV. MSA is a directory (or `use_msa_server=True`).
- **Output** is `num_diffn_samples` CIF files + a `SampleRanking` each
  (`aggregate_score`, pTM/ipTM via `ptm_scores`, `clash_scores`). `.sorted()`
  ranks by `aggregate_score`.

## Intended HTTP contract

```
POST /v1/fold/jobs        -> {job_id, status}          (async; runs are minutes)
GET  /v1/fold/jobs/{id}   -> {status, result?}
POST /v1/fold             -> blocking single fold       (for quick calls / tests)
GET  /health
```

### FoldRequest (new)
```jsonc
{
  "chains": [
    {"entity_type": "protein", "sequence": "MKT...", "name": "target"},
    {"entity_type": "protein", "sequence": "GSHM...", "name": "binder"},
    {"entity_type": "ligand",  "smiles": "CC(=O)O", "name": "lig"}      // optional
  ],
  "constraints": [                                                       // optional
    {"chain_a": "target", "res_a": 45, "chain_b": "binder", "res_b": 3, "max_dist": 8.0}
  ],
  "use_msa_server": false,          // or supply precomputed MSA
  "use_esm_embeddings": true,
  "num_diffn_samples": 5,
  "num_trunk_recycles": 3,
  "num_diffn_timesteps": 200,
  "seed": 42
}
```

### FoldResponse (new)
```jsonc
{
  "object": "fold",
  "model": "chai-1",
  "data": [                         // sorted best-first
    {"index": 0, "cif": "<CIF text or artifact path>",
     "scores": {"aggregate_score": .., "ptm": .., "iptm": .., "clash": ..}}
  ],
  "meta": {"num_samples": 5, "seed": 42, "runtime_s": ..}
}
```

## What copies from the template vs. what is new

| Box | Action | Template anchor |
| --- | --- | --- |
| `ChaiSettings` | copy shape, new fields (device, scratch_dir, msa/esm flags, sampling defaults) | server.py:57 |
| allowlist | replace `MODEL_LAYER_COUNTS` with `ENTITY_TYPES` + alphabets | server.py:41 |
| `FoldRequest` / validators | **new**; keep `extra='forbid'`; per-entity alphabet, SMILES, constraint-range | server.py:86 / :104 / :122 |
| `FoldResponse` / `Structure` | **new**; scores from `SampleRanking` | server.py:162 |
| `FoldBackend(Protocol)` | copy; methods `load`/`unload`/`fold` | server.py:171 |
| `ChaiFoldBackend.load`/`unload` | copy one-resident + free | server.py:208 / :237 |
| `ChaiFoldBackend.fold` | **new**: build FASTA → `run_inference` → read CIF+scores → sort | replaces server.py:246 (embed) |
| `create_app` / lifespan / lock | copy | server.py:305 / :314 / :312 |
| `/health` / `main` | copy ~verbatim | server.py:332 / :376 |
| async job store + worker | **new** (see `_pattern/PATTERN.md`) | — |
| packaging envelope | copy; swap dep → `chai_lab`, scripts → `chai-serve` | pyproject.toml |

## Open decisions

- **CIF transport:** base64 in JSON vs. write to shared storage + return a path.
  Decide repo-wide (affects all archetype-B/C services).
- **Holo folding:** ligand/metal/glycan chains come in via `entity_type`; mirror
  the `cofactor-ligand-setup` conventions so metalloenzyme targets fold with a
  physiological pocket.
- **MSA:** allow `use_msa_server` (ColabFold) vs. require precomputed MSA dir
  for offline/HPC runs.
- **Weights/device:** Chai downloads weights on first use; preload in `lifespan`
  and pin to the process's GPU.
