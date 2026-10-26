# boltz2-service

A FastAPI wrapper around **Boltz-2** that co-folds a complex of protein / DNA /
RNA / ligand chains and returns the predicted structures as mmCIF plus Boltz-2's
confidence scores, and optionally a small-molecule **binding-affinity**
prediction.

Same shape as the other services in this repo: typed `Settings`, an injectable
backend `Protocol` (so the API is tested without a GPU or weights), inference
serialized per process. Each request folds one complex and returns the
structures **synchronously, inline in the JSON response**, ranked best-first.

Boltz-2 ships as the `boltz` command-line tool, so the backend writes a YAML
input, invokes `boltz predict` as a subprocess, and reads back the CIF files and
confidence JSON it produces.

**Status: implemented and tested** (`src/boltz2_service/server.py` + `tests/`).
ruff, mypy, and the pytest suite pass (API validation, id assignment, YAML
rendering, command construction, and output parsing/ranking against a faked
subprocess). A live run needs the real `boltz` executable + weights on a GPU.

## Install and run

```bash
cd boltz2
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"      # pulls boltz + torch (large)
boltz-serve                      # http://localhost:8000
```

Boltz-2 downloads its weights on the first run (cached under `~/.boltz`, or
`BOLTZ_CACHE_DIR`). Settings come from `BOLTZ_*` environment variables (see
`.env.example`): `BOLTZ_EXECUTABLE`, `BOLTZ_CACHE_DIR`, `BOLTZ_SCRATCH_DIR`,
`BOLTZ_DEVICES`, `BOLTZ_USE_MSA_SERVER`, `BOLTZ_HOST`, `BOLTZ_PORT`.

## Fold a complex

Each entry in `chains` is one component. Protein/DNA/RNA chains carry a
`sequence`; a `ligand` carries either `smiles` or `ccd`. Give an explicit `id`
or let the service assign `A`, `B`, `C`, … in order. At least one
protein/DNA/RNA chain is required.

```bash
# protein + ligand, with a binding-affinity prediction for the ligand
curl http://localhost:8000/v1/fold -H 'Content-Type: application/json' -d '{
  "chains": [
    {"id": "A", "entity_type": "protein", "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"},
    {"id": "L", "entity_type": "ligand",  "smiles": "CC(=O)Oc1ccccc1C(=O)O"}
  ],
  "affinity_binder": "L",
  "diffusion_samples": 1,
  "recycling_steps": 3
}'
```

Response — `results` holds one entry per diffusion sample, **ranked best-first**
by Boltz-2's `confidence_score`. Each sample's `structure` is mmCIF text.
`affinity` is present only when `affinity_binder` is set.

```json
{
  "object": "fold",
  "model": "boltz-2",
  "num_chains": 2,
  "chain_ids": ["A", "L"],
  "num_samples": 1,
  "results": [
    {
      "format": "mmcif",
      "structure": "data_...",
      "scores": {"confidence_score": 0.84, "ptm": 0.81, "iptm": 0.62, "complex_plddt": 0.88}
    }
  ],
  "affinity": {"affinity_pred_value": 1.2, "...": "..."},
  "usage": {"num_chains": 2, "total_residues": 33}
}
```

`confidence_score` is Boltz-2's own ranking metric
(`0.8*complex_plddt + 0.2*iptm`); samples are returned in that order. The
`affinity` object is Boltz-2's `affinity_*.json` passed through verbatim.

### Request fields

| Field | Default | Notes |
| --- | --- | --- |
| `chains` | — | list of components; ≥1, and ≥1 must be protein/dna/rna |
| chain `entity_type` | `protein` | `protein`/`dna`/`rna` (need `sequence`) or `ligand` (needs `smiles` or `ccd`) |
| `use_msa_server` | service default | mmseqs2 MSA server (needs network) |
| `diffusion_samples` | 1 | 1–25 |
| `recycling_steps` | 3 | 1–10 |
| `affinity_binder` | null | chain id of a ligand to score binding affinity for |

## HTTP contract

```
POST /v1/fold   -> {object:'fold', model, chain_ids, num_samples, results:[{structure, scores}], affinity?, usage}
GET  /health
```

## Design notes

- **Co-folding is native** to Boltz-2: each chain becomes a `sequences` entry in
  the YAML (`protein`/`dna`/`rna` with `sequence`; `ligand` with `smiles` or
  `ccd`). An optional `properties: [{affinity: {binder: <id>}}]` block requests a
  ligand affinity prediction.
- **Subprocess seam:** the single `subprocess.run` call lives in
  `BoltzFoldBackend._run`, so the backend is tested without the real executable
  by faking that one method.
- **Transport:** mmCIF is returned inline, like the other fold services. PAE/PDE
  `.npz` files Boltz-2 writes are not returned (binary; add a path option if
  needed).

## Deferred (documented follow-ups)

- **Async job API** (`POST /v1/fold/jobs` + `GET /v1/fold/jobs/{id}`). Runs are
  minutes; a job store would decouple the HTTP request from the run. The sync
  endpoint is the single-fold path for now (mirrors the ESMFold2/Chai-1 cut).
- **Templates, modifications, cyclic polymers, MSA directories** — Boltz-2's YAML
  supports them; only chains, ligands, and affinity are exposed today.

## What copies from the template vs. what is new

| Box | Action |
| --- | --- |
| `Settings` | copy; env prefix `BOLTZ_`; new fields (executable, cache/scratch dirs, devices, msa) |
| allowlist | → `EntityType` + polymer alphabets (Boltz-2 is one model) |
| request/response models | new: `FoldRequest`/`ChainInput`/`FoldResponse`/`FoldScores` |
| `FoldBackend(Protocol)` | copy; methods `load`/`unload`/`fold` |
| `BoltzFoldBackend.fold` | new: build YAML → `boltz predict` subprocess → read CIF + confidence JSON |
| `create_app` / lifespan / lock / routes / `main` | copy; route `/v1/fold` |
| packaging envelope | copy; dep → `boltz`, scripts → `boltz-serve` |
| test strategy | copy: `FakeBackend` + faked subprocess seam |
