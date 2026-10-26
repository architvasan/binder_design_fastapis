# chai1-service

A FastAPI wrapper around **Chai-1** (`chai_lab.chai1.run_inference`) that co-folds
a complex of protein / DNA / RNA / ligand chains and returns the predicted
structures as mmCIF plus Chai-1's ranking scores.

Same shape as the other services in this repo: typed `Settings`, an injectable
backend `Protocol` (so the API is tested without a GPU or weights), inference
serialized per process. Each request folds one complex and returns the
structures **synchronously, inline in the JSON response**, ranked best-first.

**Status: implemented and tested** (`src/chai1_service/server.py` + `tests/`).
ruff, mypy, and the pytest suite pass (API validation, chain-name assignment,
FASTA rendering, and the candidate-ranking/score mapping against a stubbed
`chai_lab`). The real `run_inference` call is exercised through an injected fake
module; a live GPU run still needs `chai_lab` weights on a healthy accelerator.

## Install and run

```bash
cd chai1
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"      # pulls chai_lab + torch (large)
chai-serve                       # http://localhost:8000
```

Chai-1 downloads its weights on the first fold. Settings come from `CHAI_*`
environment variables (see `.env.example`): `CHAI_DEVICE`, `CHAI_SCRATCH_DIR`,
`CHAI_USE_ESM_EMBEDDINGS`, `CHAI_USE_MSA_SERVER`, `CHAI_HOST`, `CHAI_PORT`.

## Fold a complex

Each entry in `chains` is one component. Protein/DNA/RNA chains carry a
`sequence`; a `ligand` carries a `smiles` string. Give an explicit `name` or let
the service assign `A`, `B`, `C`, … in order. At least one protein/DNA/RNA chain
is required (a ligand-only request is rejected).

```bash
# protein + protein + ligand holo-complex
curl http://localhost:8000/v1/fold -H 'Content-Type: application/json' -d '{
  "chains": [
    {"entity_type": "protein", "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ", "name": "target"},
    {"entity_type": "protein", "sequence": "GSHMKLVINGKTLKGEITVELDTGELLHKFTNG", "name": "binder"},
    {"entity_type": "ligand",  "smiles": "CC(=O)Oc1ccccc1C(=O)O", "name": "lig"}
  ],
  "num_diffn_samples": 5,
  "num_trunk_recycles": 3,
  "num_diffn_timesteps": 200,
  "seed": 42
}'
```

Response — `results` holds one entry per diffusion sample, **ranked best-first**
by Chai-1's `sorted()` (descending `aggregate_score`). Each sample's `structure`
is mmCIF text:

```json
{
  "object": "fold",
  "model": "chai-1",
  "num_chains": 3,
  "chain_names": ["target", "binder", "lig"],
  "num_samples": 5,
  "results": [
    {
      "format": "mmcif",
      "structure": "data_...",
      "scores": {
        "aggregate_score": 0.83,
        "complex_ptm": 0.81,
        "interface_ptm": 0.62,
        "has_inter_chain_clashes": false
      },
      "pae": null
    }
  ],
  "usage": {"num_chains": 3, "total_residues": 66}
}
```

`aggregate_score` is Chai-1's own ranking metric
(`0.2*complex_ptm + 0.8*interface_ptm - 100*has_inter_chain_clashes`); samples
are returned in that order.

### Request fields

| Field | Default | Notes |
| --- | --- | --- |
| `chains` | — | list of components; ≥1, and ≥1 must be protein/dna/rna |
| chain `entity_type` | `protein` | `protein`/`dna`/`rna` (need `sequence`) or `ligand` (needs `smiles`) |
| `use_esm_embeddings` | service default | override per request |
| `use_msa_server` | service default | ColabFold MSA server (needs network) |
| `num_trunk_recycles` | 3 | 1–10 |
| `num_diffn_timesteps` | 200 | 1–500 |
| `num_diffn_samples` | 5 | 1–5 |
| `seed` | null | RNG seed |
| `return_pae` | false | include the (large) PAE matrix |

## HTTP contract

```
POST /v1/fold   -> {object:'fold', model, chain_names, num_samples, results:[{structure, scores, pae?}], usage}
GET  /health
```

## Design notes

- **Co-folding is native** to Chai-1: each chain becomes a FASTA record whose
  header carries the entity type (`>protein|name=A`, `>ligand|name=L` with the
  SMILES as the body). The backend writes the FASTA to a scratch dir, calls
  `run_inference`, and reads back the ranked CIFs + `SampleRanking` scores.
- **Transport:** mmCIF is returned inline, like the ESMFold2 service. For large
  multi-sample outputs a shared-storage path option (the archetype-B pattern)
  would be the next step.
- **Scores** come from `SampleRanking`: `aggregate_score`, `ptm_scores`
  (`complex_ptm`, `interface_ptm`), and `clash_scores.has_inter_chain_clashes`.

## Deferred (documented follow-ups)

Sketched in the original design draft but intentionally **not** implemented here,
to keep the first cut small and testable (mirrors how the ESMFold2 service
deferred its async API):

- **Async job API** (`POST /v1/fold/jobs` + `GET /v1/fold/jobs/{id}`). Runs are
  minutes; a job store would decouple the HTTP request from the run. The sync
  endpoint is the single-fold path for now.
- **Residue constraints** (the Chai-1 constraint CSV). Modeled in the draft but
  the exact CSV schema is unverified, so it is left out rather than guessed.
- **Precomputed MSA directory** (`msa_directory`). Only the `use_msa_server`
  toggle is exposed today.

## What copies from the template vs. what is new

| Box | Action |
| --- | --- |
| `Settings` | copy; env prefix `CHAI_`; new fields (device, scratch_dir, msa/esm flags) |
| allowlist | → `EntityType` + polymer alphabets (no model-name allowlist; Chai-1 is one model) |
| request/response models | new: `FoldRequest`/`ChainInput`/`FoldResponse`/`FoldScores` |
| `FoldBackend(Protocol)` | copy; methods `load`/`unload`/`fold` |
| `ChaiFoldBackend.fold` | new: build FASTA → `run_inference` → read CIF + scores → map |
| `create_app` / lifespan / lock / routes / `main` | copy; route `/v1/fold` |
| packaging envelope | copy; dep → `chai_lab`, scripts → `chai-serve` |
| test strategy | copy: `FakeBackend` + injected fake `chai_lab` module |
