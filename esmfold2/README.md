# esmfold2-service

A FastAPI wrapper around **ESMFold2** (EvolutionaryScale `esm` SDK) that folds
one or more protein chains into a single complex and returns the structure as
mmCIF plus confidence scores.

Same shape as the ESM-C service in this repo: typed `Settings`, an injectable
backend `Protocol` (so the API is tested without a GPU or weights), one model
resident at a time, inference serialized per process. It runs in
**single-sequence mode** (no MSA) and returns the structure **synchronously,
inline in the JSON response** — one request folds one complex.

**Status: implemented and tested.** `src/esmfold2_service/server.py` + `tests/`.
ruff, mypy, and the pytest suite pass (API validation, chain-id assignment, and
the result-mapping math); the real server boots and serves `/health`,
`/v1/fold`, and `/docs`.

## Install and run

```bash
cd esmfold2
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"      # pulls esm + torch (large)
esmfold2-serve                   # http://localhost:8000
```

Preload a model at startup instead of on the first request:

```bash
ESMFOLD2_PRELOAD_MODEL=esmfold2 esmfold2-serve
```

## Fold a complex

Each entry in `sequences` is one chain. Give an explicit `id` or let the service
assign `A`, `B`, `C`, … in order. ipTM (the interface score) is returned only
when there is more than one chain.

```bash
curl http://localhost:8000/v1/fold -H 'Content-Type: application/json' -d '{
  "model": "esmfold2",
  "sequences": [
    {"id": "A", "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"},
    {"id": "B", "sequence": "GSHMKLVINGKTLKGEITVE"}
  ],
  "num_loops": 20,
  "num_sampling_steps": 100,
  "seed": 0
}'
```

### Chain types: protein, DNA, RNA, ligand

Each chain carries a `type` (default `protein`). Protein/DNA/RNA chains carry a
`sequence`; a `ligand` carries **exactly one** of `smiles` or `ccd` (a list of
CCD component codes) and **no** `sequence`. At least one protein/DNA/RNA chain is
required — a ligand-only request is rejected. Sequence alphabets are validated
per type (protein = amino acids; DNA = `ACGTN`; RNA = `ACGUN`), and SMILES is
kept case-sensitive.

```bash
# protein + nucleic acid (DNA here; use "rna" with a U-containing sequence for RNA)
curl http://localhost:8000/v1/fold -H 'Content-Type: application/json' -d '{
  "model": "esmfold2",
  "sequences": [
    {"type": "protein", "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"},
    {"type": "dna", "sequence": "ACGTACGTACGT"}
  ]
}'

# protein + ligand (SMILES; or pass "ccd": ["HEM"] instead of "smiles")
curl http://localhost:8000/v1/fold -H 'Content-Type: application/json' -d '{
  "model": "esmfold2",
  "sequences": [
    {"type": "protein", "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"},
    {"type": "ligand", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}
  ]
}'
```

`usage.total_residues` counts polymer residues only; ligands contribute no
residues.

Response — `results` holds one entry per diffusion sample, **ranked best
structure first** (by `ranking_score`). Each sample's `structure` is mmCIF text
and `plddt` is per-residue:

```json
{
  "object": "fold",
  "model": "esmfold2",
  "num_chains": 2,
  "chain_ids": ["A", "B"],
  "num_samples": 1,
  "results": [
    {
      "format": "mmcif",
      "structure": "data_pred\n...",
      "plddt_mean": 84.2,
      "plddt": [/* per residue */],
      "ptm": 0.81,
      "iptm": 0.62,
      "pae": null,
      "ranking_score": 0.648
    }
  ],
  "usage": {"num_chains": 2, "total_residues": 53}
}
```

`ranking_score` is the quality metric the samples are sorted by (higher is
better): `0.8*iptm + 0.2*ptm` for complexes, else `ptm`. With
`num_diffusion_samples > 1`, `results` lists every sample in that order so the
caller can pick an alternative.

### Request fields

| Field | Default | Notes |
| --- | --- | --- |
| `model` | — | `esmfold2` or `esmfold2-fast` |
| `sequences` | — | list of chains; ≥1, and ≥1 must be protein/dna/rna |
| chain `type` | `protein` | `protein`/`dna`/`rna` (need `sequence`) or `ligand` (needs `smiles` or `ccd`) |
| `num_loops` | 20 | recycling loops (1–100) |
| `num_sampling_steps` | 100 | diffusion steps (1–500) |
| `num_diffusion_samples` | 1 | samples per fold (1–10) |
| `seed` | 0 | RNG seed |
| `return_pae` | false | include the (large) PAE matrix |

Supported models: `esmfold2` (`biohub/ESMFold2`) and `esmfold2-fast`
(`biohub/esmfold2-fast`, single-sequence only). Add more to `MODEL_REPOS` in
`server.py` once their weights are available.

## HTTP contract

```
POST /v1/fold   -> {object:'fold', chain_ids, num_samples, results:[{structure, plddt, ptm, iptm, pae?, ranking_score}], usage}
GET  /health
```

## Design notes

- **Multichain is native** to ESMFold2: each chain is a separate input in a
  single `StructurePredictionInput`. The service maps each `ChainInput` to the
  SDK input class its `type` selects — `ProteinInput`, `DNAInput`, `RNAInput`,
  or `LigandInput` (SMILES or CCD) — so protein, nucleic-acid, and small-molecule
  components fold together in one complex.
- **Transport:** mmCIF is returned inline. For single-sequence folds this is
  small. If very long multi-chain inputs become common, add a shared-storage
  path option (the archetype-B pattern) rather than inflating the JSON.
- **ipTM** is the reason multichain folding is worth serving — it scores the
  predicted interface, not just per-chain confidence.
- **PAE** is opt-in (`return_pae`) because the matrix is O(L²); its exact
  attribute on the result object is treated as best-effort.

## What copies from the ESM-C service vs. what is new

| Box | Action |
| --- | --- |
| `Settings` | copy; env prefix `ESMFOLD2_` |
| `EmbeddingBackend(Protocol)` | → `FoldBackend`; `embed` → `fold` |
| `load`/`unload` | copy; `EsmFold2Model.from_pretrained(repo, device=...)` |
| request/response models | new: `FoldRequest`/`ChainInput`/`FoldResponse` |
| `create_app` / lock / routes / `main` | copy ~verbatim; route `/v1/fold` |
| packaging envelope | copy; scripts → `esmfold2-serve` |
| test strategy | copy: `FakeBackend` + injected fake `esm` module |
