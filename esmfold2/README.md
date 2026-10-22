# esmfold2-service (design draft)

Archetype **B** — async fold job (the simplest one). Single-sequence structure
prediction: no MSA, no ligands, one chain in, one structure out. Clone the
[`chai1`](../chai1/README.md) scaffold and strip it down.

> **Confirm the exact model.** "ESMFold2" is ambiguous — most likely ESMFold
> from the `esm` repo (`esm.pretrained.esmfold_v1()`), ESM-2 based. If you mean
> an ESM3-derived folding path, the backend call changes but the service shape
> does not. Pin the exact package/weights before coding.

## Backend API being wrapped

```python
import esm
model = esm.pretrained.esmfold_v1().eval().to(device)
pdb_str = model.infer_pdb(sequence)        # -> PDB text; mean pLDDT in B-factors
```

Fast relative to MSA-based folders (seconds), single sequence, no MSA server.

## Intended HTTP contract

Job API like chai1, but a **much simpler request**:

```jsonc
POST /v1/fold  (blocking is often fine here — runs are seconds)
{ "sequences": ["MKT..."], "num_recycles": 4 }
```

`FoldResponse.data[]`: `{index, pdb, scores: {mean_plddt, ptm}}`.

## Diffs from chai1

| Box | Action |
| --- | --- |
| request | drop chains/ligands/MSA/constraints → just `sequences` + `num_recycles` |
| backend `fold` | `model.infer_pdb(seq)`; parse mean pLDDT (and pTM if exposed) |
| response | PDB text instead of CIF; `mean_plddt` headline score |
| concurrency | sync `POST /v1/fold` may suffice; keep job API optional for batches |
| everything else | copy from chai1 |

## Open decisions

- Keep it sync (seconds/seq) or reuse the job API for large batches.
- Batch multiple sequences per request (ESMFold batches on one GPU within memory).
