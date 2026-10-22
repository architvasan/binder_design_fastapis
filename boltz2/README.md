# boltz2-service (design draft)

Archetype **B** — async fold job. Clone the [`chai1`](../chai1/README.md)
scaffold; swap the backend. Boltz-2 adds **binding-affinity prediction** on top
of co-folding, so the response carries an affinity field Chai-1 does not.

## Backend API being wrapped

Boltz-2 (`boltz` package). Input is a YAML (or FASTA) spec listing sequences,
optional ligands (SMILES/CCD), MSA, and an `affinity` block naming the binder.
Driven by CLI (`boltz predict input.yaml --out_dir ...`) or the Python API.

- **Output:** predicted CIF structure(s) + confidence (pTM/ipTM/PAE) **and**
  predicted binding affinity / `affinity_probability` for the designated ligand
  or binder chain.

## Intended HTTP contract

Same job API as chai1 (`POST /v1/fold/jobs`, `GET /v1/fold/jobs/{id}`,
`POST /v1/fold`, `/health`). `FoldRequest` adds an optional `affinity` target
(which chain/ligand to score); `FoldResponse.data[].scores` adds `affinity` and
`affinity_probability`.

## Diffs from chai1

| Box | Action |
| --- | --- |
| request | add `affinity` block (binder chain id); accept YAML-equivalent fields |
| backend `fold` | write Boltz YAML spec instead of Chai FASTA; call `boltz predict`; read CIF + confidence JSON + affinity |
| response scores | add `affinity`, `affinity_probability` |
| everything else | copy from chai1 (job store, lock, lifecycle, envelope) |

## Open decisions

- CLI subprocess vs. in-process Python API for `fold`.
- Shared MSA handling with chai1 (same MSA dir convention).
- Whether affinity is always computed or opt-in per request.
