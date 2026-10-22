# proteinmpnn-service (design draft)

Archetype **C** — async design job (sequence output). Inverse folding: a fixed
backbone in, designed **sequences** out. Reuses the archetype-B job machinery
from [`chai1`](../chai1/README.md); different contract.

## Backend API being wrapped

ProteinMPNN (`$PROTEINMPNN_PATH` checkout; `protein_mpnn_run.py` or the
in-process API used by our `inverse-folding` skill). Input is a PDB (e.g. a
Chai-1 complex), with optional fixed/tied positions and a sampling temperature;
output is N sequences with per-sequence score (negative log-likelihood) and
global/per-residue confidence.

## Intended HTTP contract

```jsonc
POST /v1/design           (sync for a single backbone; job API for batches)
{
  "pdb": "<PDB text or path>",
  "chains_to_design": ["B"],          // e.g. redesign the binder chain
  "fixed_positions": {"A": [1,2,3]},  // keep target fixed
  "num_sequences": 8,
  "sampling_temp": 0.1
}
```

`DesignResponse.data[]`: `{index, sequence, score, global_score, seq_recovery}`.

## What copies vs. what is new

| Box | Action | Template anchor |
| --- | --- | --- |
| `Settings` | copy; `MPNN_` prefix; `$PROTEINMPNN_PATH`, weights, device | server.py:57 |
| `DesignRequest` | **new**; `extra='forbid'`; validate chain ids, fixed-pos ranges vs. PDB | server.py:86 |
| `DesignResponse` | **new**; sequences + scores | server.py:162 |
| `DesignBackend(Protocol)` | copy; methods `load`/`unload`/`design` | server.py:171 |
| `load`/`unload` | copy; load MPNN `.pt` weights, one resident | server.py:208 / :237 |
| `design` | **new**: write/parse PDB, run MPNN, collect sequences+scores | replaces server.py:246 |
| `create_app` / lock / `main` | copy | server.py:305 / :376 |
| job API | optional (output is small; sync is fine for single design) | see `_pattern/PATTERN.md` |

## Open decisions

- Accept PDB as inline text vs. a path to shared storage (ties into the
  repo-wide transport decision).
- Expose tied positions / symmetry and per-residue bias (MPNN supports both).
- Whether to fold-back-validate here or leave that to `chai1`/`esmfold2`.
