# esmc-service (design draft)

Archetype **A** — sync vector service. The closest to the template: ESM-C is the
same ESM-C family `genslm-esm-service` already serves, so this is nearly a copy
with the GenSLM-specific tokenization swapped for stock ESM-C.

## Backend API being wrapped

EvolutionaryScale ESM-C (`esm` package), or the HF checkpoints:

```python
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
model = ESMC.from_pretrained("esmc_300m").to(device)        # 300m / 600m / 6b
p = model.encode(ESMProtein(sequence="MKT..."))
out = model.logits(p, LogitsConfig(return_embeddings=True))  # out.embeddings
```

Mean-pool `out.embeddings` over real tokens (exclude BOS/EOS/pad), exactly as
the template's `mean_pool` does. A requested transformer layer maps to hidden
states if exposed; otherwise serve the final representation.

## Intended HTTP contract

Keep the template's OpenAI-compatible contract **unchanged**:

```
POST /v1/embeddings   -> {object:'list', data:[{embedding, index}], model, usage}
GET  /health
```

`EmbeddingsRequest`: `input: list[str]`, `model`, optional `transformer_layer`.
Drop `modality` (ESM-C is amino-acid only) unless you also serve codon variants.

## What copies vs. what is new

| Box | Action | Template anchor |
| --- | --- | --- |
| allowlist | redefine `MODEL_LAYER_COUNTS` for `esmc_300m/600m/6b` layer counts | server.py:41 |
| `Settings` | copy; env prefix `ESMC_` | server.py:57 |
| `EmbeddingsRequest/Response` | copy; drop `modality` | server.py:86 / :162 |
| `EmbeddingBackend(Protocol)` | copy verbatim | server.py:171 |
| `load`/`unload` | copy; `ESMC.from_pretrained` instead of `AutoModel` | server.py:208 / :237 |
| `embed` | copy; swap tokenization/encode; keep `mean_pool` | server.py:246 |
| `create_app` / lock / routes / `main` | copy ~verbatim | server.py:305 / :337 / :376 |
| packaging envelope | copy; swap dep → `esm`, scripts → `esmc-serve` | pyproject.toml |

## Open decisions

- Whether to expose per-layer embeddings (needs hidden-state access) or only the
  final representation.
- License/weights access for ESM-C 6B (gated) and GPU memory (6B needs bf16 + a
  big card; keep the one-model-resident policy).

**Lowest-risk service to build first** — it validates that the pattern transfers
before tackling the archetype-B/C structure services.
