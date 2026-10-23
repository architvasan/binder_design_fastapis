"""FastAPI service for ESM-C sequence embeddings.

Serves mean-pooled ESM-C representations over an OpenAI-compatible
``/v1/embeddings`` endpoint. Follows the ``genslm-esm-service`` pattern: typed
settings, an injectable backend ``Protocol`` (so the API is testable without a
GPU or model weights), one model resident at a time, and inference serialized
per process.

ESM-C is loaded through the EvolutionaryScale ``esm`` SDK
(``ESMC.from_pretrained``). The import is deferred to :meth:`load` so this module
imports without ``esm`` installed, which keeps the test suite GPU- and
weight-free via the fake backend.
"""

from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from typing import cast
from typing import Literal
from typing import Protocol

import torch
import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from esmc_service import __version__

LOGGER = logging.getLogger(__name__)

# The allowlist of locally loadable ESM-C checkpoints mapped to their
# transformer-layer count. This is both the set of legal models and the
# per-model metadata the layer-bounds validation reads. All three checkpoints
# are public on the HuggingFace Hub (``biohub/ESMC-*``) and load locally; the
# 6B weights are large (~12 GB in bf16) and download on first use.
MODEL_LAYER_COUNTS = {
    'esmc_300m': 30,
    'esmc_600m': 36,
    'esmc_6b': 80,
}


class Settings(BaseSettings):
    """Configuration loaded from ``ESMC_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix='ESMC_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    hf_token: SecretStr | None = None
    preload_model: str | None = None
    device: str | None = None
    host: str = '0.0.0.0'
    port: int = Field(default=8000, ge=1, le=65535)

    @field_validator('preload_model')
    @classmethod
    def validate_preload_model(cls, value: str | None) -> str | None:
        """Only allow a supported ESM-C checkpoint to be preloaded."""
        if value is not None and value not in MODEL_LAYER_COUNTS:
            raise ValueError(
                f'preload_model must be one of {sorted(MODEL_LAYER_COUNTS)}',
            )
        return value


class EmbeddingsRequest(BaseModel):
    """Request for ESM-C sequence embeddings."""

    model_config = ConfigDict(extra='forbid')

    input: list[str]
    model: str
    stream: Literal[False] = Field(
        default=False,
        description='Non-streaming marker accepted for OpenAI-client compatibility.',
    )
    transformer_layer: int | None = Field(
        default=None,
        ge=1,
        description='1-based transformer layer; defaults to the final layer.',
    )

    @field_validator('input')
    @classmethod
    def validate_input(cls, value: list[str]) -> list[str]:
        """Reject empty batches and empty biological sequences."""
        if not value:
            raise ValueError('input must contain at least one sequence')
        if not all(sequence.strip() for sequence in value):
            raise ValueError('every input sequence must be a non-empty string')
        return value

    @field_validator('model')
    @classmethod
    def validate_model(cls, value: str) -> str:
        """Only allow supported ESM-C checkpoints."""
        if value not in MODEL_LAYER_COUNTS:
            raise ValueError(f'model must be one of {sorted(MODEL_LAYER_COUNTS)}')
        return value

    @model_validator(mode='after')
    def validate_transformer_layer(self) -> EmbeddingsRequest:
        """Reject transformer layers outside the selected model."""
        n_layers = MODEL_LAYER_COUNTS[self.model]
        if self.transformer_layer is not None and self.transformer_layer > n_layers:
            raise ValueError(
                f'{self.model} has {n_layers} transformer layers; '
                f'transformer_layer must be between 1 and {n_layers}',
            )
        return self

    def resolved_transformer_layer(self) -> int:
        """Return the requested layer or the model's final layer."""
        return self.transformer_layer or MODEL_LAYER_COUNTS[self.model]


class EmbeddingObject(BaseModel):
    """One sequence embedding."""

    object: Literal['embedding'] = 'embedding'
    embedding: list[float]
    index: int


class Usage(BaseModel):
    """Biological token counts for an embeddings request."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingsResponse(BaseModel):
    """Response containing sequence embeddings and token usage."""

    object: Literal['list'] = 'list'
    data: list[EmbeddingObject]
    model: str
    usage: Usage


class EmbeddingBackend(Protocol):
    """Interface used by the HTTP layer and test backends."""

    def load(self, model_id: str) -> object:
        """Load or return the requested model."""

    def unload(self) -> None:
        """Release the resident model."""

    def embed(
        self,
        model: str,
        sequences: list[str],
        transformer_layer: int,
    ) -> list[list[float]]:
        """Return one embedding per input sequence."""
        ...


@dataclass
class LoadedModel:
    """The single model currently resident in memory."""

    model_id: str
    model: torch.nn.Module
    device: torch.device


class EsmcEmbeddingBackend:
    """Load one ESM-C model at a time and perform per-sequence inference."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loaded: LoadedModel | None = None

    def _resolve_device(self) -> torch.device:
        """Pick the configured device, else CUDA when present, else CPU."""
        if self.settings.device is not None:
            return torch.device(self.settings.device)
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load(self, model_id: str) -> LoadedModel:
        """Load or return the requested model."""
        if self.loaded is not None and self.loaded.model_id == model_id:
            return self.loaded

        self.unload()
        LOGGER.info('Loading model %s', model_id)
        if self.settings.hf_token is not None:
            # Imported lazily so the module loads without the optional
            # dependency, keeping the test suite GPU- and weight-free.
            from huggingface_hub import login  # noqa: PLC0415

            login(
                token=self.settings.hf_token.get_secret_value(),
                add_to_git_credential=False,
            )

        from esm.models.esmc import ESMC  # noqa: PLC0415

        device = self._resolve_device()
        model = ESMC.from_pretrained(model_id)
        model = model.to(device).eval()
        if device.type == 'cpu':
            model = model.float()
        else:
            model = model.to(torch.bfloat16)

        self.loaded = LoadedModel(model_id, model, device)
        LOGGER.info('Loaded model %s on %s', model_id, device)
        return self.loaded

    def unload(self) -> None:
        """Release the resident model and unused accelerator allocations."""
        if self.loaded is None:
            return
        self.loaded = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def embed(
        self,
        model: str,
        sequences: list[str],
        transformer_layer: int,
    ) -> list[list[float]]:
        """Compute BOS/EOS-aware mean-pooled embeddings for one layer."""
        from esm.sdk.api import ESMProtein  # noqa: PLC0415
        from esm.sdk.api import LogitsConfig  # noqa: PLC0415

        loaded = self.load(model)
        # The ESM-C module exposes custom encode/logits methods beyond the
        # nn.Module interface, so address it dynamically.
        net = cast(Any, loaded.model)
        # Request a single hidden layer rather than all of them. The SDK's
        # all-layers mode (ith_hidden_layer=-1) is unsupported for ESM-C 6B, and
        # a single-layer request also keeps memory flat for the 80-block model.
        # The SDK indexes hidden states 0-based with index 0 = the first block's
        # output, so a 1-based transformer layer k maps to ith_hidden_layer
        # k - 1; the final layer is k = len(blocks). This is verified to match
        # the previous all-layers indexing bit-for-bit on esmc_300m.
        ith_hidden_layer = transformer_layer - 1
        config = LogitsConfig(
            return_hidden_states=True,
            ith_hidden_layer=ith_hidden_layer,
        )
        result: list[list[float]] = []

        with torch.inference_mode():
            for sequence in sequences:
                tensor = net.encode(ESMProtein(sequence=sequence))
                output = net.logits(tensor, config)
                hidden_states = output.hidden_states
                if hidden_states is None:
                    raise RuntimeError('model did not return hidden states')
                # A single-layer request returns shape [1, batch, length, dim].
                # An out-of-range layer comes back empty ([0, ...]), so guard
                # before indexing to turn that into a clear error.
                if hidden_states.shape[0] != 1:
                    raise RuntimeError(
                        f'expected one hidden state for transformer layer '
                        f'{transformer_layer}, got {hidden_states.shape[0]}',
                    )
                # One sequence per call, so batch index 0; exclude the BOS and
                # EOS tokens ESM-C adds before mean-pooling over residues.
                layer = hidden_states[0][0]
                pooled = layer[1:-1].mean(dim=0)
                result.append(pooled.float().cpu().tolist())
        return result


def token_count(sequences: list[str]) -> int:
    """Count amino-acid characters across all sequences."""
    return sum(len(sequence) for sequence in sequences)


def create_app(
    settings: Settings | None = None,
    backend: EmbeddingBackend | None = None,
) -> FastAPI:
    """Create the application, optionally injecting a backend for tests."""
    app_settings = settings or Settings()
    embedding_backend = backend or EsmcEmbeddingBackend(app_settings)
    inference_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        if app_settings.preload_model is not None:
            await asyncio.to_thread(
                embedding_backend.load,
                app_settings.preload_model,
            )
        try:
            yield
        finally:
            embedding_backend.unload()

    app = FastAPI(
        title='ESM-C Embeddings API',
        version=__version__,
        lifespan=lifespan,
    )

    @app.get('/health', tags=['health'])
    async def health() -> dict[str, str]:
        """Report process health without forcing a model download."""
        return {'status': 'healthy'}

    @app.post(
        '/v1/embeddings',
        response_model=EmbeddingsResponse,
        tags=['embeddings'],
    )
    async def embeddings(request: EmbeddingsRequest) -> EmbeddingsResponse:
        """Compute one pooled embedding for each amino-acid sequence."""
        sequences = request.input
        try:
            # Keep the event loop responsive, while serializing access to the
            # resident model and accelerator within this server process.
            async with inference_lock:
                vectors = await asyncio.to_thread(
                    embedding_backend.embed,
                    request.model,
                    sequences,
                    request.resolved_transformer_layer(),
                )
        except (RuntimeError, ValueError) as error:
            LOGGER.exception('Embedding request failed')
            raise HTTPException(status_code=500, detail=str(error)) from error

        count = token_count(sequences)
        return EmbeddingsResponse(
            data=[
                EmbeddingObject(embedding=vector, index=index)
                for index, vector in enumerate(vectors)
            ],
            model=request.model,
            usage=Usage(prompt_tokens=count, total_tokens=count),
        )

    return app


app = create_app()


def main() -> None:
    """Run the API using configuration from environment variables."""
    settings = Settings()
    uvicorn.run(
        'esmc_service.server:app',
        host=settings.host,
        port=settings.port,
    )
