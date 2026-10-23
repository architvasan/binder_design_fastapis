"""FastAPI service for BioEmu2 equilibrium conformational sampling.

Given a single protein sequence (BioEmu2 is monomer-only), draws samples from
the approximated equilibrium distribution and returns the ensemble topology as
mmCIF plus the sampled trajectory. Follows the ``genslm-esm-service`` pattern:
typed settings, an injectable backend ``Protocol`` (so the API is testable
without a GPU or model weights), one model resident at a time, and inference
serialized per process.

BioEmu2 is loaded through the Microsoft ``bioemu2`` package. The import is
deferred to :meth:`load`/:meth:`sample` so this module imports without
``bioemu2`` installed, which keeps the test suite GPU- and weight-free via the
fake backend.

Sampling runs synchronously: one request produces one ensemble. The topology
(mmCIF text) is returned inline; the binary ``.xtc`` trajectory is opt-in and
returned base64-encoded.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import logging
import tempfile
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
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
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from bioemu2_service import __version__

LOGGER = logging.getLogger(__name__)

# The allowlist of BioEmu2 checkpoints. The value is the checkpoint argument for
# ``load_model`` (None selects the package default). This is both the set of
# legal model names and the loader lookup.
MODEL_CHECKPOINTS: dict[str, str | None] = {
    'bioemu2': None,
}

# Accepted residue alphabet: the 20 standard amino acids plus the ambiguity and
# non-standard codes (X/B/U/Z/O).
_AA_ALPHABET = frozenset('ACDEFGHIKLMNPQRSTVWYXBUZO')

# Filenames written by ``bioemu2.save_frames`` into the output directory.
_TOPOLOGY_FILE = 'topology.cif'
_TRAJECTORY_FILE = 'samples.xtc'


class Settings(BaseSettings):
    """Configuration loaded from ``BIOEMU2_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix='BIOEMU2_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    preload_model: str | None = None
    host: str = '0.0.0.0'
    port: int = Field(default=8000, ge=1, le=65535)

    @field_validator('preload_model')
    @classmethod
    def validate_preload_model(cls, value: str | None) -> str | None:
        """Only allow a supported BioEmu2 checkpoint to be preloaded."""
        if value is not None and value not in MODEL_CHECKPOINTS:
            raise ValueError(
                f'preload_model must be one of {sorted(MODEL_CHECKPOINTS)}',
            )
        return value


class SampleRequest(BaseModel):
    """Request to sample an equilibrium ensemble for one protein sequence."""

    model_config = ConfigDict(extra='forbid')

    model: str
    sequence: str
    num_samples: int = Field(default=10, ge=1, le=1000)
    mode: Literal['equilibrium'] = 'equilibrium'
    return_trajectory: bool = Field(
        default=False,
        description='Include the base64-encoded .xtc trajectory in the response.',
    )

    @field_validator('model')
    @classmethod
    def validate_model(cls, value: str) -> str:
        """Only allow supported BioEmu2 checkpoints."""
        if value not in MODEL_CHECKPOINTS:
            raise ValueError(f'model must be one of {sorted(MODEL_CHECKPOINTS)}')
        return value

    @field_validator('sequence')
    @classmethod
    def validate_sequence(cls, value: str) -> str:
        """Uppercase and reject empty or non-amino-acid sequences."""
        sequence = value.strip().upper()
        if not sequence:
            raise ValueError('sequence must be a non-empty amino-acid string')
        invalid = sorted(set(sequence) - _AA_ALPHABET)
        if invalid:
            raise ValueError(f'sequence has invalid residues: {invalid}')
        return sequence


class SampleUsage(BaseModel):
    """Size of a sampling request."""

    sequence_length: int
    num_samples: int


class SampleResponse(BaseModel):
    """An ensemble topology, optional trajectory, and request usage."""

    object: Literal['sample'] = 'sample'
    model: str
    format: Literal['mmcif'] = 'mmcif'
    topology: str
    num_samples: int
    trajectory_xtc_base64: str | None = None
    usage: SampleUsage


@dataclass
class SampleOutput:
    """Backend sampling result, decoupled from the HTTP response model."""

    topology_cif: str
    num_samples: int
    trajectory_xtc: bytes | None = field(default=None)


class SampleBackend(Protocol):
    """Interface used by the HTTP layer and test backends."""

    def load(self, model_id: str) -> object:
        """Load or return the requested model."""

    def unload(self) -> None:
        """Release the resident model."""

    def sample(
        self,
        model: str,
        sequence: str,
        *,
        num_samples: int,
        mode: str,
        return_trajectory: bool,
    ) -> SampleOutput:
        """Sample an equilibrium ensemble for the sequence."""
        ...


@dataclass
class LoadedModel:
    """The single model currently resident in memory."""

    model_id: str
    model: object


class BioEmu2Backend:
    """Load one BioEmu2 model at a time and sample ensembles synchronously."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loaded: LoadedModel | None = None

    def load(self, model_id: str) -> LoadedModel:
        """Load or return the requested model."""
        if self.loaded is not None and self.loaded.model_id == model_id:
            return self.loaded

        self.unload()
        LOGGER.info('Loading model %s', model_id)
        from bioemu2 import load_model  # noqa: PLC0415

        checkpoint = MODEL_CHECKPOINTS[model_id]
        if checkpoint is None:
            model = load_model()
        else:
            model = load_model(checkpoint=checkpoint)

        self.loaded = LoadedModel(model_id, model)
        LOGGER.info('Loaded model %s', model_id)
        return self.loaded

    def unload(self) -> None:
        """Release the resident model and unused accelerator allocations."""
        if self.loaded is None:
            return
        self.loaded = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def sample(
        self,
        model: str,
        sequence: str,
        *,
        num_samples: int,
        mode: str,
        return_trajectory: bool,
    ) -> SampleOutput:
        """Sample an ensemble and map the written frames to SampleOutput."""
        from bioemu2 import EmuSession  # noqa: PLC0415
        from bioemu2 import save_frames  # noqa: PLC0415
        from bioemu2 import System  # noqa: PLC0415

        loaded = self.load(model)
        # NOTE: confirm the exact sequence constructor against bioemu2
        # docs/system.md (System.from_sequence vs from_fasta); this is the only
        # call that depends on the real package surface.
        system = cast(Any, System).from_sequence(sequence)
        session = cast(Any, EmuSession)(system, model=loaded.model, mode=mode)
        samples = session.sample(num_samples=num_samples)

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            cast(Any, save_frames)(samples, out_dir)
            topology_path = out_dir / _TOPOLOGY_FILE
            if not topology_path.exists():
                raise RuntimeError(
                    f'bioemu2 did not write {_TOPOLOGY_FILE} to the output dir',
                )
            topology_cif = topology_path.read_text()
            trajectory: bytes | None = None
            if return_trajectory:
                trajectory_path = out_dir / _TRAJECTORY_FILE
                if trajectory_path.exists():
                    trajectory = trajectory_path.read_bytes()

        return SampleOutput(
            topology_cif=topology_cif,
            num_samples=num_samples,
            trajectory_xtc=trajectory,
        )


def create_app(
    settings: Settings | None = None,
    backend: SampleBackend | None = None,
) -> FastAPI:
    """Create the application, optionally injecting a backend for tests."""
    app_settings = settings or Settings()
    sample_backend = backend or BioEmu2Backend(app_settings)
    inference_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        if app_settings.preload_model is not None:
            await asyncio.to_thread(sample_backend.load, app_settings.preload_model)
        try:
            yield
        finally:
            sample_backend.unload()

    app = FastAPI(
        title='BioEmu2 Equilibrium Sampling API',
        version=__version__,
        lifespan=lifespan,
    )

    @app.get('/health', tags=['health'])
    async def health() -> dict[str, str]:
        """Report process health without forcing a model download."""
        return {'status': 'healthy'}

    @app.post('/v1/sample', response_model=SampleResponse, tags=['sample'])
    async def sample(request: SampleRequest) -> SampleResponse:
        """Sample an equilibrium ensemble for the requested sequence."""
        try:
            # Keep the event loop responsive while serializing access to the
            # resident model and accelerator within this server process.
            async with inference_lock:
                output = await asyncio.to_thread(
                    sample_backend.sample,
                    request.model,
                    request.sequence,
                    num_samples=request.num_samples,
                    mode=request.mode,
                    return_trajectory=request.return_trajectory,
                )
        except (RuntimeError, ValueError) as error:
            LOGGER.exception('Sampling request failed')
            raise HTTPException(status_code=500, detail=str(error)) from error

        trajectory_b64 = (
            base64.b64encode(output.trajectory_xtc).decode('ascii')
            if output.trajectory_xtc is not None
            else None
        )
        return SampleResponse(
            model=request.model,
            topology=output.topology_cif,
            num_samples=output.num_samples,
            trajectory_xtc_base64=trajectory_b64,
            usage=SampleUsage(
                sequence_length=len(request.sequence),
                num_samples=output.num_samples,
            ),
        )

    return app


app = create_app()


def main() -> None:
    """Run the API using configuration from environment variables."""
    settings = Settings()
    uvicorn.run(
        'bioemu2_service.server:app',
        host=settings.host,
        port=settings.port,
    )
