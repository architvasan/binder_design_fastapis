"""FastAPI service for RFD4-Proteina binder/complex design.

Wraps the public RosettaCommons **RFD4-Proteina** model
(``rfd4.Proteina.from_pretrained("nvidia/RFD4-Proteina")`` +
``rfd4.generate(model, overrides=[...])``). Follows the ``genslm-esm-service``
pattern: typed settings, an injectable backend ``Protocol`` (so the API is
testable without a GPU or model weights), one model resident at a time, and
inference serialized per process.

This is a design-campaign service (archetype C): a request supplies the
condition spec (a contig/conditions YAML) and generation settings; the model
writes designs to a per-run output directory, and the service returns their
file paths on the node's shared storage rather than inline structures.

``rfd4_proteina`` is imported lazily in :meth:`load`/:meth:`generate` so this
module imports without it installed, which keeps the test suite GPU- and
weight-free via the fake backend.

The canonical invocation this wraps (from the repo README)::

    import rfd4_proteina as rfd4
    model = rfd4.Proteina.from_pretrained("nvidia/RFD4-Proteina")
    rfd4.generate(model, overrides=["inputs=binder.yaml", "out_dir=./runs/first",
                                    "num_replicates=2"])
"""

from __future__ import annotations

import asyncio
import gc
import logging
import re
import uuid
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
from pydantic import model_validator
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from rfd4_proteina_service import __version__

LOGGER = logging.getLogger(__name__)

# Allowlist of model names mapped to their Hugging Face checkpoint. This is both
# the set of legal model names and the loader lookup.
MODEL_REPOS = {
    'rfd4-proteina': 'nvidia/RFD4-Proteina',
}

# Overrides the service controls itself; a request may not set them.
_RESERVED_OVERRIDE_KEYS = frozenset(
    {'inputs', 'out_dir', 'num_replicates', 'seed', 'inference_experiment'},
)

# Safe run name: no shell metacharacters, no spaces.
_NAME_RE = re.compile(r'^[A-Za-z0-9._-]+$')
# A single Hydra override token: key=value, optional +/++ prefix.
_OVERRIDE_RE = re.compile(r'^\+?\+?[A-Za-z0-9_.]+=[A-Za-z0-9_./:-]+$')
# Structure extensions a design run writes.
_STRUCTURE_SUFFIXES = ('.cif', '.pdb')


class Settings(BaseSettings):
    """Configuration loaded from ``RFD4_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix='RFD4_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    hf_token: SecretStr | None = None
    hf_revision: str | None = None
    preload_model: str | None = None
    runs_root: str = './runs'
    host: str = '0.0.0.0'
    port: int = Field(default=8000, ge=1, le=65535)

    @field_validator('preload_model')
    @classmethod
    def validate_preload_model(cls, value: str | None) -> str | None:
        """Only allow a supported checkpoint to be preloaded."""
        if value is not None and value not in MODEL_REPOS:
            raise ValueError(f'preload_model must be one of {sorted(MODEL_REPOS)}')
        return value


class DesignRequest(BaseModel):
    """Request to run one RFD4-Proteina design campaign."""

    model_config = ConfigDict(extra='forbid')

    model: str = 'rfd4-proteina'
    inputs_yaml: str | None = Field(
        default=None,
        description='The condition-spec YAML (contig/conditions) as text.',
    )
    inputs_path: str | None = Field(
        default=None,
        description='Path to an existing condition-spec YAML on the node.',
    )
    run_name: str | None = Field(
        default=None,
        description='Unique run name; auto-generated when omitted.',
    )
    num_replicates: int = Field(default=1, ge=1, le=100)
    seed: int = Field(default=0, ge=0)
    inference_experiment: str = Field(default='quickstart')
    extra_overrides: list[str] = Field(default_factory=list)

    @field_validator('model')
    @classmethod
    def validate_model(cls, value: str) -> str:
        """Only allow supported checkpoints."""
        if value not in MODEL_REPOS:
            raise ValueError(f'model must be one of {sorted(MODEL_REPOS)}')
        return value

    @field_validator('run_name')
    @classmethod
    def validate_run_name(cls, value: str | None) -> str | None:
        """Reject run names with shell metacharacters or spaces."""
        if value is not None and not _NAME_RE.match(value):
            raise ValueError('run_name must match [A-Za-z0-9._-]+')
        return value

    @field_validator('inference_experiment')
    @classmethod
    def validate_experiment(cls, value: str) -> str:
        """Reject experiment names with unsafe characters."""
        if not _NAME_RE.match(value):
            raise ValueError('inference_experiment must match [A-Za-z0-9._-]+')
        return value

    @field_validator('extra_overrides')
    @classmethod
    def validate_overrides(cls, value: list[str]) -> list[str]:
        """Validate each override token and block service-controlled keys."""
        for override in value:
            if not _OVERRIDE_RE.match(override):
                raise ValueError(f'invalid override token: {override!r}')
            key = override.lstrip('+').split('=', 1)[0]
            if key in _RESERVED_OVERRIDE_KEYS:
                raise ValueError(f'override {key!r} is controlled by the service')
        return value

    @model_validator(mode='after')
    def validate_inputs_source(self) -> DesignRequest:
        """Require exactly one of inputs_yaml / inputs_path."""
        if (self.inputs_yaml is None) == (self.inputs_path is None):
            raise ValueError('provide exactly one of inputs_yaml or inputs_path')
        if self.inputs_yaml is not None and not self.inputs_yaml.strip():
            raise ValueError('inputs_yaml must not be empty')
        return self


class DesignItem(BaseModel):
    """One produced design (a structure file on the node's storage)."""

    id: str
    structure_path: str
    format: Literal['cif', 'pdb']


class DesignResponse(BaseModel):
    """Result of a design campaign: where it ran and what it produced."""

    object: Literal['design'] = 'design'
    model: str
    run_name: str
    output_dir: str
    num_designs: int
    designs: list[DesignItem]


@dataclass
class DesignOutput:
    """Backend campaign result, decoupled from the HTTP response model."""

    run_name: str
    output_dir: str
    designs: list[DesignItem] = field(default_factory=list)


class DesignBackend(Protocol):
    """Interface used by the HTTP layer and test backends."""

    def load(self, model_id: str) -> object:
        """Load or return the requested model."""

    def unload(self) -> None:
        """Release the resident model."""

    def generate(self, request: DesignRequest) -> DesignOutput:
        """Run one design campaign and return its outputs."""
        ...


@dataclass
class LoadedModel:
    """The single model currently resident in memory."""

    model_id: str
    model: object


class ProteinaBackend:
    """Load one RFD4-Proteina model and run design campaigns in-process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.loaded: LoadedModel | None = None

    def load(self, model_id: str) -> LoadedModel:
        """Load or return the requested model."""
        if self.loaded is not None and self.loaded.model_id == model_id:
            return self.loaded

        self.unload()
        LOGGER.info('Loading model %s', model_id)
        if self.settings.hf_token is not None:
            from huggingface_hub import login  # noqa: PLC0415

            login(
                token=self.settings.hf_token.get_secret_value(),
                add_to_git_credential=False,
            )

        import rfd4_proteina as rfd4  # noqa: PLC0415

        repo = MODEL_REPOS[model_id]
        proteina = cast(Any, rfd4).Proteina
        if self.settings.hf_revision is not None:
            model = proteina.from_pretrained(repo, revision=self.settings.hf_revision)
        else:
            model = proteina.from_pretrained(repo)

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

    def _build_overrides(
        self,
        request: DesignRequest,
        inputs_arg: str,
        out_dir: Path,
    ) -> list[str]:
        """Assemble the Hydra override list passed to rfd4.generate."""
        return [
            f'inputs={inputs_arg}',
            f'out_dir={out_dir}',
            f'num_replicates={request.num_replicates}',
            f'seed={request.seed}',
            f'inference_experiment={request.inference_experiment}',
            *request.extra_overrides,
        ]

    def _collect_designs(self, out_dir: Path) -> list[DesignItem]:
        """Gather produced structures from the run directory.

        NOTE: designs are written under ``<out_dir>/inference_samples/`` per the
        repo README; confirm the exact layout against the repo's inference docs.
        Falls back to any structure file under the run directory.
        """
        candidates: list[Path] = []
        for sample_dir in out_dir.rglob('inference_samples'):
            for suffix in _STRUCTURE_SUFFIXES:
                candidates.extend(sorted(sample_dir.glob(f'*{suffix}')))
        if not candidates:
            for suffix in _STRUCTURE_SUFFIXES:
                candidates.extend(sorted(out_dir.rglob(f'*{suffix}')))
        return [
            DesignItem(
                id=path.stem,
                structure_path=str(path),
                format=path.suffix.lstrip('.'),  # type: ignore[arg-type]
            )
            for path in candidates
        ]

    def generate(self, request: DesignRequest) -> DesignOutput:
        """Run the campaign in-process and collect its designs."""
        import rfd4_proteina as rfd4  # noqa: PLC0415

        run_name = request.run_name or f'rfd4-{request.seed}-{uuid.uuid4().hex[:8]}'
        out_dir = Path(self.settings.runs_root) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        if request.inputs_yaml is not None:
            inputs_file = out_dir / 'inputs.yaml'
            inputs_file.write_text(request.inputs_yaml)
            inputs_arg = str(inputs_file)
        else:
            inputs_arg = cast(str, request.inputs_path)

        overrides = self._build_overrides(request, inputs_arg, out_dir)
        loaded = self.load(request.model)
        LOGGER.info('Running rfd4.generate with overrides: %s', overrides)
        with torch.inference_mode():
            cast(Any, rfd4).generate(loaded.model, overrides=overrides)

        designs = self._collect_designs(out_dir)
        if not designs:
            raise RuntimeError(f'campaign produced no designs in {out_dir}')

        return DesignOutput(
            run_name=run_name,
            output_dir=str(out_dir),
            designs=designs,
        )


def create_app(
    settings: Settings | None = None,
    backend: DesignBackend | None = None,
) -> FastAPI:
    """Create the application, optionally injecting a backend for tests."""
    app_settings = settings or Settings()
    design_backend = backend or ProteinaBackend(app_settings)
    inference_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        if app_settings.preload_model is not None:
            await asyncio.to_thread(design_backend.load, app_settings.preload_model)
        try:
            yield
        finally:
            design_backend.unload()

    app = FastAPI(
        title='RFD4-Proteina Design API',
        version=__version__,
        lifespan=lifespan,
    )

    @app.get('/health', tags=['health'])
    async def health() -> dict[str, str]:
        """Report process health without forcing a model download."""
        return {'status': 'healthy'}

    @app.post('/v1/design', response_model=DesignResponse, tags=['design'])
    async def design(request: DesignRequest) -> DesignResponse:
        """Run one design campaign and return its produced designs."""
        try:
            # Serialize campaigns: one resident model / accelerator at a time.
            async with inference_lock:
                output = await asyncio.to_thread(design_backend.generate, request)
        except (RuntimeError, ValueError, OSError) as error:
            LOGGER.exception('Design campaign failed')
            raise HTTPException(status_code=500, detail=str(error)) from error

        return DesignResponse(
            model=request.model,
            run_name=output.run_name,
            output_dir=output.output_dir,
            num_designs=len(output.designs),
            designs=output.designs,
        )

    return app


app = create_app()


def main() -> None:
    """Run the API using configuration from environment variables."""
    settings = Settings()
    uvicorn.run(
        'rfd4_proteina_service.server:app',
        host=settings.host,
        port=settings.port,
    )
