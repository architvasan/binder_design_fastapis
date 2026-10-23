"""FastAPI service for RFD4 / Proteina-Complexa binder design.

Wraps the ``complexa`` Hydra pipeline (RosettaCommons RFD4-Proteina, as ported
to Aurora in the ``aurora-proteina-complexa`` recipe) as a design-campaign
service. This is archetype C — an async design job — so unlike the sync
embedding/fold/sample services it:

* launches the pipeline as a **subprocess per request** (Hydra + ``main()``
  carry global state, and the recipe itself runs a fresh process per call),
* forces ``gen_njobs=1`` / ``eval_njobs=1`` (the pipeline's parallel branch
  resets ``ZE_AFFINITY_MASK`` and would fight the node's tile pinning),
* serializes jobs with a lock (one campaign per tile at a time), and
* returns design **file paths** on the node's shared storage rather than inline
  structures (a campaign produces many large files).

The service process is expected to run on the compute node inside the activated
Proteina-Complexa environment (see ``deploy/aurora_serve.sh``), so ``complexa``
is on ``PATH`` and the framework's checkpoint/cache env vars are set.

The canonical invocation this wraps (from the recipe)::

    complexa design configs/search_binder_local_pipeline.yaml \
        ++run_name=<unique> ++generation.task_name=<target> \
        ++gen_njobs=1 ++eval_njobs=1 ++seed=<n>
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Literal
from typing import Protocol

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from rfd4_proteina_service import __version__

LOGGER = logging.getLogger(__name__)

# Allowlist of pipelines mapped to their Hydra config (relative to the repo).
# This is both the set of legal pipeline names and the config lookup.
PIPELINES = {
    'binder': 'configs/search_binder_local_pipeline.yaml',
}

# Overrides the service controls itself; a request may not set them (notably
# gen_njobs/eval_njobs, which must stay 1 to preserve the node's tile pinning).
_RESERVED_OVERRIDE_KEYS = frozenset(
    {'run_name', 'seed', 'gen_njobs', 'eval_njobs', 'generation.task_name'},
)

# Safe identifier for task/run names: no shell metacharacters, no spaces.
_NAME_RE = re.compile(r'^[A-Za-z0-9._-]+$')
# A single Hydra override token: key=value, optional +/++ prefix.
_OVERRIDE_RE = re.compile(r'^\+?\+?[A-Za-z0-9_.]+=[A-Za-z0-9_./:-]+$')

# Structure extensions a design campaign writes.
_STRUCTURE_SUFFIXES = ('.cif', '.pdb')


class Settings(BaseSettings):
    """Configuration loaded from ``PROTEINA_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix='PROTEINA_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    # Defaults are the Aurora port paths from the aurora-proteina-complexa
    # recipe; override via PROTEINA_REPO_DIR / PROTEINA_COMMAND / etc.
    repo_dir: str = '/lus/flare/projects/FRAME-IDP/mmeigooni/software/Proteina-Complexa'
    command: str = 'complexa'
    runs_root: str | None = None
    job_timeout: float = Field(default=3600.0, gt=0)
    host: str = '0.0.0.0'
    port: int = Field(default=8000, ge=1, le=65535)

    def resolved_runs_root(self) -> Path:
        """Return the base directory for run outputs (default ``<repo>/runs``)."""
        if self.runs_root is not None:
            return Path(self.runs_root)
        return Path(self.repo_dir) / 'runs'


class DesignRequest(BaseModel):
    """Request to run one Proteina-Complexa binder-design campaign."""

    model_config = ConfigDict(extra='forbid')

    task_name: str = Field(
        description='Target key configured in the repo (e.g. 02_PDL1).',
    )
    pipeline: str = 'binder'
    run_name: str | None = Field(
        default=None,
        description='Unique run name; auto-generated when omitted.',
    )
    seed: int = Field(default=42, ge=0)
    extra_overrides: list[str] = Field(
        default_factory=list,
        description='Additional Hydra overrides as key=value tokens.',
    )

    @field_validator('pipeline')
    @classmethod
    def validate_pipeline(cls, value: str) -> str:
        """Only allow supported pipelines."""
        if value not in PIPELINES:
            raise ValueError(f'pipeline must be one of {sorted(PIPELINES)}')
        return value

    @field_validator('task_name', 'run_name')
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        """Reject names with shell metacharacters or spaces."""
        if value is not None and not _NAME_RE.match(value):
            raise ValueError('name must match [A-Za-z0-9._-]+')
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


class DesignItem(BaseModel):
    """One produced design (a structure file on the node's storage)."""

    id: str
    structure_path: str
    format: Literal['cif', 'pdb']


class DesignResponse(BaseModel):
    """Result of a design campaign: where it ran and what it produced."""

    object: Literal['design'] = 'design'
    pipeline: str
    task_name: str
    run_name: str
    output_dir: str
    returncode: int
    num_designs: int
    designs: list[DesignItem]
    log_tail: str


@dataclass
class DesignOutput:
    """Backend campaign result, decoupled from the HTTP response model."""

    run_name: str
    output_dir: str
    returncode: int
    designs: list[DesignItem]
    log_tail: str = field(default='')


class DesignBackend(Protocol):
    """Interface used by the HTTP layer and test backends."""

    def generate(self, request: DesignRequest) -> DesignOutput:
        """Run one design campaign and return its outputs."""
        ...


def _tail(text: str, limit: int = 4000) -> str:
    """Return the last ``limit`` characters of ``text``."""
    return text[-limit:]


class ProteinaBackend:
    """Run the Proteina-Complexa pipeline as a subprocess per campaign."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _build_argv(
        self,
        request: DesignRequest,
        run_name: str,
        out_dir: Path,
    ) -> list[str]:
        """Assemble the ``complexa design`` command line (no shell)."""
        config = PIPELINES[request.pipeline]
        return [
            self.settings.command,
            'design',
            config,
            f'++run_name={run_name}',
            f'++generation.task_name={request.task_name}',
            # NOTE: output location key is best-effort; confirm against
            # configs/search_binder_local_pipeline.yaml + docs/INFERENCE.md.
            f'out_dir={out_dir}',
            '++gen_njobs=1',
            '++eval_njobs=1',
            f'++seed={request.seed}',
            *request.extra_overrides,
        ]

    def _collect_designs(self, out_dir: Path) -> list[DesignItem]:
        """Gather produced structures from the run directory.

        NOTE: the exact layout is confirmed on-cluster against
        ``docs/INFERENCE.md``; designs are written under an
        ``inference_samples`` directory, with a fallback to any structure file.
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
        """Launch the campaign, verify it, and collect its designs."""
        run_name = request.run_name or (
            f'{request.task_name}-{request.seed}-{uuid.uuid4().hex[:8]}'
        )
        out_dir = self.settings.resolved_runs_root() / run_name
        argv = self._build_argv(request, run_name, out_dir)
        LOGGER.info('Launching Proteina campaign: %s', ' '.join(argv))

        try:
            proc = subprocess.run(
                argv,
                cwd=self.settings.repo_dir,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=self.settings.job_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f'Proteina campaign timed out after {self.settings.job_timeout}s',
            ) from error
        except FileNotFoundError as error:
            raise RuntimeError(
                f'Proteina command not found: {self.settings.command!r}',
            ) from error

        log_tail = _tail((proc.stdout or '') + (proc.stderr or ''))
        if proc.returncode != 0:
            raise RuntimeError(
                f'Proteina campaign failed (rc={proc.returncode}): {log_tail}',
            )

        designs = self._collect_designs(out_dir)
        if not designs:
            raise RuntimeError(
                f'Proteina campaign produced no designs in {out_dir}',
            )

        return DesignOutput(
            run_name=run_name,
            output_dir=str(out_dir),
            returncode=proc.returncode,
            designs=designs,
            log_tail=log_tail,
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
        yield

    app = FastAPI(
        title='RFD4 / Proteina-Complexa Design API',
        version=__version__,
        lifespan=lifespan,
    )

    @app.get('/health', tags=['health'])
    async def health() -> dict[str, str]:
        """Report process health without launching a campaign."""
        return {'status': 'healthy'}

    @app.post('/v1/design', response_model=DesignResponse, tags=['design'])
    async def design(request: DesignRequest) -> DesignResponse:
        """Run one binder-design campaign and return its produced designs."""
        try:
            # Serialize campaigns: one model job per assigned tile at a time.
            async with inference_lock:
                output = await asyncio.to_thread(design_backend.generate, request)
        except (RuntimeError, ValueError) as error:
            LOGGER.exception('Design campaign failed')
            raise HTTPException(status_code=500, detail=str(error)) from error

        return DesignResponse(
            pipeline=request.pipeline,
            task_name=request.task_name,
            run_name=output.run_name,
            output_dir=output.output_dir,
            returncode=output.returncode,
            num_designs=len(output.designs),
            designs=output.designs,
            log_tail=output.log_tail,
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
