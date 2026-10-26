"""FastAPI service for Boltz-2 co-folding.

Folds a complex of protein / nucleic-acid / ligand chains with Boltz-2 and
returns the predicted structures as mmCIF text plus Boltz-2's confidence scores
(the aggregate ``confidence_score`` it ranks by, pTM, ipTM, and complex pLDDT),
and optionally a small-molecule binding-affinity prediction. Follows the
``genslm-esm-service`` pattern: typed settings, an injectable backend
``Protocol`` (so the API is testable without a GPU or weights), inference
serialized per process, and the structures returned inline in the response.

Boltz-2 ships as the ``boltz`` command-line tool rather than a library call, so
the backend writes a YAML input, invokes ``boltz predict`` as a subprocess, and
reads back the CIF files and confidence JSON it writes. The subprocess call is a
single seam (:meth:`BoltzFoldBackend._run`) so the backend is testable without
the real executable.

This service runs one fold per request synchronously. An async job API (runs are
minutes) is a documented follow-up, not implemented here.
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import shutil
import string
import subprocess
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
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from boltz2_service import __version__

LOGGER = logging.getLogger(__name__)

MODEL_NAME = 'boltz-2'

_AA_ALPHABET = frozenset('ACDEFGHIKLMNPQRSTVWYXBUZO')
# Nucleotide alphabets (N = any base). RNA uses U where DNA uses T.
_DNA_ALPHABET = frozenset('ACGTN')
_RNA_ALPHABET = frozenset('ACGUN')
_POLYMER_ALPHABETS = {
    'protein': (_AA_ALPHABET, 'amino-acid'),
    'dna': (_DNA_ALPHABET, 'DNA'),
    'rna': (_RNA_ALPHABET, 'RNA'),
}

# A chain is a protein/DNA/RNA polymer (carries a ``sequence``) or a ligand
# (carries a ``smiles`` string or a ``ccd`` code). Boltz-2 co-folds them.
EntityType = Literal['protein', 'dna', 'rna', 'ligand']


class Settings(BaseSettings):
    """Configuration loaded from ``BOLTZ_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix='BOLTZ_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    # Path to the ``boltz`` executable, and where it caches weights.
    executable: str = 'boltz'
    cache_dir: str | None = None
    # Where YAML inputs and prediction outputs are staged per run.
    scratch_dir: str | None = None
    devices: int = Field(default=1, ge=1)
    use_msa_server: bool = False
    host: str = '0.0.0.0'
    port: int = Field(default=8000, ge=1, le=65535)


class ChainInput(BaseModel):
    """One component of a complex and an optional explicit chain id.

    A ``protein``/``dna``/``rna`` chain carries a ``sequence``; a ``ligand``
    carries exactly one of ``smiles`` or ``ccd`` and never a ``sequence``. ``id``
    becomes the chain id in the Boltz-2 YAML; it is assigned automatically (A, B,
    C, …) when omitted.
    """

    model_config = ConfigDict(extra='forbid')

    entity_type: EntityType = 'protein'
    sequence: str | None = Field(
        default=None,
        description='Polymer sequence (protein/dna/rna chains only).',
    )
    smiles: str | None = Field(
        default=None,
        description='Ligand SMILES string (ligand chains only).',
    )
    ccd: str | None = Field(
        default=None,
        description='Ligand CCD code (ligand chains only).',
    )
    id: str | None = Field(
        default=None,
        description='Chain id (e.g. A, B). Assigned automatically when omitted.',
    )

    @field_validator('id')
    @classmethod
    def validate_id(cls, value: str | None) -> str | None:
        """Reject a blank or non-token explicit chain id."""
        if value is None:
            return value
        name = value.strip()
        if not name:
            raise ValueError('chain id must be non-empty when provided')
        if any(character in name for character in '|= \t\n:'):
            raise ValueError('chain id must not contain whitespace, :, | or =')
        return name

    @model_validator(mode='after')
    def validate_chain(self) -> ChainInput:
        """Enforce the fields and alphabet that match the chain entity type."""
        if self.entity_type == 'ligand':
            if self.sequence is not None:
                raise ValueError('ligand chain has no sequence; use smiles or ccd')
            has_smiles = bool(self.smiles and self.smiles.strip())
            has_ccd = bool(self.ccd and self.ccd.strip())
            if has_smiles == has_ccd:
                raise ValueError('ligand chain requires exactly one of smiles or ccd')
            if has_smiles:
                # SMILES is case-sensitive (c vs C); strip only.
                self.smiles = cast(str, self.smiles).strip()
            else:
                self.ccd = cast(str, self.ccd).strip().upper()
            return self

        # protein / dna / rna: a sequence, never ligand fields.
        if self.smiles is not None or self.ccd is not None:
            raise ValueError(f'{self.entity_type} chain must not carry smiles or ccd')
        if self.sequence is None:
            raise ValueError(f'{self.entity_type} chain requires a sequence')
        sequence = self.sequence.strip().upper()
        if not sequence:
            raise ValueError('sequence must be a non-empty string')
        alphabet, label = _POLYMER_ALPHABETS[self.entity_type]
        invalid = sorted(set(sequence) - alphabet)
        if invalid:
            raise ValueError(f'{label} sequence has invalid residues: {invalid}')
        self.sequence = sequence
        return self


@dataclass(frozen=True)
class ChainSpec:
    """A resolved chain (id guaranteed) handed to the backend."""

    id: str
    entity_type: str
    sequence: str | None = None
    smiles: str | None = None
    ccd: str | None = None


def _next_chain_id(used: set[str]) -> str:
    """Return the next unused single-letter id, falling back to ``chainN``."""
    for letter in string.ascii_uppercase:
        if letter not in used:
            return letter
    index = 0
    while (candidate := f'chain{index}') in used:
        index += 1
    return candidate


def _yaml_scalar(value: str) -> str:
    """Single-quote a scalar for YAML, escaping embedded single quotes."""
    return "'" + value.replace("'", "''") + "'"


def build_yaml(chains: list[ChainSpec], affinity_binder: str | None) -> str:
    """Render chains (and an optional affinity target) as a Boltz-2 YAML input."""
    lines = ['version: 1', 'sequences:']
    for chain in chains:
        lines.append(f'  - {chain.entity_type}:')
        lines.append(f'      id: {chain.id}')
        if chain.entity_type == 'ligand':
            if chain.smiles is not None:
                lines.append(f'      smiles: {_yaml_scalar(chain.smiles)}')
            else:
                lines.append(f'      ccd: {_yaml_scalar(cast(str, chain.ccd))}')
        else:
            lines.append(f'      sequence: {_yaml_scalar(cast(str, chain.sequence))}')
    if affinity_binder is not None:
        lines.append('properties:')
        lines.append('  - affinity:')
        lines.append(f'      binder: {affinity_binder}')
    return '\n'.join(lines) + '\n'


class FoldRequest(BaseModel):
    """Request to co-fold one or more chains into a single complex."""

    model_config = ConfigDict(extra='forbid')

    chains: list[ChainInput]
    use_msa_server: bool | None = Field(
        default=None,
        description='Override the service default for the mmseqs2 MSA server.',
    )
    diffusion_samples: int = Field(default=1, ge=1, le=25)
    recycling_steps: int = Field(default=3, ge=1, le=10)
    affinity_binder: str | None = Field(
        default=None,
        description='Chain id of a ligand to predict binding affinity for.',
    )

    @field_validator('chains')
    @classmethod
    def validate_chains(cls, value: list[ChainInput]) -> list[ChainInput]:
        """Require at least one chain."""
        if not value:
            raise ValueError('chains must contain at least one chain')
        return value

    @model_validator(mode='after')
    def finalize(self) -> FoldRequest:
        """Reject ligand-only folds, assign ids, and check the affinity target."""
        if all(chain.entity_type == 'ligand' for chain in self.chains):
            raise ValueError('at least one protein, dna, or rna chain is required')
        used: set[str] = set()
        for chain in self.chains:
            if chain.id is not None:
                if chain.id in used:
                    raise ValueError(f'duplicate chain id: {chain.id!r}')
                used.add(chain.id)
        for chain in self.chains:
            if chain.id is None:
                chain.id = _next_chain_id(used)
                used.add(chain.id)
        if self.affinity_binder is not None:
            target = next(
                (chain for chain in self.chains if chain.id == self.affinity_binder),
                None,
            )
            if target is None:
                raise ValueError(
                    f'affinity_binder {self.affinity_binder!r} is not a chain id',
                )
            if target.entity_type != 'ligand':
                raise ValueError('affinity_binder must name a ligand chain')
        return self

    def resolved_chains(self) -> list[ChainSpec]:
        """Return chains with ids resolved (always set after validation)."""
        return [
            ChainSpec(
                id=cast(str, chain.id),
                entity_type=chain.entity_type,
                sequence=chain.sequence,
                smiles=chain.smiles,
                ccd=chain.ccd,
            )
            for chain in self.chains
        ]


class FoldScores(BaseModel):
    """Boltz-2 confidence scores for one predicted sample."""

    confidence_score: float
    ptm: float
    iptm: float
    complex_plddt: float


class FoldSample(BaseModel):
    """One predicted structure sample and its scores."""

    structure: str
    format: Literal['mmcif'] = 'mmcif'
    scores: FoldScores


class FoldUsage(BaseModel):
    """Size of a fold request."""

    num_chains: int
    total_residues: int


class FoldResponse(BaseModel):
    """All predicted samples for one request, ranked best structure first."""

    object: Literal['fold'] = 'fold'
    model: str = MODEL_NAME
    num_chains: int
    chain_ids: list[str]
    num_samples: int
    results: list[FoldSample]
    affinity: dict[str, Any] | None = None
    usage: FoldUsage


@dataclass
class FoldOutput:
    """One backend fold sample, decoupled from the HTTP response model."""

    mmcif: str
    confidence_score: float
    ptm: float
    iptm: float
    complex_plddt: float


@dataclass
class FoldResult:
    """Everything the backend returns for a request: ranked samples + affinity."""

    samples: list[FoldOutput]
    affinity: dict[str, Any] | None = field(default=None)


class FoldBackend(Protocol):
    """Interface used by the HTTP layer and test backends."""

    def load(self) -> None:
        """Prepare the backend (verify the executable / select device)."""

    def unload(self) -> None:
        """Release accelerator allocations."""

    def fold(
        self,
        chains: list[ChainSpec],
        *,
        use_msa_server: bool,
        diffusion_samples: int,
        recycling_steps: int,
        affinity_binder: str | None,
    ) -> FoldResult:
        """Co-fold the chains into one or more ranked complex samples."""
        ...


class BoltzFoldBackend:
    """Run Boltz-2 co-folding by invoking the ``boltz`` CLI as a subprocess."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load(self) -> None:
        """Verify the ``boltz`` executable is on PATH (or an absolute path)."""
        exe = self.settings.executable
        if shutil.which(exe) is None and not Path(exe).exists():
            raise RuntimeError(f'boltz executable not found: {exe!r}')

    def unload(self) -> None:
        """Release unused accelerator allocations."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run(self, cmd: list[str]) -> None:
        """Run ``boltz predict``; raise RuntimeError on a non-zero exit."""
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or '').strip()[-500:]
            raise RuntimeError(f'boltz predict failed ({completed.returncode}): {tail}')

    def fold(
        self,
        chains: list[ChainSpec],
        *,
        use_msa_server: bool,
        diffusion_samples: int,
        recycling_steps: int,
        affinity_binder: str | None,
    ) -> FoldResult:
        """Write a YAML, run Boltz-2, and map every sample to a FoldOutput."""
        yaml_text = build_yaml(chains, affinity_binder)
        with tempfile.TemporaryDirectory(dir=self.settings.scratch_dir) as tmp:
            tmp_path = Path(tmp)
            stem = 'input'
            yaml_file = tmp_path / f'{stem}.yaml'
            yaml_file.write_text(yaml_text)
            out_dir = tmp_path / 'out'

            cmd = [
                self.settings.executable,
                'predict',
                str(yaml_file),
                '--out_dir',
                str(out_dir),
                '--output_format',
                'mmcif',
                '--diffusion_samples',
                str(diffusion_samples),
                '--recycling_steps',
                str(recycling_steps),
                '--devices',
                str(self.settings.devices),
            ]
            if use_msa_server:
                cmd.append('--use_msa_server')
            if self.settings.cache_dir is not None:
                cmd.extend(['--cache', self.settings.cache_dir])

            self._run(cmd)
            return self._collect(out_dir / 'predictions' / stem, stem)

    @staticmethod
    def _collect(pred_dir: Path, stem: str) -> FoldResult:
        """Read the CIF + confidence files Boltz-2 wrote, ranked best-first."""
        samples: list[FoldOutput] = []
        cif_files = sorted(pred_dir.glob(f'{stem}_model_*.cif'))
        if not cif_files:
            raise RuntimeError(f'no Boltz-2 predictions found in {pred_dir}')
        for cif in cif_files:
            index = cif.stem.rsplit('_', 1)[-1]
            confidence_file = pred_dir / f'confidence_{stem}_model_{index}.json'
            scores = json.loads(confidence_file.read_text())
            samples.append(
                FoldOutput(
                    mmcif=cif.read_text(),
                    confidence_score=float(scores.get('confidence_score', 0.0)),
                    ptm=float(scores.get('ptm', 0.0)),
                    iptm=float(scores.get('iptm', 0.0)),
                    complex_plddt=float(scores.get('complex_plddt', 0.0)),
                ),
            )
        samples.sort(key=lambda sample: sample.confidence_score, reverse=True)

        affinity: dict[str, Any] | None = None
        affinity_file = pred_dir / f'affinity_{stem}.json'
        if affinity_file.exists():
            affinity = json.loads(affinity_file.read_text())
        return FoldResult(samples=samples, affinity=affinity)


def create_app(
    settings: Settings | None = None,
    backend: FoldBackend | None = None,
) -> FastAPI:
    """Create the application, optionally injecting a backend for tests."""
    app_settings = settings or Settings()
    fold_backend = backend or BoltzFoldBackend(app_settings)
    inference_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        await asyncio.to_thread(fold_backend.load)
        try:
            yield
        finally:
            fold_backend.unload()

    app = FastAPI(
        title='Boltz-2 Co-Folding API',
        version=__version__,
        lifespan=lifespan,
    )

    @app.get('/health', tags=['health'])
    async def health() -> dict[str, str]:
        """Report process health without forcing a model download."""
        return {'status': 'healthy'}

    @app.post('/v1/fold', response_model=FoldResponse, tags=['fold'])
    async def fold(request: FoldRequest) -> FoldResponse:
        """Co-fold the requested chains into a single complex structure."""
        chains = request.resolved_chains()
        use_msa = (
            request.use_msa_server
            if request.use_msa_server is not None
            else app_settings.use_msa_server
        )
        try:
            async with inference_lock:
                result = await asyncio.to_thread(
                    fold_backend.fold,
                    chains,
                    use_msa_server=use_msa,
                    diffusion_samples=request.diffusion_samples,
                    recycling_steps=request.recycling_steps,
                    affinity_binder=request.affinity_binder,
                )
        except (RuntimeError, ValueError, OSError) as error:
            LOGGER.exception('Fold request failed')
            raise HTTPException(status_code=500, detail=str(error)) from error

        samples = [
            FoldSample(
                structure=output.mmcif,
                scores=FoldScores(
                    confidence_score=output.confidence_score,
                    ptm=output.ptm,
                    iptm=output.iptm,
                    complex_plddt=output.complex_plddt,
                ),
            )
            for output in result.samples
        ]
        return FoldResponse(
            num_chains=len(chains),
            chain_ids=[chain.id for chain in chains],
            num_samples=len(samples),
            results=samples,
            affinity=result.affinity,
            usage=FoldUsage(
                num_chains=len(chains),
                total_residues=sum(
                    len(chain.sequence) for chain in chains if chain.sequence
                ),
            ),
        )

    return app


app = create_app()


def main() -> None:
    """Run the API using configuration from environment variables."""
    settings = Settings()
    uvicorn.run(
        'boltz2_service.server:app',
        host=settings.host,
        port=settings.port,
    )


if __name__ == '__main__':
    main()
