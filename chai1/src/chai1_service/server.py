"""FastAPI service for Chai-1 co-folding.

Folds a complex of protein / nucleic-acid / ligand chains and returns the
predicted structures as mmCIF text plus Chai-1's ranking scores (aggregate
score, complex pTM, interface ipTM, and an inter-chain clash flag). Follows the
``genslm-esm-service`` pattern: typed settings, an injectable backend
``Protocol`` (so the API is testable without a GPU or model weights), inference
serialized per process, and the structures returned inline in the JSON response.

Chai-1 is driven through ``chai_lab.chai1.run_inference``. The import is deferred
to the backend so this module imports without ``chai_lab`` installed, which keeps
the test suite GPU- and weight-free via the fake backend.

This service runs one fold per request synchronously. The async job API, residue
constraints, and precomputed-MSA directories sketched in ``README.md`` are
documented follow-ups, not implemented here.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import string
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

from chai1_service import __version__

LOGGER = logging.getLogger(__name__)

# Chai-1 is a single model; there is no model-name allowlist. The spine instead
# enumerates the entity types a chain may declare and the alphabet each polymer
# is validated against. Ligands carry a SMILES string rather than a sequence.
MODEL_NAME = 'chai-1'

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
# (carries a ``smiles`` string). Chai-1 co-folds them in a single complex.
EntityType = Literal['protein', 'dna', 'rna', 'ligand']


class Settings(BaseSettings):
    """Configuration loaded from ``CHAI_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix='CHAI_',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    device: str | None = None
    # Where FASTA inputs and CIF outputs are staged for each run. Defaults to a
    # per-run temporary directory under the system temp root.
    scratch_dir: str | None = None
    use_esm_embeddings: bool = True
    use_msa_server: bool = False
    host: str = '0.0.0.0'
    port: int = Field(default=8000, ge=1, le=65535)


class ChainInput(BaseModel):
    """One component of a complex and an optional explicit chain name.

    A ``protein``/``dna``/``rna`` chain carries a ``sequence``; a ``ligand``
    carries a ``smiles`` string instead and never a ``sequence``. ``name``
    becomes the chain id in the Chai-1 FASTA header; it is assigned
    automatically (A, B, C, …) when omitted.
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
    name: str | None = Field(
        default=None,
        description='Chain name (e.g. A, B). Assigned automatically when omitted.',
    )

    @field_validator('name')
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        """Reject a blank or non-token explicit chain name."""
        if value is None:
            return value
        name = value.strip()
        if not name:
            raise ValueError('chain name must be non-empty when provided')
        if any(character in name for character in '|= \t\n'):
            raise ValueError('chain name must not contain whitespace, | or =')
        return name

    @model_validator(mode='after')
    def validate_chain(self) -> ChainInput:
        """Enforce the fields and alphabet that match the chain entity type."""
        if self.entity_type == 'ligand':
            if self.sequence is not None:
                raise ValueError('ligand chain has no sequence; use smiles')
            if not (self.smiles and self.smiles.strip()):
                raise ValueError('ligand chain requires a smiles string')
            # SMILES is case-sensitive (c vs C); strip only.
            self.smiles = self.smiles.strip()
            return self

        # protein / dna / rna: a sequence, never a ligand smiles.
        if self.smiles is not None:
            raise ValueError(f'{self.entity_type} chain must not carry smiles')
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
    """A resolved chain (name guaranteed) handed to the backend.

    ``sequence`` is set for protein/dna/rna chains; ``smiles`` for ligands.
    ``entity_type`` selects the Chai-1 FASTA header keyword.
    """

    name: str
    entity_type: str
    sequence: str | None = None
    smiles: str | None = None

    def fasta_record(self) -> str:
        """Render this chain as a Chai-1 FASTA record (header + body)."""
        body = self.smiles if self.entity_type == 'ligand' else self.sequence
        return f'>{self.entity_type}|name={self.name}\n{body}\n'


def _next_chain_name(used: set[str]) -> str:
    """Return the next unused single-letter name, falling back to ``chainN``."""
    for letter in string.ascii_uppercase:
        if letter not in used:
            return letter
    index = 0
    while (candidate := f'chain{index}') in used:
        index += 1
    return candidate


class FoldRequest(BaseModel):
    """Request to co-fold one or more chains into a single complex."""

    model_config = ConfigDict(extra='forbid')

    chains: list[ChainInput]
    use_esm_embeddings: bool | None = Field(
        default=None,
        description='Override the service default for ESM embeddings.',
    )
    use_msa_server: bool | None = Field(
        default=None,
        description='Override the service default for the ColabFold MSA server.',
    )
    num_trunk_recycles: int = Field(default=3, ge=1, le=10)
    num_diffn_timesteps: int = Field(default=200, ge=1, le=500)
    num_diffn_samples: int = Field(default=5, ge=1, le=5)
    seed: int | None = Field(default=None, ge=0)
    return_pae: bool = Field(
        default=False,
        description='Include the predicted aligned error (PAE) matrix; large.',
    )

    @field_validator('chains')
    @classmethod
    def validate_chains(cls, value: list[ChainInput]) -> list[ChainInput]:
        """Require at least one chain."""
        if not value:
            raise ValueError('chains must contain at least one chain')
        return value

    @model_validator(mode='after')
    def assign_chain_names(self) -> FoldRequest:
        """Reject a ligand-only fold, dedupe names, and fill in missing ones."""
        if all(chain.entity_type == 'ligand' for chain in self.chains):
            raise ValueError('at least one protein, dna, or rna chain is required')
        used: set[str] = set()
        for chain in self.chains:
            if chain.name is not None:
                if chain.name in used:
                    raise ValueError(f'duplicate chain name: {chain.name!r}')
                used.add(chain.name)
        for chain in self.chains:
            if chain.name is None:
                chain.name = _next_chain_name(used)
                used.add(chain.name)
        return self

    def resolved_chains(self) -> list[ChainSpec]:
        """Return chains with names resolved (always set after validation)."""
        return [
            ChainSpec(
                name=cast(str, chain.name),
                entity_type=chain.entity_type,
                sequence=chain.sequence,
                smiles=chain.smiles,
            )
            for chain in self.chains
        ]


class FoldScores(BaseModel):
    """Chai-1 ranking scores for one predicted sample."""

    aggregate_score: float
    complex_ptm: float
    interface_ptm: float
    has_inter_chain_clashes: bool


class FoldSample(BaseModel):
    """One predicted structure sample and its scores."""

    structure: str
    format: Literal['mmcif'] = 'mmcif'
    scores: FoldScores
    pae: list[list[float]] | None = None


class FoldUsage(BaseModel):
    """Size of a fold request."""

    num_chains: int
    total_residues: int


class FoldResponse(BaseModel):
    """All predicted samples for one request, ranked best structure first."""

    object: Literal['fold'] = 'fold'
    model: str = MODEL_NAME
    num_chains: int
    chain_names: list[str]
    num_samples: int
    results: list[FoldSample]
    usage: FoldUsage


@dataclass
class FoldOutput:
    """One backend fold sample, decoupled from the HTTP response model."""

    mmcif: str
    aggregate_score: float
    complex_ptm: float
    interface_ptm: float
    has_inter_chain_clashes: bool
    pae: list[list[float]] | None = field(default=None)


def _scalar(value: Any) -> float:
    """Coerce a 0-d/1-element tensor or a number to a float."""
    item = getattr(value, 'item', None)
    if callable(item):
        return float(item())
    return float(value)


def _flag(value: Any) -> bool:
    """Coerce a 0-d/1-element tensor or a value to a bool."""
    item = getattr(value, 'item', None)
    if callable(item):
        return bool(item())
    return bool(value)


class FoldBackend(Protocol):
    """Interface used by the HTTP layer and test backends."""

    def load(self) -> None:
        """Prepare the backend (download weights / select device)."""

    def unload(self) -> None:
        """Release accelerator allocations."""

    def fold(
        self,
        chains: list[ChainSpec],
        *,
        use_esm_embeddings: bool,
        use_msa_server: bool,
        num_trunk_recycles: int,
        num_diffn_timesteps: int,
        num_diffn_samples: int,
        seed: int | None,
        return_pae: bool,
    ) -> list[FoldOutput]:
        """Co-fold the chains into one or more ranked complex samples."""
        ...


class ChaiFoldBackend:
    """Run Chai-1 co-folding synchronously through ``chai_lab``."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ready = False

    def _resolve_device(self) -> str:
        """Pick the configured device, else CUDA when present, else CPU."""
        if self.settings.device is not None:
            return self.settings.device
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def load(self) -> None:
        """Chai-1 loads weights inside ``run_inference``; nothing to preload."""
        # Importing here surfaces a missing dependency at startup rather than on
        # the first request, matching the preload intent of the template.
        from chai_lab.chai1 import run_inference  # noqa: F401, PLC0415

        self._ready = True

    def unload(self) -> None:
        """Release unused accelerator allocations."""
        self._ready = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def fold(
        self,
        chains: list[ChainSpec],
        *,
        use_esm_embeddings: bool,
        use_msa_server: bool,
        num_trunk_recycles: int,
        num_diffn_timesteps: int,
        num_diffn_samples: int,
        seed: int | None,
        return_pae: bool,
    ) -> list[FoldOutput]:
        """Write a FASTA, run Chai-1, and map every sample to a FoldOutput."""
        from chai_lab.chai1 import run_inference  # noqa: PLC0415

        device = self._resolve_device()
        fasta = ''.join(chain.fasta_record() for chain in chains)
        with tempfile.TemporaryDirectory(dir=self.settings.scratch_dir) as tmp:
            tmp_path = Path(tmp)
            fasta_file = tmp_path / 'input.fasta'
            fasta_file.write_text(fasta)
            output_dir = tmp_path / 'output'

            candidates = run_inference(
                fasta_file=fasta_file,
                output_dir=output_dir,
                use_esm_embeddings=use_esm_embeddings,
                use_msa_server=use_msa_server,
                num_trunk_recycles=num_trunk_recycles,
                num_diffn_timesteps=num_diffn_timesteps,
                num_diffn_samples=num_diffn_samples,
                seed=seed,
                device=device,
            )
            ranked = candidates.sorted()
            return self._map_candidates(ranked, return_pae=return_pae)

    @staticmethod
    def _map_candidates(candidates: Any, *, return_pae: bool) -> list[FoldOutput]:
        """Map a (sorted) StructureCandidates to one FoldOutput per sample."""
        cif_paths = list(candidates.cif_paths)
        ranking_data = list(candidates.ranking_data)
        pae_all = getattr(candidates, 'pae', None) if return_pae else None

        outputs: list[FoldOutput] = []
        for index, (cif_path, ranking) in enumerate(
            zip(cif_paths, ranking_data, strict=True),
        ):
            ptm = ranking.ptm_scores
            clashes = ranking.clash_scores
            pae: list[list[float]] | None = None
            if pae_all is not None:
                pae = [
                    [float(value) for value in row]
                    for row in pae_all[index].float().cpu().tolist()
                ]
            outputs.append(
                FoldOutput(
                    mmcif=Path(cif_path).read_text(),
                    aggregate_score=_scalar(ranking.aggregate_score),
                    complex_ptm=_scalar(ptm.complex_ptm),
                    interface_ptm=_scalar(ptm.interface_ptm),
                    has_inter_chain_clashes=_flag(clashes.has_inter_chain_clashes),
                    pae=pae,
                ),
            )
        return outputs


def create_app(
    settings: Settings | None = None,
    backend: FoldBackend | None = None,
) -> FastAPI:
    """Create the application, optionally injecting a backend for tests."""
    app_settings = settings or Settings()
    fold_backend = backend or ChaiFoldBackend(app_settings)
    inference_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        await asyncio.to_thread(fold_backend.load)
        try:
            yield
        finally:
            fold_backend.unload()

    app = FastAPI(
        title='Chai-1 Co-Folding API',
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
        use_esm = (
            request.use_esm_embeddings
            if request.use_esm_embeddings is not None
            else app_settings.use_esm_embeddings
        )
        use_msa = (
            request.use_msa_server
            if request.use_msa_server is not None
            else app_settings.use_msa_server
        )
        try:
            # Keep the event loop responsive while serializing access to the
            # resident model and accelerator within this server process.
            async with inference_lock:
                outputs = await asyncio.to_thread(
                    fold_backend.fold,
                    chains,
                    use_esm_embeddings=use_esm,
                    use_msa_server=use_msa,
                    num_trunk_recycles=request.num_trunk_recycles,
                    num_diffn_timesteps=request.num_diffn_timesteps,
                    num_diffn_samples=request.num_diffn_samples,
                    seed=request.seed,
                    return_pae=request.return_pae,
                )
        except (RuntimeError, ValueError, OSError) as error:
            LOGGER.exception('Fold request failed')
            raise HTTPException(status_code=500, detail=str(error)) from error

        samples = [
            FoldSample(
                structure=output.mmcif,
                scores=FoldScores(
                    aggregate_score=output.aggregate_score,
                    complex_ptm=output.complex_ptm,
                    interface_ptm=output.interface_ptm,
                    has_inter_chain_clashes=output.has_inter_chain_clashes,
                ),
                pae=output.pae,
            )
            for output in outputs
        ]
        # Chai-1's sorted() already ranks best-first; keep that order.
        return FoldResponse(
            num_chains=len(chains),
            chain_names=[chain.name for chain in chains],
            num_samples=len(samples),
            results=samples,
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
        'chai1_service.server:app',
        host=settings.host,
        port=settings.port,
    )


if __name__ == '__main__':
    main()
