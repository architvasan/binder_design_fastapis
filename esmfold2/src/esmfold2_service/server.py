"""FastAPI service for ESMFold2 multichain structure prediction.

Folds one or more protein chains into a single complex and returns the structure
as mmCIF text plus confidence scores (pLDDT, pTM, and the interface ipTM that
makes multichain folding worthwhile). Follows the ``genslm-esm-service`` pattern:
typed settings, an injectable backend ``Protocol`` (so the API is testable
without a GPU or model weights), one model resident at a time, and inference
serialized per process.

ESMFold2 is loaded through the EvolutionaryScale ``esm`` SDK
(``EsmFold2Model.from_pretrained``). The import is deferred to :meth:`load` so
this module imports without ``esm`` installed, which keeps the test suite GPU-
and weight-free via the fake backend.

The service runs in single-sequence mode (no MSA): one request folds one
complex synchronously and the mmCIF is returned inline in the JSON response.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import string
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from dataclasses import field
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

from esmfold2_service import __version__

LOGGER = logging.getLogger(__name__)

# The allowlist of locally loadable ESMFold2 checkpoints mapped to their
# Hugging Face repository id. This is both the set of legal model names and the
# lookup the loader uses. ``esmfold2-fast`` is single-sequence only; both run
# multichain protein folding without an MSA.
MODEL_REPOS = {
    'esmfold2': 'biohub/ESMFold2',
    'esmfold2-fast': 'biohub/esmfold2-fast',
}

# Accepted residue alphabet: the 20 standard amino acids plus the ambiguity and
# non-standard codes ESMFold2 tolerates (X/B/U/Z/O).
_AA_ALPHABET = frozenset('ACDEFGHIKLMNPQRSTVWYXBUZO')
# Nucleotide alphabets (N = any base). RNA uses U where DNA uses T.
_DNA_ALPHABET = frozenset('ACGTN')
_RNA_ALPHABET = frozenset('ACGUN')
# Per-type polymer alphabet and a human-readable label for error messages.
_POLYMER_ALPHABETS = {
    'protein': (_AA_ALPHABET, 'amino-acid'),
    'dna': (_DNA_ALPHABET, 'DNA'),
    'rna': (_RNA_ALPHABET, 'RNA'),
}

# A chain is a protein/DNA/RNA polymer (carries a ``sequence``) or a ligand
# (carries ``smiles`` or ``ccd``). ESMFold2 folds them together in one complex.
ChainType = Literal['protein', 'dna', 'rna', 'ligand']


class Settings(BaseSettings):
    """Configuration loaded from ``ESMFOLD2_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix='ESMFOLD2_',
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
        """Only allow a supported ESMFold2 checkpoint to be preloaded."""
        if value is not None and value not in MODEL_REPOS:
            raise ValueError(f'preload_model must be one of {sorted(MODEL_REPOS)}')
        return value


class ChainInput(BaseModel):
    """One component of a complex and an optional explicit chain id.

    A ``protein``/``dna``/``rna`` chain carries a ``sequence``. A ``ligand``
    carries exactly one of ``smiles`` or ``ccd`` (a list of CCD codes) instead,
    and never a ``sequence``. ``type`` defaults to ``protein`` so a bare
    ``{"sequence": ...}`` keeps working.
    """

    model_config = ConfigDict(extra='forbid')

    type: ChainType = 'protein'
    sequence: str | None = Field(
        default=None,
        description='Polymer sequence (protein/dna/rna chains only).',
    )
    smiles: str | None = Field(
        default=None,
        description='Ligand SMILES string (ligand chains only).',
    )
    ccd: list[str] | None = Field(
        default=None,
        description='Ligand CCD component codes (ligand chains only).',
    )
    id: str | None = Field(
        default=None,
        description='Chain id (e.g. A, B). Assigned automatically when omitted.',
    )

    @field_validator('id')
    @classmethod
    def validate_id(cls, value: str | None) -> str | None:
        """Reject a blank explicit chain id."""
        if value is not None and not value.strip():
            raise ValueError('chain id must be non-empty when provided')
        return value

    @model_validator(mode='after')
    def validate_chain(self) -> ChainInput:
        """Enforce the fields and alphabet that match the chain ``type``."""
        if self.type == 'ligand':
            if self.sequence is not None:
                raise ValueError('ligand chain has no sequence; use smiles or ccd')
            has_smiles = bool(self.smiles and self.smiles.strip())
            has_ccd = bool(self.ccd)
            if has_smiles == has_ccd:
                raise ValueError('ligand chain requires exactly one of smiles or ccd')
            if has_smiles:
                # SMILES is case-sensitive (c vs C); strip only.
                self.smiles = cast(str, self.smiles).strip()
            else:
                codes = [code.strip().upper() for code in cast('list[str]', self.ccd)]
                if any(not code for code in codes):
                    raise ValueError('ccd codes must be non-empty')
                self.ccd = codes
            return self

        # protein / dna / rna: a sequence, never ligand fields.
        if self.smiles is not None or self.ccd is not None:
            raise ValueError(f'{self.type} chain must not carry smiles or ccd')
        if self.sequence is None:
            raise ValueError(f'{self.type} chain requires a sequence')
        sequence = self.sequence.strip().upper()
        if not sequence:
            raise ValueError('sequence must be a non-empty string')
        alphabet, label = _POLYMER_ALPHABETS[self.type]
        invalid = sorted(set(sequence) - alphabet)
        if invalid:
            raise ValueError(f'{label} sequence has invalid residues: {invalid}')
        self.sequence = sequence
        return self


@dataclass(frozen=True)
class ChainSpec:
    """A resolved chain (id guaranteed) handed to the backend.

    ``sequence`` is set for protein/dna/rna chains; ``smiles``/``ccd`` for
    ligands. ``type`` selects the SDK input class the backend builds.
    """

    id: str
    type: str = 'protein'
    sequence: str | None = None
    smiles: str | None = None
    ccd: tuple[str, ...] | None = None


def _next_chain_id(used: set[str]) -> str:
    """Return the next unused single-letter id, falling back to ``chainN``."""
    for letter in string.ascii_uppercase:
        if letter not in used:
            return letter
    index = 0
    while (candidate := f'chain{index}') in used:
        index += 1
    return candidate


class FoldRequest(BaseModel):
    """Request to fold one or more protein chains into a single complex."""

    model_config = ConfigDict(extra='forbid')

    model: str
    sequences: list[ChainInput]
    num_loops: int = Field(default=20, ge=1, le=100)
    num_sampling_steps: int = Field(default=100, ge=1, le=500)
    num_diffusion_samples: int = Field(default=1, ge=1, le=10)
    seed: int = Field(default=0, ge=0)
    return_pae: bool = Field(
        default=False,
        description='Include the predicted aligned error (PAE) matrix; large.',
    )

    @field_validator('model')
    @classmethod
    def validate_model(cls, value: str) -> str:
        """Only allow supported ESMFold2 checkpoints."""
        if value not in MODEL_REPOS:
            raise ValueError(f'model must be one of {sorted(MODEL_REPOS)}')
        return value

    @field_validator('sequences')
    @classmethod
    def validate_sequences(cls, value: list[ChainInput]) -> list[ChainInput]:
        """Require at least one chain."""
        if not value:
            raise ValueError('sequences must contain at least one chain')
        return value

    @model_validator(mode='after')
    def assign_chain_ids(self) -> FoldRequest:
        """Reject a ligand-only fold, dedupe ids, and auto-assign missing ones."""
        if all(chain.type == 'ligand' for chain in self.sequences):
            raise ValueError('at least one protein, dna, or rna chain is required')
        used: set[str] = set()
        for chain in self.sequences:
            if chain.id is not None:
                if chain.id in used:
                    raise ValueError(f'duplicate chain id: {chain.id!r}')
                used.add(chain.id)
        for chain in self.sequences:
            if chain.id is None:
                chain.id = _next_chain_id(used)
                used.add(chain.id)
        return self

    def resolved_chains(self) -> list[ChainSpec]:
        """Return chains with ids resolved (always set after validation)."""
        return [
            ChainSpec(
                id=cast(str, chain.id),
                type=chain.type,
                sequence=chain.sequence,
                smiles=chain.smiles,
                ccd=tuple(chain.ccd) if chain.ccd is not None else None,
            )
            for chain in self.sequences
        ]


class FoldUsage(BaseModel):
    """Size of a fold request."""

    num_chains: int
    total_residues: int


class FoldSample(BaseModel):
    """One predicted structure sample and its confidence scores."""

    structure: str
    format: Literal['mmcif'] = 'mmcif'
    plddt_mean: float
    plddt: list[float]
    ptm: float
    iptm: float | None = None
    pae: list[list[float]] | None = None
    ranking_score: float = Field(
        description='Quality score used to rank samples; higher is better. '
        '0.8*iptm + 0.2*ptm for complexes, else ptm.',
    )


class FoldResponse(BaseModel):
    """All predicted samples for one request, ranked best structure first."""

    object: Literal['fold'] = 'fold'
    model: str
    num_chains: int
    chain_ids: list[str]
    num_samples: int
    results: list[FoldSample]
    usage: FoldUsage


@dataclass
class FoldOutput:
    """One backend fold sample, decoupled from the HTTP response model."""

    mmcif: str
    plddt: list[float]
    ptm: float
    iptm: float | None
    pae: list[list[float]] | None = field(default=None)


def ranking_score(ptm: float, iptm: float | None) -> float:
    """Rank samples by overall quality; higher is better.

    Uses the AlphaFold-Multimer convention ``0.8*iptm + 0.2*ptm`` when an
    interface score is available (multichain complexes), otherwise ``ptm``.
    """
    if iptm is not None:
        return 0.8 * iptm + 0.2 * ptm
    return ptm


class FoldBackend(Protocol):
    """Interface used by the HTTP layer and test backends."""

    def load(self, model_id: str) -> object:
        """Load or return the requested model."""

    def unload(self) -> None:
        """Release the resident model."""

    def fold(
        self,
        model: str,
        chains: list[ChainSpec],
        *,
        num_loops: int,
        num_sampling_steps: int,
        num_diffusion_samples: int,
        seed: int,
        return_pae: bool,
    ) -> list[FoldOutput]:
        """Fold the chains into one or more complex samples."""
        ...


@dataclass
class LoadedModel:
    """The single model currently resident in memory."""

    model_id: str
    model: torch.nn.Module
    device: torch.device


class EsmFold2Backend:
    """Load one ESMFold2 model at a time and fold complexes synchronously."""

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

        from esm.models.esmfold2 import EsmFold2Model  # noqa: PLC0415

        device = self._resolve_device()
        repo = MODEL_REPOS[model_id]
        model = EsmFold2Model.from_pretrained(repo, device=str(device))
        model = cast(torch.nn.Module, model).eval()

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

    def fold(
        self,
        model: str,
        chains: list[ChainSpec],
        *,
        num_loops: int,
        num_sampling_steps: int,
        num_diffusion_samples: int,
        seed: int,
        return_pae: bool,
    ) -> list[FoldOutput]:
        """Fold the chains and map every sample to a FoldOutput."""
        from esm.models.esmfold2 import DNAInput  # noqa: PLC0415
        from esm.models.esmfold2 import ESMFold2InputBuilder  # noqa: PLC0415
        from esm.models.esmfold2 import LigandInput  # noqa: PLC0415
        from esm.models.esmfold2 import ProteinInput  # noqa: PLC0415
        from esm.models.esmfold2 import RNAInput  # noqa: PLC0415
        from esm.models.esmfold2 import StructurePredictionInput  # noqa: PLC0415

        def build_input(chain: ChainSpec) -> Any:
            """Map a resolved chain to its ESMFold2 SDK input dataclass."""
            if chain.type == 'protein':
                return ProteinInput(id=chain.id, sequence=chain.sequence)
            if chain.type == 'dna':
                return DNAInput(id=chain.id, sequence=chain.sequence)
            if chain.type == 'rna':
                return RNAInput(id=chain.id, sequence=chain.sequence)
            # ligand: carries smiles or ccd, never a sequence.
            return LigandInput(
                id=chain.id,
                smiles=chain.smiles,
                ccd=list(chain.ccd) if chain.ccd is not None else None,
            )

        loaded = self.load(model)
        # ESMFold2 exposes custom methods beyond the nn.Module interface.
        net = cast(Any, loaded.model)
        spi = StructurePredictionInput(
            sequences=[build_input(chain) for chain in chains],
        )

        with torch.inference_mode():
            result = ESMFold2InputBuilder().fold(
                net,
                spi,
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                num_diffusion_samples=num_diffusion_samples,
                seed=seed,
            )

        # fold() returns a single result when num_diffusion_samples == 1 and a
        # list when more than one sample is drawn; normalise to a list so the
        # service always returns one FoldOutput per sample.
        samples = result if isinstance(result, list) else [result]
        return [self._map_sample(cast(Any, sample), return_pae) for sample in samples]

    @staticmethod
    def _map_sample(sample: Any, return_pae: bool) -> FoldOutput:
        """Map one ESMFold2 result to a FoldOutput, guarding optional scores."""
        plddt: list[float] = []
        if sample.plddt is not None:
            plddt = [
                float(value)
                for value in sample.plddt.float().cpu().reshape(-1).tolist()
            ]
        ptm = 0.0 if sample.ptm is None else float(sample.ptm)
        iptm = None if sample.iptm is None else float(sample.iptm)
        pae: list[list[float]] | None = None
        if return_pae:
            raw_pae = getattr(sample, 'pae', None)
            if raw_pae is not None:
                pae = [
                    [float(value) for value in row]
                    for row in raw_pae.float().cpu().tolist()
                ]

        return FoldOutput(
            mmcif=sample.complex.to_mmcif(),
            plddt=plddt,
            ptm=ptm,
            iptm=iptm,
            pae=pae,
        )


def create_app(
    settings: Settings | None = None,
    backend: FoldBackend | None = None,
) -> FastAPI:
    """Create the application, optionally injecting a backend for tests."""
    app_settings = settings or Settings()
    fold_backend = backend or EsmFold2Backend(app_settings)
    inference_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
        if app_settings.preload_model is not None:
            await asyncio.to_thread(fold_backend.load, app_settings.preload_model)
        try:
            yield
        finally:
            fold_backend.unload()

    app = FastAPI(
        title='ESMFold2 Structure Prediction API',
        version=__version__,
        lifespan=lifespan,
    )

    @app.get('/health', tags=['health'])
    async def health() -> dict[str, str]:
        """Report process health without forcing a model download."""
        return {'status': 'healthy'}

    @app.post('/v1/fold', response_model=FoldResponse, tags=['fold'])
    async def fold(request: FoldRequest) -> FoldResponse:
        """Fold the requested chains into a single complex structure."""
        chains = request.resolved_chains()
        try:
            # Keep the event loop responsive while serializing access to the
            # resident model and accelerator within this server process.
            async with inference_lock:
                outputs = await asyncio.to_thread(
                    fold_backend.fold,
                    request.model,
                    chains,
                    num_loops=request.num_loops,
                    num_sampling_steps=request.num_sampling_steps,
                    num_diffusion_samples=request.num_diffusion_samples,
                    seed=request.seed,
                    return_pae=request.return_pae,
                )
        except (RuntimeError, ValueError) as error:
            LOGGER.exception('Fold request failed')
            raise HTTPException(status_code=500, detail=str(error)) from error

        samples = [
            FoldSample(
                structure=output.mmcif,
                plddt_mean=(
                    sum(output.plddt) / len(output.plddt) if output.plddt else 0.0
                ),
                plddt=output.plddt,
                ptm=output.ptm,
                iptm=output.iptm,
                pae=output.pae,
                ranking_score=ranking_score(output.ptm, output.iptm),
            )
            for output in outputs
        ]
        # Best structure first; the request may draw several diffusion samples.
        samples.sort(key=lambda sample: sample.ranking_score, reverse=True)

        return FoldResponse(
            model=request.model,
            num_chains=len(chains),
            chain_ids=[chain.id for chain in chains],
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
        'esmfold2_service.server:app',
        host=settings.host,
        port=settings.port,
    )
