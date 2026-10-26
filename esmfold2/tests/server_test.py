"""API tests for the ESMFold2 structure-prediction service."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.status import HTTP_200_OK
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT

from esmfold2_service import __version__
from esmfold2_service.server import ChainSpec
from esmfold2_service.server import create_app
from esmfold2_service.server import EsmFold2Backend
from esmfold2_service.server import FoldOutput
from esmfold2_service.server import Settings


class FakeBackend:
    """Small deterministic backend that avoids downloading a model."""

    def __init__(self) -> None:
        self.loaded_models: list[str] = []
        self.unload_calls = 0
        self.fold_calls: list[dict[str, Any]] = []

    def load(self, model_id: str) -> None:
        self.loaded_models.append(model_id)

    def unload(self) -> None:
        self.unload_calls += 1

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
        self.fold_calls.append(
            {
                'model': model,
                'chain_ids': [chain.id for chain in chains],
                'num_loops': num_loops,
                'num_sampling_steps': num_sampling_steps,
                'num_diffusion_samples': num_diffusion_samples,
                'seed': seed,
                'return_pae': return_pae,
            },
        )
        total = sum(len(chain.sequence or '') for chain in chains)
        return [
            FoldOutput(
                mmcif='CIF',
                plddt=[float(index) for index in range(total)],
                ptm=0.5,
                iptm=0.8 if len(chains) > 1 else None,
                pae=[[0.0, 1.0]] if return_pae else None,
            ),
        ]


@pytest.fixture
def client() -> TestClient:
    """Return an API client backed by deterministic folds."""
    return TestClient(create_app(Settings(), FakeBackend()))


def test_single_chain_fold(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={'model': 'esmfold2', 'sequences': [{'sequence': 'MKT'}]},
    )

    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body['object'] == 'fold'
    assert body['num_chains'] == 1
    assert body['chain_ids'] == ['A']
    assert body['num_samples'] == 1
    assert len(body['results']) == 1
    sample = body['results'][0]
    assert sample['format'] == 'mmcif'
    assert sample['structure'] == 'CIF'
    assert sample['plddt'] == [0.0, 1.0, 2.0]
    assert sample['plddt_mean'] == 1.0
    assert sample['ptm'] == 0.5
    assert sample['iptm'] is None
    assert sample['pae'] is None
    assert sample['ranking_score'] == 0.5
    assert body['usage'] == {'num_chains': 1, 'total_residues': 3}


def test_multichain_assigns_sequential_ids() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/fold',
        json={
            'model': 'esmfold2',
            'sequences': [{'sequence': 'MK'}, {'sequence': 'AAG'}],
        },
    )

    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body['chain_ids'] == ['A', 'B']
    assert body['num_chains'] == 2
    assert body['results'][0]['iptm'] == 0.8
    # Complex ranking score: 0.8*iptm + 0.2*ptm = 0.8*0.8 + 0.2*0.5 = 0.74.
    assert body['results'][0]['ranking_score'] == pytest.approx(0.74)
    assert body['usage'] == {'num_chains': 2, 'total_residues': 5}
    assert backend.fold_calls[0]['chain_ids'] == ['A', 'B']


def test_explicit_chain_ids_are_preserved(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={
            'model': 'esmfold2',
            'sequences': [
                {'id': 'H', 'sequence': 'MK'},
                {'id': 'L', 'sequence': 'AA'},
            ],
        },
    )

    assert response.status_code == HTTP_200_OK
    assert response.json()['chain_ids'] == ['H', 'L']


def test_mixed_ids_fill_around_explicit_ones(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={
            'model': 'esmfold2',
            'sequences': [
                {'id': 'B', 'sequence': 'MK'},
                {'sequence': 'AA'},
                {'sequence': 'GG'},
            ],
        },
    )

    assert response.status_code == HTTP_200_OK
    # 'B' is taken, so the unnamed chains get 'A' then 'C'.
    assert response.json()['chain_ids'] == ['B', 'A', 'C']


def test_fold_parameters_are_forwarded() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/fold',
        json={
            'model': 'esmfold2-fast',
            'sequences': [{'sequence': 'MK'}],
            'num_loops': 5,
            'num_sampling_steps': 50,
            'num_diffusion_samples': 2,
            'seed': 7,
        },
    )

    assert response.status_code == HTTP_200_OK
    call = backend.fold_calls[0]
    assert call['model'] == 'esmfold2-fast'
    assert call['num_loops'] == 5
    assert call['num_sampling_steps'] == 50
    assert call['num_diffusion_samples'] == 2
    assert call['seed'] == 7


def test_return_pae_includes_matrix(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={
            'model': 'esmfold2',
            'sequences': [{'sequence': 'MK'}],
            'return_pae': True,
        },
    )

    assert response.status_code == HTTP_200_OK
    assert response.json()['results'][0]['pae'] == [[0.0, 1.0]]


class _MultiSampleBackend:
    """Backend returning several samples out of rank order (monomer scores)."""

    def load(self, model_id: str) -> None:
        return None

    def unload(self) -> None:
        return None

    def fold(self, model: str, chains: list[ChainSpec], **_: Any) -> list[FoldOutput]:
        return [
            FoldOutput(mmcif='c1', plddt=[10.0], ptm=0.30, iptm=None, pae=None),
            FoldOutput(mmcif='c2', plddt=[90.0], ptm=0.90, iptm=None, pae=None),
            FoldOutput(mmcif='c3', plddt=[60.0], ptm=0.60, iptm=None, pae=None),
        ]


def test_results_are_ranked_best_first() -> None:
    client = TestClient(create_app(Settings(), _MultiSampleBackend()))

    response = client.post(
        '/v1/fold',
        json={
            'model': 'esmfold2',
            'sequences': [{'sequence': 'MKT'}],
            'num_diffusion_samples': 3,
        },
    )

    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body['num_samples'] == 3
    # Monomer, so ranking_score == ptm; best structure first.
    assert [sample['structure'] for sample in body['results']] == ['c2', 'c3', 'c1']
    scores = [sample['ranking_score'] for sample in body['results']]
    assert scores == sorted(scores, reverse=True)
    assert scores == [0.9, 0.6, 0.3]


def test_dna_and_rna_chains_accepted() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/fold',
        json={
            'model': 'esmfold2',
            'sequences': [
                {'type': 'protein', 'sequence': 'MKT'},
                {'type': 'dna', 'sequence': 'ACGT'},
                {'type': 'rna', 'sequence': 'ACGU'},
            ],
        },
    )

    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body['chain_ids'] == ['A', 'B', 'C']
    # All three polymer chains count toward total_residues: 3 + 4 + 4.
    assert body['usage']['total_residues'] == 11


def test_ligand_chain_accepted_with_smiles() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/fold',
        json={
            'model': 'esmfold2',
            'sequences': [
                {'type': 'protein', 'sequence': 'MKT'},
                {'type': 'ligand', 'smiles': 'CC(=O)Oc1ccccc1C(=O)O'},
            ],
        },
    )

    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body['num_chains'] == 2
    # The ligand has no sequence, so only the protein's 3 residues are counted.
    assert body['usage']['total_residues'] == 3


def test_ligand_chain_accepted_with_ccd() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/fold',
        json={
            'model': 'esmfold2',
            'sequences': [
                {'type': 'protein', 'sequence': 'MKT'},
                {'type': 'ligand', 'ccd': ['HEM']},
            ],
        },
    )

    assert response.status_code == HTTP_200_OK


def test_backend_maps_each_chain_type_to_its_sdk_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    module = _install_fake_esm(monkeypatch)

    backend = EsmFold2Backend(Settings())
    chains = [
        ChainSpec(id='A', type='protein', sequence='MKT'),
        ChainSpec(id='B', type='dna', sequence='ACGT'),
        ChainSpec(id='C', type='rna', sequence='ACGU'),
        ChainSpec(id='D', type='ligand', smiles='CCO'),
        ChainSpec(id='E', type='ligand', ccd=('HEM',)),
    ]
    backend.fold(
        'esmfold2',
        chains,
        num_loops=20,
        num_sampling_steps=100,
        num_diffusion_samples=1,
        seed=0,
        return_pae=False,
    )

    built = module.ESMFold2InputBuilder.last_spi.sequences
    assert [type(item).__name__ for item in built] == [
        'ProteinInput',
        'DNAInput',
        'RNAInput',
        'LigandInput',
        'LigandInput',
    ]
    assert built[3].smiles == 'CCO'
    assert built[3].ccd is None
    assert built[4].ccd == ['HEM']
    assert built[4].smiles is None


@pytest.mark.parametrize(
    'chain',
    (
        {'type': 'ligand', 'sequence': 'MK'},  # ligand must not have a sequence
        {'type': 'ligand'},  # neither smiles nor ccd
        {'type': 'ligand', 'smiles': 'CCO', 'ccd': ['HEM']},  # both
        {'type': 'protein', 'sequence': 'MK', 'smiles': 'CCO'},  # polymer w/ ligand
        {'type': 'dna', 'sequence': 'ACGU'},  # U is not a DNA base
        {'type': 'rna', 'sequence': 'ACGT'},  # T is not an RNA base
        {'type': 'protein'},  # polymer with no sequence
    ),
)
def test_invalid_chain_shapes_are_rejected(
    client: TestClient,
    chain: dict[str, Any],
) -> None:
    response = client.post(
        '/v1/fold',
        json={'model': 'esmfold2', 'sequences': [chain]},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_ligand_only_fold_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={
            'model': 'esmfold2',
            'sequences': [{'type': 'ligand', 'smiles': 'CCO'}],
        },
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_lowercase_sequence_is_uppercased() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/fold',
        json={'model': 'esmfold2', 'sequences': [{'sequence': 'mkt'}]},
    )

    assert response.status_code == HTTP_200_OK
    # 3 residues -> total_residues 3 confirms the sequence survived intact.
    assert response.json()['usage']['total_residues'] == 3


def test_duplicate_chain_ids_are_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={
            'model': 'esmfold2',
            'sequences': [
                {'id': 'A', 'sequence': 'MK'},
                {'id': 'A', 'sequence': 'AA'},
            ],
        },
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_empty_sequences_list_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={'model': 'esmfold2', 'sequences': []},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.parametrize('sequence', ('', '   ', 'MK1', 'MK*', 'MJ@'))
def test_invalid_sequences_are_rejected(
    client: TestClient,
    sequence: str,
) -> None:
    response = client.post(
        '/v1/fold',
        json={'model': 'esmfold2', 'sequences': [{'sequence': sequence}]},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_unknown_model_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={'model': 'alphafold', 'sequences': [{'sequence': 'MK'}]},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_unknown_request_field_remains_forbidden(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={'model': 'esmfold2', 'sequences': [{'sequence': 'MK'}], 'x': 1},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_out_of_range_parameter_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={
            'model': 'esmfold2',
            'sequences': [{'sequence': 'MK'}],
            'num_loops': 0,
        },
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_openapi_request_schema(client: TestClient) -> None:
    schema = client.get('/openapi.json').json()
    properties = schema['components']['schemas']['FoldRequest']['properties']

    assert schema['info']['version'] == __version__
    assert properties['sequences']['type'] == 'array'
    assert 'num_loops' in properties
    assert 'return_pae' in properties


def test_health(client: TestClient) -> None:
    assert client.get('/health').json() == {'status': 'healthy'}


def test_lifespan_uses_backend_lifecycle() -> None:
    backend = FakeBackend()

    with TestClient(create_app(Settings(preload_model='esmfold2'), backend)):
        assert backend.loaded_models == ['esmfold2']
        assert backend.unload_calls == 0

    assert backend.unload_calls == 1


def test_preload_model_must_be_supported() -> None:
    with pytest.raises(ValidationError, match='preload_model must be one of'):
        Settings(preload_model='unsupported/model')


def _install_fake_esm(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, str | None]] | None = None,
) -> types.ModuleType:
    """Install a stub ``esm.models.esmfold2`` so the real backend can run."""

    class EsmFold2Model:
        @classmethod
        def from_pretrained(
            cls,
            model_id: str,
            device: str | None = None,
        ) -> torch.nn.Module:
            if calls is not None:
                calls.append((model_id, device))
            return torch.nn.Linear(2, 2)

    class ProteinInput:
        def __init__(self, id: str, sequence: str) -> None:
            self.id = id
            self.sequence = sequence

    class DNAInput:
        def __init__(self, id: str, sequence: str) -> None:
            self.id = id
            self.sequence = sequence

    class RNAInput:
        def __init__(self, id: str, sequence: str) -> None:
            self.id = id
            self.sequence = sequence

    class LigandInput:
        def __init__(
            self,
            id: str,
            smiles: str | None = None,
            ccd: list[str] | None = None,
        ) -> None:
            self.id = id
            self.smiles = smiles
            self.ccd = ccd

    class StructurePredictionInput:
        def __init__(self, sequences: list[Any]) -> None:
            self.sequences = sequences

    class _Complex:
        def __init__(self, mmcif: str) -> None:
            self._mmcif = mmcif

        def to_mmcif(self) -> str:
            return self._mmcif

    class _Result:
        def __init__(
            self,
            plddt: torch.Tensor,
            ptm: float,
            iptm: float | None,
            complex_: _Complex,
            pae: torch.Tensor,
        ) -> None:
            self.plddt = plddt
            self.ptm = ptm
            self.iptm = iptm
            self.complex = complex_
            self.pae = pae

    class ESMFold2InputBuilder:
        last_spi: Any = None
        last_kwargs: Any = None

        def fold(self, model: object, spi: Any, **kwargs: object) -> Any:
            type(self).last_spi = spi
            type(self).last_kwargs = kwargs
            total = sum(
                len(getattr(chain, 'sequence', '') or '') for chain in spi.sequences
            )
            plddt = torch.arange(total, dtype=torch.float32)
            pae = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
            samples = int(kwargs.get('num_diffusion_samples', 1))  # type: ignore[call-overload]
            # Mirror the SDK: one result for a single sample, a list otherwise.
            if samples == 1:
                return _Result(plddt, 0.5, 0.8, _Complex('data_pred\n'), pae)
            return [
                _Result(plddt, 0.5 + 0.1 * i, 0.8, _Complex(f'data_pred_{i}\n'), pae)
                for i in range(samples)
            ]

    module = types.ModuleType('esm.models.esmfold2')
    module.EsmFold2Model = EsmFold2Model  # type: ignore[attr-defined]
    module.ProteinInput = ProteinInput  # type: ignore[attr-defined]
    module.DNAInput = DNAInput  # type: ignore[attr-defined]
    module.RNAInput = RNAInput  # type: ignore[attr-defined]
    module.LigandInput = LigandInput  # type: ignore[attr-defined]
    module.StructurePredictionInput = StructurePredictionInput  # type: ignore[attr-defined]
    module.ESMFold2InputBuilder = ESMFold2InputBuilder  # type: ignore[attr-defined]

    esm_pkg = types.ModuleType('esm')
    models_pkg = types.ModuleType('esm.models')
    monkeypatch.setitem(sys.modules, 'esm', esm_pkg)
    monkeypatch.setitem(sys.modules, 'esm.models', models_pkg)
    monkeypatch.setitem(sys.modules, 'esm.models.esmfold2', module)
    return module


def test_backend_loads_and_caches_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []
    _install_fake_esm(monkeypatch, calls)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    backend = EsmFold2Backend(Settings())
    loaded = backend.load('esmfold2')
    cached = backend.load('esmfold2')

    assert cached is loaded
    assert loaded.device == torch.device('cpu')
    assert not loaded.model.training
    # The HF repo id is what gets passed to from_pretrained, on cpu.
    assert calls == [('biohub/ESMFold2', 'cpu')]

    backend.unload()
    assert backend.loaded is None


def test_backend_swaps_resident_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []
    _install_fake_esm(monkeypatch, calls)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    backend = EsmFold2Backend(Settings())
    backend.load('esmfold2')
    backend.load('esmfold2-fast')

    assert [repo for repo, _ in calls] == ['biohub/ESMFold2', 'biohub/esmfold2-fast']
    assert backend.loaded is not None
    assert backend.loaded.model_id == 'esmfold2-fast'


def test_fold_builds_complex_and_maps_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    module = _install_fake_esm(monkeypatch)

    backend = EsmFold2Backend(Settings())
    chains = [ChainSpec(id='A', sequence='MKT'), ChainSpec(id='B', sequence='AA')]
    outputs = backend.fold(
        'esmfold2',
        chains,
        num_loops=7,
        num_sampling_steps=9,
        num_diffusion_samples=1,
        seed=3,
        return_pae=True,
    )

    assert len(outputs) == 1
    output = outputs[0]
    # 5 residues total -> plddt = [0, 1, 2, 3, 4].
    assert output.plddt == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert output.ptm == 0.5
    assert output.iptm == 0.8
    assert output.mmcif == 'data_pred\n'
    assert output.pae == [[0.0, 1.0], [2.0, 3.0]]

    builder = module.ESMFold2InputBuilder
    assert builder.last_kwargs == {
        'num_loops': 7,
        'num_sampling_steps': 9,
        'num_diffusion_samples': 1,
        'seed': 3,
    }
    forwarded = [(c.id, c.sequence) for c in builder.last_spi.sequences]
    assert forwarded == [('A', 'MKT'), ('B', 'AA')]


def test_fold_omits_pae_when_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    _install_fake_esm(monkeypatch)

    backend = EsmFold2Backend(Settings())
    outputs = backend.fold(
        'esmfold2',
        [ChainSpec(id='A', sequence='MK')],
        num_loops=20,
        num_sampling_steps=100,
        num_diffusion_samples=1,
        seed=0,
        return_pae=False,
    )

    assert outputs[0].pae is None


def test_fold_returns_one_output_per_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    _install_fake_esm(monkeypatch)

    backend = EsmFold2Backend(Settings())
    # num_diffusion_samples > 1 -> the SDK returns a list; the backend must map
    # every sample, not assume a single result.
    outputs = backend.fold(
        'esmfold2',
        [ChainSpec(id='A', sequence='MK')],
        num_loops=20,
        num_sampling_steps=100,
        num_diffusion_samples=3,
        seed=0,
        return_pae=False,
    )

    assert len(outputs) == 3
    assert [output.ptm for output in outputs] == [0.5, 0.6, pytest.approx(0.7)]
    assert [output.mmcif for output in outputs] == [
        'data_pred_0\n',
        'data_pred_1\n',
        'data_pred_2\n',
    ]
