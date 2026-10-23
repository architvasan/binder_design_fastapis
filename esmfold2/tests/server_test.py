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
    ) -> FoldOutput:
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
        total = sum(len(chain.sequence) for chain in chains)
        return FoldOutput(
            mmcif='CIF',
            plddt=[float(index) for index in range(total)],
            ptm=0.5,
            iptm=0.8 if len(chains) > 1 else None,
            pae=[[0.0, 1.0]] if return_pae else None,
        )


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
    assert body['format'] == 'mmcif'
    assert body['structure'] == 'CIF'
    assert body['num_chains'] == 1
    assert body['chain_ids'] == ['A']
    assert body['plddt'] == [0.0, 1.0, 2.0]
    assert body['plddt_mean'] == 1.0
    assert body['ptm'] == 0.5
    assert body['iptm'] is None
    assert body['pae'] is None
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
    assert body['iptm'] == 0.8
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
    assert response.json()['pae'] == [[0.0, 1.0]]


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

        def fold(self, model: object, spi: Any, **kwargs: object) -> _Result:
            type(self).last_spi = spi
            type(self).last_kwargs = kwargs
            total = sum(len(chain.sequence) for chain in spi.sequences)
            plddt = torch.arange(total, dtype=torch.float32)
            pae = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
            return _Result(plddt, 0.5, 0.8, _Complex('data_pred\n'), pae)

    module = types.ModuleType('esm.models.esmfold2')
    module.EsmFold2Model = EsmFold2Model  # type: ignore[attr-defined]
    module.ProteinInput = ProteinInput  # type: ignore[attr-defined]
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
    output = backend.fold(
        'esmfold2',
        chains,
        num_loops=7,
        num_sampling_steps=9,
        num_diffusion_samples=1,
        seed=3,
        return_pae=True,
    )

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
    output = backend.fold(
        'esmfold2',
        [ChainSpec(id='A', sequence='MK')],
        num_loops=20,
        num_sampling_steps=100,
        num_diffusion_samples=1,
        seed=0,
        return_pae=False,
    )

    assert output.pae is None
