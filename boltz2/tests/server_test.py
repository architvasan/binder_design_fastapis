"""API tests for the Boltz-2 co-folding service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from starlette.status import HTTP_200_OK
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT

from boltz2_service import __version__
from boltz2_service.server import BoltzFoldBackend
from boltz2_service.server import build_yaml
from boltz2_service.server import ChainSpec
from boltz2_service.server import create_app
from boltz2_service.server import FoldOutput
from boltz2_service.server import FoldResult
from boltz2_service.server import Settings


class FakeBackend:
    """Deterministic backend that avoids running the boltz executable."""

    def __init__(self) -> None:
        self.load_calls = 0
        self.unload_calls = 0
        self.fold_calls: list[dict[str, Any]] = []

    def load(self) -> None:
        self.load_calls += 1

    def unload(self) -> None:
        self.unload_calls += 1

    def fold(
        self,
        chains: list[ChainSpec],
        *,
        use_msa_server: bool,
        diffusion_samples: int,
        recycling_steps: int,
        affinity_binder: str | None,
    ) -> FoldResult:
        self.fold_calls.append(
            {
                'chain_ids': [chain.id for chain in chains],
                'entity_types': [chain.entity_type for chain in chains],
                'use_msa_server': use_msa_server,
                'diffusion_samples': diffusion_samples,
                'recycling_steps': recycling_steps,
                'affinity_binder': affinity_binder,
            },
        )
        samples = [
            FoldOutput(
                mmcif='data_pred\n',
                confidence_score=0.85,
                ptm=0.80,
                iptm=0.70 if len(chains) > 1 else 0.0,
                complex_plddt=0.88,
            ),
        ]
        affinity = {'affinity_pred_value': 1.2} if affinity_binder else None
        return FoldResult(samples=samples, affinity=affinity)


@pytest.fixture
def client() -> TestClient:
    """Return an API client backed by a deterministic fold."""
    return TestClient(create_app(Settings(), FakeBackend()))


def test_single_protein_fold(client: TestClient) -> None:
    response = client.post('/v1/fold', json={'chains': [{'sequence': 'MKT'}]})

    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body['object'] == 'fold'
    assert body['model'] == 'boltz-2'
    assert body['chain_ids'] == ['A']
    assert body['num_samples'] == 1
    sample = body['results'][0]
    assert sample['format'] == 'mmcif'
    assert sample['structure'] == 'data_pred\n'
    assert sample['scores'] == {
        'confidence_score': 0.85,
        'ptm': 0.80,
        'iptm': 0.0,
        'complex_plddt': 0.88,
    }
    assert body['affinity'] is None
    assert body['usage'] == {'num_chains': 1, 'total_residues': 3}


def test_multichain_assigns_sequential_ids() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/fold',
        json={'chains': [{'sequence': 'MK'}, {'sequence': 'AAG'}]},
    )

    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body['chain_ids'] == ['A', 'B']
    assert body['results'][0]['scores']['iptm'] == 0.70
    assert body['usage'] == {'num_chains': 2, 'total_residues': 5}


def test_explicit_ids_are_preserved(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={
            'chains': [
                {'id': 'H', 'sequence': 'MK'},
                {'id': 'L', 'sequence': 'AA'},
            ],
        },
    )

    assert response.status_code == HTTP_200_OK
    assert response.json()['chain_ids'] == ['H', 'L']


def test_protein_plus_ligand_smiles() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/fold',
        json={
            'chains': [
                {'entity_type': 'protein', 'sequence': 'MKT'},
                {'entity_type': 'ligand', 'smiles': 'CCO'},
            ],
        },
    )

    assert response.status_code == HTTP_200_OK
    assert response.json()['usage']['total_residues'] == 3
    assert backend.fold_calls[0]['entity_types'] == ['protein', 'ligand']


def test_protein_plus_ligand_ccd(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={
            'chains': [
                {'entity_type': 'protein', 'sequence': 'MKT'},
                {'entity_type': 'ligand', 'ccd': 'HEM'},
            ],
        },
    )
    assert response.status_code == HTTP_200_OK


def test_protein_plus_nucleic_acid(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={
            'chains': [
                {'entity_type': 'protein', 'sequence': 'MKT'},
                {'entity_type': 'dna', 'sequence': 'ACGT'},
                {'entity_type': 'rna', 'sequence': 'ACGU'},
            ],
        },
    )
    assert response.status_code == HTTP_200_OK
    assert response.json()['usage']['total_residues'] == 11


def test_affinity_binder_is_returned() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/fold',
        json={
            'chains': [
                {'id': 'A', 'entity_type': 'protein', 'sequence': 'MKT'},
                {'id': 'L', 'entity_type': 'ligand', 'smiles': 'CCO'},
            ],
            'affinity_binder': 'L',
        },
    )

    assert response.status_code == HTTP_200_OK
    assert response.json()['affinity'] == {'affinity_pred_value': 1.2}
    assert backend.fold_calls[0]['affinity_binder'] == 'L'


def test_request_overrides_msa_default() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(use_msa_server=False), backend))

    response = client.post(
        '/v1/fold',
        json={
            'chains': [{'sequence': 'MK'}],
            'use_msa_server': True,
            'diffusion_samples': 3,
            'recycling_steps': 5,
        },
    )

    assert response.status_code == HTTP_200_OK
    call = backend.fold_calls[0]
    assert call['use_msa_server'] is True
    assert call['diffusion_samples'] == 3
    assert call['recycling_steps'] == 5


def test_lowercase_sequence_is_uppercased() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post('/v1/fold', json={'chains': [{'sequence': 'mkt'}]})

    assert response.status_code == HTTP_200_OK
    assert response.json()['usage']['total_residues'] == 3


def test_ligand_only_fold_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={'chains': [{'entity_type': 'ligand', 'smiles': 'CCO'}]},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_affinity_binder_must_be_a_ligand(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={
            'chains': [{'id': 'A', 'sequence': 'MKT'}],
            'affinity_binder': 'A',
        },
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_affinity_binder_must_exist(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={
            'chains': [{'id': 'A', 'sequence': 'MKT'}],
            'affinity_binder': 'Z',
        },
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_duplicate_chain_ids_are_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={
            'chains': [
                {'id': 'A', 'sequence': 'MK'},
                {'id': 'A', 'sequence': 'AA'},
            ],
        },
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.parametrize(
    'chain',
    (
        {'sequence': ''},  # empty
        {'sequence': 'MK1'},  # non-AA
        {'entity_type': 'dna', 'sequence': 'ACGU'},  # U is not a DNA base
        {'entity_type': 'rna', 'sequence': 'ACGT'},  # T is not an RNA base
        {'entity_type': 'ligand', 'sequence': 'MK'},  # ligand with a sequence
        {'entity_type': 'ligand'},  # ligand with neither smiles nor ccd
        {'entity_type': 'ligand', 'smiles': 'CCO', 'ccd': 'HEM'},  # both
        {'entity_type': 'protein', 'sequence': 'MK', 'ccd': 'HEM'},  # polymer+lig
        {'entity_type': 'protein'},  # polymer with no sequence
        {'id': 'bad id', 'sequence': 'MK'},  # id has whitespace
    ),
)
def test_invalid_chain_shapes_are_rejected(
    client: TestClient,
    chain: dict[str, Any],
) -> None:
    response = client.post('/v1/fold', json={'chains': [chain]})
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_out_of_range_parameter_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={'chains': [{'sequence': 'MK'}], 'diffusion_samples': 99},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_unknown_request_field_remains_forbidden(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={'chains': [{'sequence': 'MK'}], 'x': 1},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_openapi_request_schema(client: TestClient) -> None:
    schema = client.get('/openapi.json').json()
    properties = schema['components']['schemas']['FoldRequest']['properties']

    assert schema['info']['version'] == __version__
    assert properties['chains']['type'] == 'array'
    assert 'diffusion_samples' in properties


def test_health(client: TestClient) -> None:
    assert client.get('/health').json() == {'status': 'healthy'}


def test_lifespan_uses_backend_lifecycle() -> None:
    backend = FakeBackend()

    with TestClient(create_app(Settings(), backend)):
        assert backend.load_calls == 1
        assert backend.unload_calls == 0

    assert backend.unload_calls == 1


# --- YAML rendering -----------------------------------------------------------


def test_build_yaml_protein_ligand_and_affinity() -> None:
    chains = [
        ChainSpec(id='A', entity_type='protein', sequence='MKT'),
        ChainSpec(id='L', entity_type='ligand', smiles='CC(=O)O'),
    ]
    yaml = build_yaml(chains, affinity_binder='L')
    assert yaml == (
        'version: 1\n'
        'sequences:\n'
        '  - protein:\n'
        "      id: A\n"
        "      sequence: 'MKT'\n"
        '  - ligand:\n'
        '      id: L\n'
        "      smiles: 'CC(=O)O'\n"
        'properties:\n'
        '  - affinity:\n'
        '      binder: L\n'
    )


def test_build_yaml_ccd_ligand_no_affinity() -> None:
    chains = [
        ChainSpec(id='A', entity_type='protein', sequence='MK'),
        ChainSpec(id='B', entity_type='ligand', ccd='HEM'),
    ]
    yaml = build_yaml(chains, affinity_binder=None)
    assert "ccd: 'HEM'" in yaml
    assert 'properties:' not in yaml


# --- backend tests against a faked subprocess ---------------------------------


def _fake_run_factory(
    captured: list[list[str]],
    *,
    samples: int,
    with_affinity: bool = False,
) -> Any:
    """Return a stand-in for ``_run`` that fabricates boltz output files."""

    def fake_run(cmd: list[str]) -> None:
        captured.append(cmd)
        out_dir = Path(cmd[cmd.index('--out_dir') + 1])
        pred = out_dir / 'predictions' / 'input'
        pred.mkdir(parents=True, exist_ok=True)
        for index in range(samples):
            (pred / f'input_model_{index}.cif').write_text(f'data_pred_{index}\n')
            # confidence rises with index, so ranking must reverse the file order.
            (pred / f'confidence_input_model_{index}.json').write_text(
                json.dumps(
                    {
                        'confidence_score': 0.1 * index,
                        'ptm': 0.5,
                        'iptm': 0.6,
                        'complex_plddt': 0.7,
                    },
                ),
            )
        if with_affinity:
            (pred / 'affinity_input.json').write_text(
                json.dumps({'affinity_pred_value': 2.5}),
            )

    return fake_run


def test_backend_builds_command_and_ranks_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    backend = BoltzFoldBackend(Settings(executable='boltz'))
    captured: list[list[str]] = []
    monkeypatch.setattr(
        backend,
        '_run',
        _fake_run_factory(captured, samples=3),
    )

    result = backend.fold(
        [
            ChainSpec(id='A', entity_type='protein', sequence='MKT'),
            ChainSpec(id='L', entity_type='ligand', smiles='CCO'),
        ],
        use_msa_server=True,
        diffusion_samples=3,
        recycling_steps=5,
        affinity_binder=None,
    )

    cmd = captured[0]
    assert cmd[:2] == ['boltz', 'predict']
    assert '--use_msa_server' in cmd
    assert cmd[cmd.index('--diffusion_samples') + 1] == '3'
    assert cmd[cmd.index('--recycling_steps') + 1] == '5'
    assert cmd[cmd.index('--output_format') + 1] == 'mmcif'

    # Three samples, ranked best (highest confidence_score) first: 2, 1, 0.
    assert [s.mmcif for s in result.samples] == [
        'data_pred_2\n',
        'data_pred_1\n',
        'data_pred_0\n',
    ]
    assert [round(s.confidence_score, 4) for s in result.samples] == [0.2, 0.1, 0.0]
    assert result.samples[0].ptm == 0.5
    assert result.affinity is None


def test_backend_returns_affinity_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    backend = BoltzFoldBackend(Settings())
    captured: list[list[str]] = []
    monkeypatch.setattr(
        backend,
        '_run',
        _fake_run_factory(captured, samples=1, with_affinity=True),
    )

    result = backend.fold(
        [
            ChainSpec(id='A', entity_type='protein', sequence='MK'),
            ChainSpec(id='L', entity_type='ligand', smiles='CCO'),
        ],
        use_msa_server=False,
        diffusion_samples=1,
        recycling_steps=3,
        affinity_binder='L',
    )

    assert result.affinity == {'affinity_pred_value': 2.5}
    assert '--use_msa_server' not in captured[0]


def test_backend_load_rejects_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr('boltz2_service.server.shutil.which', lambda _exe: None)
    backend = BoltzFoldBackend(Settings(executable='definitely-not-here'))
    with pytest.raises(RuntimeError, match='boltz executable not found'):
        backend.load()


def test_backend_load_accepts_resolvable_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'boltz2_service.server.shutil.which',
        lambda _exe: '/usr/bin/boltz',
    )
    backend = BoltzFoldBackend(Settings(executable='boltz'))
    backend.load()  # does not raise
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    backend.unload()
