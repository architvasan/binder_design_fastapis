"""API tests for the Chai-1 co-folding service."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from starlette.status import HTTP_200_OK
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT

from chai1_service import __version__
from chai1_service.server import ChaiFoldBackend
from chai1_service.server import ChainSpec
from chai1_service.server import create_app
from chai1_service.server import FoldOutput
from chai1_service.server import Settings


class FakeBackend:
    """Deterministic backend that avoids downloading weights or a GPU."""

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
        use_esm_embeddings: bool,
        use_msa_server: bool,
        num_trunk_recycles: int,
        num_diffn_timesteps: int,
        num_diffn_samples: int,
        seed: int | None,
        return_pae: bool,
    ) -> list[FoldOutput]:
        self.fold_calls.append(
            {
                'chain_names': [chain.name for chain in chains],
                'entity_types': [chain.entity_type for chain in chains],
                'use_esm_embeddings': use_esm_embeddings,
                'use_msa_server': use_msa_server,
                'num_trunk_recycles': num_trunk_recycles,
                'num_diffn_timesteps': num_diffn_timesteps,
                'num_diffn_samples': num_diffn_samples,
                'seed': seed,
                'return_pae': return_pae,
            },
        )
        return [
            FoldOutput(
                mmcif='data_pred\n',
                aggregate_score=0.70,
                complex_ptm=0.50,
                interface_ptm=0.80 if len(chains) > 1 else 0.0,
                has_inter_chain_clashes=False,
                pae=[[0.0, 1.0]] if return_pae else None,
            ),
        ]


@pytest.fixture
def client() -> TestClient:
    """Return an API client backed by a deterministic fold."""
    return TestClient(create_app(Settings(), FakeBackend()))


def test_single_protein_fold(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={'chains': [{'sequence': 'MKT'}]},
    )

    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body['object'] == 'fold'
    assert body['model'] == 'chai-1'
    assert body['num_chains'] == 1
    assert body['chain_names'] == ['A']
    assert body['num_samples'] == 1
    sample = body['results'][0]
    assert sample['format'] == 'mmcif'
    assert sample['structure'] == 'data_pred\n'
    assert sample['scores']['aggregate_score'] == 0.70
    assert sample['scores']['complex_ptm'] == 0.50
    assert sample['scores']['has_inter_chain_clashes'] is False
    assert sample['pae'] is None
    assert body['usage'] == {'num_chains': 1, 'total_residues': 3}


def test_multichain_assigns_sequential_names() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/fold',
        json={'chains': [{'sequence': 'MK'}, {'sequence': 'AAG'}]},
    )

    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body['chain_names'] == ['A', 'B']
    assert body['results'][0]['scores']['interface_ptm'] == 0.80
    assert body['usage'] == {'num_chains': 2, 'total_residues': 5}
    assert backend.fold_calls[0]['chain_names'] == ['A', 'B']


def test_explicit_names_are_preserved(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={
            'chains': [
                {'name': 'target', 'sequence': 'MK'},
                {'name': 'binder', 'sequence': 'AA'},
            ],
        },
    )

    assert response.status_code == HTTP_200_OK
    assert response.json()['chain_names'] == ['target', 'binder']


def test_mixed_names_fill_around_explicit_ones(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={
            'chains': [
                {'name': 'B', 'sequence': 'MK'},
                {'sequence': 'AA'},
                {'sequence': 'GG'},
            ],
        },
    )

    assert response.status_code == HTTP_200_OK
    # 'B' is taken, so the unnamed chains get 'A' then 'C'.
    assert response.json()['chain_names'] == ['B', 'A', 'C']


def test_protein_plus_ligand_accepted() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/fold',
        json={
            'chains': [
                {'entity_type': 'protein', 'sequence': 'MKT'},
                {'entity_type': 'ligand', 'smiles': 'CC(=O)Oc1ccccc1C(=O)O'},
            ],
        },
    )

    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body['num_chains'] == 2
    # Ligand has no sequence, so only the protein residues are counted.
    assert body['usage']['total_residues'] == 3
    assert backend.fold_calls[0]['entity_types'] == ['protein', 'ligand']


def test_protein_plus_nucleic_acid_accepted() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

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


def test_request_overrides_msa_and_esm_defaults() -> None:
    backend = FakeBackend()
    # Service defaults: esm on, msa off.
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/fold',
        json={
            'chains': [{'sequence': 'MK'}],
            'use_esm_embeddings': False,
            'use_msa_server': True,
            'num_trunk_recycles': 2,
            'num_diffn_timesteps': 80,
            'num_diffn_samples': 2,
            'seed': 7,
        },
    )

    assert response.status_code == HTTP_200_OK
    call = backend.fold_calls[0]
    assert call['use_esm_embeddings'] is False
    assert call['use_msa_server'] is True
    assert call['num_trunk_recycles'] == 2
    assert call['num_diffn_timesteps'] == 80
    assert call['num_diffn_samples'] == 2
    assert call['seed'] == 7


def test_defaults_come_from_settings_when_unset() -> None:
    backend = FakeBackend()
    client = TestClient(
        create_app(
            Settings(use_esm_embeddings=True, use_msa_server=False),
            backend,
        ),
    )

    response = client.post('/v1/fold', json={'chains': [{'sequence': 'MK'}]})

    assert response.status_code == HTTP_200_OK
    call = backend.fold_calls[0]
    assert call['use_esm_embeddings'] is True
    assert call['use_msa_server'] is False


def test_return_pae_includes_matrix(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={'chains': [{'sequence': 'MK'}], 'return_pae': True},
    )

    assert response.status_code == HTTP_200_OK
    assert response.json()['results'][0]['pae'] == [[0.0, 1.0]]


def test_lowercase_sequence_is_uppercased() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post('/v1/fold', json={'chains': [{'sequence': 'mkt'}]})

    assert response.status_code == HTTP_200_OK
    assert response.json()['usage']['total_residues'] == 3


def test_duplicate_chain_names_are_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={
            'chains': [
                {'name': 'A', 'sequence': 'MK'},
                {'name': 'A', 'sequence': 'AA'},
            ],
        },
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_empty_chains_list_is_rejected(client: TestClient) -> None:
    response = client.post('/v1/fold', json={'chains': []})
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_ligand_only_fold_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/fold',
        json={'chains': [{'entity_type': 'ligand', 'smiles': 'CCO'}]},
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
        {'entity_type': 'ligand'},  # ligand with no smiles
        {'entity_type': 'protein', 'sequence': 'MK', 'smiles': 'CCO'},  # polymer+lig
        {'entity_type': 'protein'},  # polymer with no sequence
        {'name': 'bad name', 'sequence': 'MK'},  # name has whitespace
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
        json={'chains': [{'sequence': 'MK'}], 'num_diffn_samples': 99},
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
    assert 'num_diffn_samples' in properties


def test_health(client: TestClient) -> None:
    assert client.get('/health').json() == {'status': 'healthy'}


def test_lifespan_uses_backend_lifecycle() -> None:
    backend = FakeBackend()

    with TestClient(create_app(Settings(), backend)):
        assert backend.load_calls == 1
        assert backend.unload_calls == 0

    assert backend.unload_calls == 1


def test_chainspec_renders_fasta_records() -> None:
    protein = ChainSpec(name='A', entity_type='protein', sequence='MKT')
    ligand = ChainSpec(name='L', entity_type='ligand', smiles='CCO')
    assert protein.fasta_record() == '>protein|name=A\nMKT\n'
    assert ligand.fasta_record() == '>ligand|name=L\nCCO\n'


# --- backend tests against an injected fake ``chai_lab`` ----------------------


class _PTM:
    def __init__(self, complex_ptm: float, interface_ptm: float) -> None:
        self.complex_ptm = torch.tensor(complex_ptm)
        self.interface_ptm = torch.tensor(interface_ptm)


class _Clash:
    def __init__(self, has_clash: bool) -> None:
        self.has_inter_chain_clashes = torch.tensor(has_clash)


class _Ranking:
    def __init__(
        self,
        aggregate: float,
        complex_ptm: float,
        interface_ptm: float,
        has_clash: bool,
    ) -> None:
        self.aggregate_score = torch.tensor([aggregate])
        self.ptm_scores = _PTM(complex_ptm, interface_ptm)
        self.clash_scores = _Clash(has_clash)


class _Candidates:
    def __init__(
        self,
        cif_paths: list[Path],
        ranking_data: list[_Ranking],
        pae: torch.Tensor,
    ) -> None:
        self.cif_paths = cif_paths
        self.ranking_data = ranking_data
        self.pae = pae

    def sorted(self) -> _Candidates:
        order = sorted(
            range(len(self.ranking_data)),
            key=lambda i: float(self.ranking_data[i].aggregate_score.item()),
            reverse=True,
        )
        return _Candidates(
            [self.cif_paths[i] for i in order],
            [self.ranking_data[i] for i in order],
            self.pae[order],
        )


def _install_fake_chai(
    monkeypatch: pytest.MonkeyPatch,
    record: list[dict[str, Any]] | None = None,
) -> None:
    """Install a stub ``chai_lab.chai1.run_inference``."""

    def run_inference(
        fasta_file: Path,
        *,
        output_dir: Path,
        num_diffn_samples: int = 5,
        seed: int | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> _Candidates:
        if record is not None:
            record.append(
                {
                    'fasta': Path(fasta_file).read_text(),
                    'num_diffn_samples': num_diffn_samples,
                    'seed': seed,
                    'device': device,
                    **kwargs,
                },
            )
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cif_paths: list[Path] = []
        rankings: list[_Ranking] = []
        for index in range(num_diffn_samples):
            cif = out / f'pred.model_idx_{index}.cif'
            cif.write_text(f'data_pred_{index}\n')
            cif_paths.append(cif)
            # aggregate rises with index, so sorted() must reverse the order.
            rankings.append(
                _Ranking(
                    aggregate=0.1 * index,
                    complex_ptm=0.5,
                    interface_ptm=0.6,
                    has_clash=(index == 0),
                ),
            )
        pae = torch.arange(num_diffn_samples * 4, dtype=torch.float32).reshape(
            num_diffn_samples,
            2,
            2,
        )
        return _Candidates(cif_paths, rankings, pae)

    chai1_mod = types.ModuleType('chai_lab.chai1')
    chai1_mod.run_inference = run_inference  # type: ignore[attr-defined]
    chai_pkg = types.ModuleType('chai_lab')
    monkeypatch.setitem(sys.modules, 'chai_lab', chai_pkg)
    monkeypatch.setitem(sys.modules, 'chai_lab.chai1', chai1_mod)


def test_backend_builds_fasta_and_maps_sorted_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    record: list[dict[str, Any]] = []
    _install_fake_chai(monkeypatch, record)

    backend = ChaiFoldBackend(Settings())
    chains = [
        ChainSpec(name='A', entity_type='protein', sequence='MKT'),
        ChainSpec(name='L', entity_type='ligand', smiles='CCO'),
    ]
    outputs = backend.fold(
        chains,
        use_esm_embeddings=True,
        use_msa_server=False,
        num_trunk_recycles=3,
        num_diffn_timesteps=200,
        num_diffn_samples=3,
        seed=11,
        return_pae=True,
    )

    # FASTA carries both entities with Chai-1 headers.
    assert record[0]['fasta'] == '>protein|name=A\nMKT\n>ligand|name=L\nCCO\n'
    assert record[0]['seed'] == 11
    assert record[0]['device'] == 'cpu'

    # Three samples, ranked best (highest aggregate) first: index 2, 1, 0.
    assert len(outputs) == 3
    assert [o.mmcif for o in outputs] == [
        'data_pred_2\n',
        'data_pred_1\n',
        'data_pred_0\n',
    ]
    assert [round(o.aggregate_score, 4) for o in outputs] == [0.2, 0.1, 0.0]
    # has_inter_chain_clashes was set on index 0, now last after sorting.
    assert [o.has_inter_chain_clashes for o in outputs] == [False, False, True]
    assert outputs[0].complex_ptm == pytest.approx(0.5)
    assert outputs[0].interface_ptm == pytest.approx(0.6)
    # PAE for the best sample (original index 2) comes back as a matrix.
    assert outputs[0].pae == [[8.0, 9.0], [10.0, 11.0]]


def test_backend_omits_pae_when_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    _install_fake_chai(monkeypatch)

    backend = ChaiFoldBackend(Settings())
    outputs = backend.fold(
        [ChainSpec(name='A', entity_type='protein', sequence='MK')],
        use_esm_embeddings=True,
        use_msa_server=False,
        num_trunk_recycles=3,
        num_diffn_timesteps=200,
        num_diffn_samples=1,
        seed=None,
        return_pae=False,
    )

    assert outputs[0].pae is None


def test_backend_load_imports_chai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    _install_fake_chai(monkeypatch)

    backend = ChaiFoldBackend(Settings())
    backend.load()
    backend.unload()  # exercises gc + empty_cache guard on CPU
