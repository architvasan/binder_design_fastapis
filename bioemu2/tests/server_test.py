"""API tests for the BioEmu2 equilibrium-sampling service."""

from __future__ import annotations

import base64
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.status import HTTP_200_OK
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT

from bioemu2_service import __version__
from bioemu2_service.server import BioEmu2Backend
from bioemu2_service.server import create_app
from bioemu2_service.server import SampleOutput
from bioemu2_service.server import Settings


class FakeBackend:
    """Small deterministic backend that avoids downloading a model."""

    def __init__(self) -> None:
        self.loaded_models: list[str] = []
        self.unload_calls = 0
        self.sample_calls: list[dict[str, Any]] = []

    def load(self, model_id: str) -> None:
        self.loaded_models.append(model_id)

    def unload(self) -> None:
        self.unload_calls += 1

    def sample(
        self,
        model: str,
        sequence: str,
        *,
        num_samples: int,
        mode: str,
        return_trajectory: bool,
    ) -> SampleOutput:
        self.sample_calls.append(
            {
                'model': model,
                'sequence': sequence,
                'num_samples': num_samples,
                'mode': mode,
                'return_trajectory': return_trajectory,
            },
        )
        return SampleOutput(
            topology_cif='CIF',
            num_samples=num_samples,
            trajectory_xtc=b'XTCDATA' if return_trajectory else None,
        )


@pytest.fixture
def client() -> TestClient:
    """Return an API client backed by deterministic sampling."""
    return TestClient(create_app(Settings(), FakeBackend()))


def test_sample_response(client: TestClient) -> None:
    response = client.post(
        '/v1/sample',
        json={'model': 'bioemu2', 'sequence': 'GYDPETGTWG', 'num_samples': 8},
    )

    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body['object'] == 'sample'
    assert body['format'] == 'mmcif'
    assert body['topology'] == 'CIF'
    assert body['num_samples'] == 8
    assert body['trajectory_xtc_base64'] is None
    assert body['usage'] == {'sequence_length': 10, 'num_samples': 8}


def test_num_samples_defaults() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/sample',
        json={'model': 'bioemu2', 'sequence': 'MK'},
    )

    assert response.status_code == HTTP_200_OK
    assert backend.sample_calls[0]['num_samples'] == 10


def test_request_parameters_are_forwarded() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/sample',
        json={
            'model': 'bioemu2',
            'sequence': 'mkt',
            'num_samples': 5,
            'mode': 'equilibrium',
        },
    )

    assert response.status_code == HTTP_200_OK
    call = backend.sample_calls[0]
    # Sequence is uppercased before reaching the backend.
    assert call['sequence'] == 'MKT'
    assert call['num_samples'] == 5
    assert call['mode'] == 'equilibrium'


def test_return_trajectory_is_base64(client: TestClient) -> None:
    response = client.post(
        '/v1/sample',
        json={
            'model': 'bioemu2',
            'sequence': 'MK',
            'return_trajectory': True,
        },
    )

    assert response.status_code == HTTP_200_OK
    encoded = response.json()['trajectory_xtc_base64']
    assert base64.b64decode(encoded) == b'XTCDATA'


@pytest.mark.parametrize('sequence', ('', '   ', 'MK1', 'MK*'))
def test_invalid_sequences_are_rejected(
    client: TestClient,
    sequence: str,
) -> None:
    response = client.post(
        '/v1/sample',
        json={'model': 'bioemu2', 'sequence': sequence},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_unknown_model_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/sample',
        json={'model': 'bioemu-v1', 'sequence': 'MK'},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.parametrize('num_samples', (0, 1001))
def test_num_samples_bounds(client: TestClient, num_samples: int) -> None:
    response = client.post(
        '/v1/sample',
        json={'model': 'bioemu2', 'sequence': 'MK', 'num_samples': num_samples},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_unknown_request_field_remains_forbidden(client: TestClient) -> None:
    response = client.post(
        '/v1/sample',
        json={'model': 'bioemu2', 'sequence': 'MK', 'temperature': 300},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_openapi_request_schema(client: TestClient) -> None:
    schema = client.get('/openapi.json').json()
    properties = schema['components']['schemas']['SampleRequest']['properties']

    assert schema['info']['version'] == __version__
    assert properties['sequence']['type'] == 'string'
    assert 'num_samples' in properties
    assert 'return_trajectory' in properties


def test_health(client: TestClient) -> None:
    assert client.get('/health').json() == {'status': 'healthy'}


def test_lifespan_uses_backend_lifecycle() -> None:
    backend = FakeBackend()

    with TestClient(create_app(Settings(preload_model='bioemu2'), backend)):
        assert backend.loaded_models == ['bioemu2']
        assert backend.unload_calls == 0

    assert backend.unload_calls == 1


def test_preload_model_must_be_supported() -> None:
    with pytest.raises(ValidationError, match='preload_model must be one of'):
        Settings(preload_model='unsupported/model')


def _install_fake_bioemu2(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str] | None = None,
) -> types.ModuleType:
    """Install a stub ``bioemu2`` package so the real backend can run."""

    def load_model(checkpoint: str | None = None) -> object:
        if calls is not None:
            calls.append(checkpoint or 'default')
        return object()

    class System:
        def __init__(self, sequence: str) -> None:
            self.sequence = sequence

        @classmethod
        def from_sequence(cls, sequence: str) -> System:
            return cls(sequence)

    class EmuSession:
        last_init: Any = None
        last_num_samples: int | None = None

        def __init__(self, system: Any, model: object, mode: str) -> None:
            type(self).last_init = {
                'sequence': system.sequence,
                'model': model,
                'mode': mode,
            }

        def sample(self, num_samples: int) -> dict[str, int]:
            type(self).last_num_samples = num_samples
            return {'num_samples': num_samples}

    def save_frames(samples: Any, out_dir: Path) -> None:
        (out_dir / 'topology.cif').write_text('data_top\n')
        (out_dir / 'samples.xtc').write_bytes(b'XTCBYTES')

    module = types.ModuleType('bioemu2')
    module.load_model = load_model  # type: ignore[attr-defined]
    module.System = System  # type: ignore[attr-defined]
    module.EmuSession = EmuSession  # type: ignore[attr-defined]
    module.save_frames = save_frames  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'bioemu2', module)
    return module


def test_backend_loads_and_caches_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _install_fake_bioemu2(monkeypatch, calls)

    backend = BioEmu2Backend(Settings())
    loaded = backend.load('bioemu2')
    cached = backend.load('bioemu2')

    assert cached is loaded
    # 'bioemu2' maps to the default checkpoint (None -> load_model()).
    assert calls == ['default']

    backend.unload()
    assert backend.loaded is None


def test_sample_builds_system_and_maps_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _install_fake_bioemu2(monkeypatch)

    backend = BioEmu2Backend(Settings())
    output = backend.sample(
        'bioemu2',
        'MKTAA',
        num_samples=4,
        mode='equilibrium',
        return_trajectory=True,
    )

    assert output.topology_cif == 'data_top\n'
    assert output.num_samples == 4
    assert output.trajectory_xtc == b'XTCBYTES'

    session_cls = module.EmuSession
    assert session_cls.last_init['sequence'] == 'MKTAA'
    assert session_cls.last_init['mode'] == 'equilibrium'
    assert session_cls.last_num_samples == 4


def test_sample_omits_trajectory_when_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_bioemu2(monkeypatch)

    backend = BioEmu2Backend(Settings())
    output = backend.sample(
        'bioemu2',
        'MK',
        num_samples=2,
        mode='equilibrium',
        return_trajectory=False,
    )

    assert output.trajectory_xtc is None


def test_sample_errors_without_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _install_fake_bioemu2(monkeypatch)

    def save_nothing(samples: Any, out_dir: Path) -> None:
        return None

    module.save_frames = save_nothing  # type: ignore[attr-defined]

    backend = BioEmu2Backend(Settings())
    with pytest.raises(RuntimeError, match='did not write topology'):
        backend.sample(
            'bioemu2',
            'MK',
            num_samples=1,
            mode='equilibrium',
            return_trajectory=False,
        )
