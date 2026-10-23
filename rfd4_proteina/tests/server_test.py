"""API tests for the RFD4-Proteina design service."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.status import HTTP_200_OK
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from rfd4_proteina_service import __version__
from rfd4_proteina_service.server import create_app
from rfd4_proteina_service.server import DesignItem
from rfd4_proteina_service.server import DesignOutput
from rfd4_proteina_service.server import DesignRequest
from rfd4_proteina_service.server import ProteinaBackend
from rfd4_proteina_service.server import Settings

_YAML = 'binder:\n  contig: "50-120, /0, A17-131"\n'


class FakeBackend:
    """Deterministic backend that avoids loading the real model."""

    def __init__(self) -> None:
        self.loaded_models: list[str] = []
        self.unload_calls = 0
        self.requests: list[DesignRequest] = []

    def load(self, model_id: str) -> None:
        self.loaded_models.append(model_id)

    def unload(self) -> None:
        self.unload_calls += 1

    def generate(self, request: DesignRequest) -> DesignOutput:
        self.requests.append(request)
        run_name = request.run_name or 'auto-run'
        return DesignOutput(
            run_name=run_name,
            output_dir=f'/runs/{run_name}',
            designs=[
                DesignItem(
                    id='design_0',
                    structure_path=f'/runs/{run_name}/inference_samples/design_0.cif',
                    format='cif',
                ),
            ],
        )


class RaisingBackend:
    """Backend that fails, to exercise the 500 path."""

    def load(self, model_id: str) -> None:
        return None

    def unload(self) -> None:
        return None

    def generate(self, request: DesignRequest) -> DesignOutput:
        raise RuntimeError('campaign exploded')


@pytest.fixture
def client() -> TestClient:
    """Return an API client backed by a deterministic campaign."""
    return TestClient(create_app(Settings(), FakeBackend()))


def test_design_response(client: TestClient) -> None:
    response = client.post(
        '/v1/design',
        json={'inputs_yaml': _YAML, 'run_name': 'campaign_a'},
    )

    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body['object'] == 'design'
    assert body['model'] == 'rfd4-proteina'
    assert body['run_name'] == 'campaign_a'
    assert body['output_dir'] == '/runs/campaign_a'
    assert body['num_designs'] == 1
    assert body['designs'][0]['format'] == 'cif'


def test_run_name_auto_generated() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post('/v1/design', json={'inputs_yaml': _YAML})

    assert response.status_code == HTTP_200_OK
    assert response.json()['run_name'] == 'auto-run'


def test_inputs_path_is_accepted() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/design',
        json={'inputs_path': '/data/binder.yaml'},
    )

    assert response.status_code == HTTP_200_OK
    assert backend.requests[0].inputs_path == '/data/binder.yaml'


def test_missing_inputs_is_rejected(client: TestClient) -> None:
    response = client.post('/v1/design', json={'run_name': 'x'})
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_both_inputs_sources_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/design',
        json={'inputs_yaml': _YAML, 'inputs_path': '/data/b.yaml'},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_unknown_model_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/design',
        json={'model': 'rfdiffusion', 'inputs_yaml': _YAML},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.parametrize('name', ('with space', 'semi;colon', '$(x)'))
def test_invalid_run_name_is_rejected(client: TestClient, name: str) -> None:
    response = client.post(
        '/v1/design',
        json={'inputs_yaml': _YAML, 'run_name': name},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.parametrize('override', ('inputs=evil.yaml', 'out_dir=/tmp', '++seed=9'))
def test_reserved_override_is_rejected(client: TestClient, override: str) -> None:
    response = client.post(
        '/v1/design',
        json={'inputs_yaml': _YAML, 'extra_overrides': [override]},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_invalid_override_token_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/design',
        json={'inputs_yaml': _YAML, 'extra_overrides': ['rm -rf /']},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_extra_field_is_forbidden(client: TestClient) -> None:
    response = client.post(
        '/v1/design',
        json={'inputs_yaml': _YAML, 'nonsense': 1},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_backend_failure_is_500() -> None:
    client = TestClient(create_app(Settings(), RaisingBackend()))
    response = client.post('/v1/design', json={'inputs_yaml': _YAML})
    assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR


def test_openapi_request_schema(client: TestClient) -> None:
    schema = client.get('/openapi.json').json()
    properties = schema['components']['schemas']['DesignRequest']['properties']

    assert schema['info']['version'] == __version__
    assert 'inputs_yaml' in properties
    assert 'num_replicates' in properties


def test_health(client: TestClient) -> None:
    assert client.get('/health').json() == {'status': 'healthy'}


def test_lifespan_uses_backend_lifecycle() -> None:
    backend = FakeBackend()

    with TestClient(create_app(Settings(preload_model='rfd4-proteina'), backend)):
        assert backend.loaded_models == ['rfd4-proteina']
        assert backend.unload_calls == 0

    assert backend.unload_calls == 1


def test_preload_model_must_be_supported() -> None:
    with pytest.raises(ValidationError, match='preload_model must be one of'):
        Settings(preload_model='unsupported/model')


def _install_fake_rfd4(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[str] | None = None,
    *,
    write: bool = True,
) -> types.ModuleType:
    """Install a stub ``rfd4_proteina`` package so the real backend can run."""

    class Proteina:
        @classmethod
        def from_pretrained(cls, repo: str, revision: str | None = None) -> object:
            if calls is not None:
                calls.append(repo)
            return object()

    captured: dict[str, Any] = {}

    def generate(model: object, overrides: list[str]) -> None:
        captured['overrides'] = overrides
        out_dir = next(
            (tok.split('=', 1)[1] for tok in overrides if tok.startswith('out_dir=')),
            None,
        )
        if write and out_dir is not None:
            samples = Path(out_dir) / 'inference_samples'
            samples.mkdir(parents=True, exist_ok=True)
            (samples / 'design_0.cif').write_text('data_x\n')

    module = types.ModuleType('rfd4_proteina')
    module.Proteina = Proteina  # type: ignore[attr-defined]
    module.generate = generate  # type: ignore[attr-defined]
    module.captured = captured  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'rfd4_proteina', module)
    return module


def test_backend_loads_and_caches_model(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _install_fake_rfd4(monkeypatch, calls)

    backend = ProteinaBackend(Settings())
    loaded = backend.load('rfd4-proteina')
    cached = backend.load('rfd4-proteina')

    assert cached is loaded
    assert calls == ['nvidia/RFD4-Proteina']

    backend.unload()
    assert backend.loaded is None


def test_generate_builds_overrides_and_collects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _install_fake_rfd4(monkeypatch)

    backend = ProteinaBackend(Settings(runs_root=str(tmp_path / 'runs')))
    output = backend.generate(
        DesignRequest(
            inputs_yaml=_YAML,
            run_name='run1',
            num_replicates=2,
            seed=7,
            extra_overrides=['++generation.num_samples=4'],
        ),
    )

    overrides = module.captured['overrides']
    # inputs= points at the YAML the service wrote into the run dir.
    inputs_file = tmp_path / 'runs' / 'run1' / 'inputs.yaml'
    assert f'inputs={inputs_file}' in overrides
    assert inputs_file.read_text() == _YAML
    assert f'out_dir={tmp_path / "runs" / "run1"}' in overrides
    assert 'num_replicates=2' in overrides
    assert 'seed=7' in overrides
    assert 'inference_experiment=quickstart' in overrides
    assert '++generation.num_samples=4' in overrides

    assert output.run_name == 'run1'
    assert len(output.designs) == 1
    assert output.designs[0].structure_path.endswith('inference_samples/design_0.cif')


def test_generate_with_inputs_path_does_not_write_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _install_fake_rfd4(monkeypatch)

    backend = ProteinaBackend(Settings(runs_root=str(tmp_path / 'runs')))
    backend.generate(
        DesignRequest(inputs_path='/data/binder.yaml', run_name='run2'),
    )

    overrides = module.captured['overrides']
    assert 'inputs=/data/binder.yaml' in overrides
    assert not (tmp_path / 'runs' / 'run2' / 'inputs.yaml').exists()


def test_generate_no_designs_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_rfd4(monkeypatch, write=False)

    backend = ProteinaBackend(Settings(runs_root=str(tmp_path / 'runs')))
    with pytest.raises(RuntimeError, match='no designs'):
        backend.generate(DesignRequest(inputs_yaml=_YAML, run_name='run3'))
