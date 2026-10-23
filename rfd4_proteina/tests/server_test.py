"""API tests for the RFD4 / Proteina-Complexa design service."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.status import HTTP_200_OK
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from rfd4_proteina_service import __version__
from rfd4_proteina_service import server
from rfd4_proteina_service.server import create_app
from rfd4_proteina_service.server import DesignItem
from rfd4_proteina_service.server import DesignOutput
from rfd4_proteina_service.server import DesignRequest
from rfd4_proteina_service.server import ProteinaBackend
from rfd4_proteina_service.server import Settings


class FakeBackend:
    """Deterministic backend that avoids launching the real pipeline."""

    def __init__(self) -> None:
        self.requests: list[DesignRequest] = []

    def generate(self, request: DesignRequest) -> DesignOutput:
        self.requests.append(request)
        run_name = request.run_name or f'{request.task_name}-auto'
        return DesignOutput(
            run_name=run_name,
            output_dir=f'/runs/{run_name}',
            returncode=0,
            designs=[
                DesignItem(
                    id='design_0',
                    structure_path=f'/runs/{run_name}/design_0.cif',
                    format='cif',
                ),
            ],
            log_tail='ok',
        )


class RaisingBackend:
    """Backend that fails, to exercise the 500 path."""

    def generate(self, request: DesignRequest) -> DesignOutput:
        raise RuntimeError('campaign exploded')


@pytest.fixture
def client() -> TestClient:
    """Return an API client backed by a deterministic campaign."""
    return TestClient(create_app(Settings(), FakeBackend()))


def test_design_response(client: TestClient) -> None:
    response = client.post(
        '/v1/design',
        json={'task_name': '02_PDL1', 'run_name': 'campaign_a', 'seed': 7},
    )

    assert response.status_code == HTTP_200_OK
    body = response.json()
    assert body['object'] == 'design'
    assert body['pipeline'] == 'binder'
    assert body['task_name'] == '02_PDL1'
    assert body['run_name'] == 'campaign_a'
    assert body['output_dir'] == '/runs/campaign_a'
    assert body['returncode'] == 0
    assert body['num_designs'] == 1
    assert body['designs'] == [
        {
            'id': 'design_0',
            'structure_path': '/runs/campaign_a/design_0.cif',
            'format': 'cif',
        },
    ]


def test_run_name_auto_generated_when_omitted() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post('/v1/design', json={'task_name': '02_PDL1'})

    assert response.status_code == HTTP_200_OK
    assert response.json()['run_name'] == '02_PDL1-auto'


def test_task_name_is_required(client: TestClient) -> None:
    response = client.post('/v1/design', json={'run_name': 'x'})
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_unknown_pipeline_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/design',
        json={'task_name': '02_PDL1', 'pipeline': 'enzyme'},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.parametrize('name', ('with space', 'semi;colon', 'back`tick', '$(x)'))
def test_invalid_task_name_is_rejected(client: TestClient, name: str) -> None:
    response = client.post('/v1/design', json={'task_name': name})
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_invalid_override_token_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/design',
        json={'task_name': '02_PDL1', 'extra_overrides': ['rm -rf /']},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.parametrize(
    'override',
    ('++gen_njobs=4', '++eval_njobs=2', '++seed=9', '++run_name=evil'),
)
def test_reserved_override_is_rejected(client: TestClient, override: str) -> None:
    response = client.post(
        '/v1/design',
        json={'task_name': '02_PDL1', 'extra_overrides': [override]},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_extra_field_is_forbidden(client: TestClient) -> None:
    response = client.post(
        '/v1/design',
        json={'task_name': '02_PDL1', 'nonsense': 1},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_negative_seed_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/design',
        json={'task_name': '02_PDL1', 'seed': -1},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_backend_failure_is_500() -> None:
    client = TestClient(create_app(Settings(), RaisingBackend()))
    response = client.post('/v1/design', json={'task_name': '02_PDL1'})
    assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR


def test_openapi_request_schema(client: TestClient) -> None:
    schema = client.get('/openapi.json').json()
    properties = schema['components']['schemas']['DesignRequest']['properties']

    assert schema['info']['version'] == __version__
    assert 'task_name' in properties
    assert 'extra_overrides' in properties


def test_health(client: TestClient) -> None:
    assert client.get('/health').json() == {'status': 'healthy'}


def _fake_run_factory(
    captured: dict[str, Any],
    *,
    returncode: int = 0,
    write: bool = True,
) -> Any:
    """Build a fake ``subprocess.run`` that records argv and writes a design."""

    def fake_run(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        captured['argv'] = argv
        captured['cwd'] = kwargs.get('cwd')
        out_dir = next(
            (tok.split('=', 1)[1] for tok in argv if tok.startswith('out_dir=')),
            None,
        )
        if write and out_dir is not None:
            samples = Path(out_dir) / 'inference_samples'
            samples.mkdir(parents=True, exist_ok=True)
            (samples / 'design_0.cif').write_text('data_x\n')
        return SimpleNamespace(returncode=returncode, stdout='log', stderr='')

    return fake_run


def _settings(tmp_path: Path) -> Settings:
    return Settings(repo_dir=str(tmp_path / 'repo'), runs_root=str(tmp_path / 'runs'))


def test_generate_builds_argv_and_collects_designs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(server.subprocess, 'run', _fake_run_factory(captured))

    backend = ProteinaBackend(_settings(tmp_path))
    output = backend.generate(
        DesignRequest(
            task_name='02_PDL1',
            run_name='run1',
            seed=7,
            extra_overrides=['++generation.num_samples=4'],
        ),
    )

    argv = captured['argv']
    assert argv[:3] == [
        'complexa',
        'design',
        'configs/search_binder_local_pipeline.yaml',
    ]
    assert '++run_name=run1' in argv
    assert '++generation.task_name=02_PDL1' in argv
    assert '++gen_njobs=1' in argv
    assert '++eval_njobs=1' in argv
    assert '++seed=7' in argv
    assert '++generation.num_samples=4' in argv
    assert captured['cwd'] == str(tmp_path / 'repo')

    assert output.returncode == 0
    assert output.run_name == 'run1'
    assert output.output_dir == str(tmp_path / 'runs' / 'run1')
    assert len(output.designs) == 1
    assert output.designs[0].format == 'cif'
    assert output.designs[0].structure_path.endswith('inference_samples/design_0.cif')


def test_generate_auto_run_name_uses_task_and_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(server.subprocess, 'run', _fake_run_factory(captured))

    backend = ProteinaBackend(_settings(tmp_path))
    output = backend.generate(DesignRequest(task_name='02_PDL1', seed=5))

    assert output.run_name.startswith('02_PDL1-5-')
    assert f'++run_name={output.run_name}' in captured['argv']


def test_generate_nonzero_returncode_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        server.subprocess,
        'run',
        _fake_run_factory(captured, returncode=2, write=False),
    )

    backend = ProteinaBackend(_settings(tmp_path))
    with pytest.raises(RuntimeError, match='failed'):
        backend.generate(DesignRequest(task_name='02_PDL1'))


def test_generate_no_designs_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        server.subprocess,
        'run',
        _fake_run_factory(captured, write=False),
    )

    backend = ProteinaBackend(_settings(tmp_path))
    with pytest.raises(RuntimeError, match='no designs'):
        backend.generate(DesignRequest(task_name='02_PDL1'))


def test_generate_timeout_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def timing_out(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        raise subprocess.TimeoutExpired(cmd=argv, timeout=1.0)

    monkeypatch.setattr(server.subprocess, 'run', timing_out)

    backend = ProteinaBackend(_settings(tmp_path))
    with pytest.raises(RuntimeError, match='timed out'):
        backend.generate(DesignRequest(task_name='02_PDL1'))


def test_generate_command_not_found_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def missing(argv: list[str], **kwargs: Any) -> SimpleNamespace:
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(server.subprocess, 'run', missing)

    backend = ProteinaBackend(_settings(tmp_path))
    with pytest.raises(RuntimeError, match='not found'):
        backend.generate(DesignRequest(task_name='02_PDL1'))
