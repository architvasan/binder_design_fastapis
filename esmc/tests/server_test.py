"""API tests for the ESM-C embeddings service."""

from __future__ import annotations

import sys
import types

import pytest
import torch
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.status import HTTP_200_OK
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT

from esmc_service import __version__
from esmc_service.server import create_app
from esmc_service.server import EsmcEmbeddingBackend
from esmc_service.server import Settings


class FakeBackend:
    """Small deterministic backend that avoids downloading a model."""

    def __init__(self) -> None:
        self.loaded_models: list[str] = []
        self.unload_calls = 0
        self.transformer_layers: list[int] = []

    def load(self, model_id: str) -> None:
        self.loaded_models.append(model_id)

    def unload(self) -> None:
        self.unload_calls += 1

    def embed(
        self,
        model: str,
        sequences: list[str],
        transformer_layer: int,
    ) -> list[list[float]]:
        self.transformer_layers.append(transformer_layer)
        return [
            [float(index), float(len(sequence))]
            for index, sequence in enumerate(sequences)
        ]


@pytest.fixture
def client() -> TestClient:
    """Return an API client backed by deterministic embeddings."""
    return TestClient(create_app(Settings(), FakeBackend()))


def test_embeddings_float_response(client: TestClient) -> None:
    response = client.post(
        '/v1/embeddings',
        json={
            'model': 'esmc_600m',
            'input': ['MTPH', 'MK'],
        },
    )

    assert response.status_code == HTTP_200_OK
    assert response.json() == {
        'object': 'list',
        'data': [
            {'object': 'embedding', 'embedding': [0.0, 4.0], 'index': 0},
            {'object': 'embedding', 'embedding': [1.0, 2.0], 'index': 1},
        ],
        'model': 'esmc_600m',
        'usage': {'prompt_tokens': 6, 'total_tokens': 6},
    }


def test_transformer_layer_defaults_to_model_final_layer() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/embeddings',
        json={'model': 'esmc_300m', 'input': ['MK']},
    )

    assert response.status_code == HTTP_200_OK
    assert backend.transformer_layers == [30]


def test_requested_transformer_layer_is_forwarded() -> None:
    backend = FakeBackend()
    client = TestClient(create_app(Settings(), backend))

    response = client.post(
        '/v1/embeddings',
        json={'model': 'esmc_600m', 'input': ['MK'], 'transformer_layer': 12},
    )

    assert response.status_code == HTTP_200_OK
    assert backend.transformer_layers == [12]


@pytest.mark.parametrize(
    ('model', 'transformer_layer'),
    (
        ('esmc_300m', 31),
        ('esmc_600m', 37),
    ),
)
def test_transformer_layer_must_exist(
    client: TestClient,
    model: str,
    transformer_layer: int,
) -> None:
    response = client.post(
        '/v1/embeddings',
        json={
            'model': model,
            'input': ['MK'],
            'transformer_layer': transformer_layer,
        },
    )

    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


@pytest.mark.parametrize('input_value', ([], ['MK', '']))
def test_empty_input_is_rejected(
    client: TestClient,
    input_value: object,
) -> None:
    response = client.post(
        '/v1/embeddings',
        json={'model': 'esmc_300m', 'input': input_value},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_scalar_input_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/embeddings',
        json={'model': 'esmc_300m', 'input': 'MK'},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_unknown_model_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/embeddings',
        json={'model': 'arbitrary/model', 'input': ['MK']},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_unknown_request_field_remains_forbidden(client: TestClient) -> None:
    response = client.post(
        '/v1/embeddings',
        json={'model': 'esmc_300m', 'input': ['MK'], 'unknown': False},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_streaming_is_rejected(client: TestClient) -> None:
    response = client.post(
        '/v1/embeddings',
        json={'model': 'esmc_300m', 'input': ['MK'], 'stream': True},
    )
    assert response.status_code == HTTP_422_UNPROCESSABLE_CONTENT


def test_openapi_request_schema(client: TestClient) -> None:
    schema = client.get('/openapi.json').json()
    properties = schema['components']['schemas']['EmbeddingsRequest']['properties']

    assert schema['info']['version'] == __version__
    assert properties['input']['type'] == 'array'
    assert properties['input']['items']['type'] == 'string'
    assert properties['stream']['const'] is False
    assert 'modality' not in properties
    assert 'encoding_format' not in properties


def test_health(client: TestClient) -> None:
    assert client.get('/health').json() == {'status': 'healthy'}


def test_lifespan_uses_backend_lifecycle() -> None:
    backend = FakeBackend()

    with TestClient(create_app(Settings(preload_model='esmc_300m'), backend)):
        assert backend.loaded_models == ['esmc_300m']
        assert backend.unload_calls == 0

    assert backend.unload_calls == 1


def test_preload_model_must_be_supported() -> None:
    with pytest.raises(ValidationError, match='preload_model must be one of'):
        Settings(preload_model='unsupported/model')


def _install_fake_esm(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Install a stub ``esm.models.esmc`` so the real backend can load."""

    class FakeESMC:
        @classmethod
        def from_pretrained(cls, model_id: str) -> torch.nn.Module:
            calls.append(model_id)
            return torch.nn.Linear(2, 2)

    esm_pkg = types.ModuleType('esm')
    models_pkg = types.ModuleType('esm.models')
    esmc_pkg = types.ModuleType('esm.models.esmc')
    esmc_pkg.ESMC = FakeESMC  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'esm', esm_pkg)
    monkeypatch.setitem(sys.modules, 'esm.models', models_pkg)
    monkeypatch.setitem(sys.modules, 'esm.models.esmc', esmc_pkg)


def test_backend_loads_and_caches_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _install_fake_esm(monkeypatch, calls)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    backend = EsmcEmbeddingBackend(Settings())
    loaded = backend.load('esmc_300m')
    cached = backend.load('esmc_300m')

    assert cached is loaded
    assert loaded.device == torch.device('cpu')
    assert not loaded.model.training
    assert next(loaded.model.parameters()).dtype == torch.float32
    assert calls == ['esmc_300m']

    backend.unload()
    assert backend.loaded is None


def test_backend_swaps_resident_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _install_fake_esm(monkeypatch, calls)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    backend = EsmcEmbeddingBackend(Settings())
    backend.load('esmc_300m')
    backend.load('esmc_600m')

    assert calls == ['esmc_300m', 'esmc_600m']
    assert backend.loaded is not None
    assert backend.loaded.model_id == 'esmc_600m'


class _FakeLogitsOutput:
    """Mimics ESM-C LogitsOutput with a known hidden_states tensor."""

    def __init__(self, hidden_states: torch.Tensor) -> None:
        self.hidden_states = hidden_states


class _FakeEsmcModel(torch.nn.Module):
    """A model whose hidden state at [layer, 0, token] equals layer*100+token.

    This makes the expected mean-pool over residues exactly computable, so the
    test pins the layer indexing and the BOS/EOS exclusion in ``embed``.
    """

    def __init__(self, n_layers: int, dim: int = 4) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim

    def encode(self, protein: object) -> object:
        return protein

    def logits(self, tensor: object, config: object) -> _FakeLogitsOutput:
        length = len(tensor.sequence)  # type: ignore[attr-defined]
        n_tokens = length + 2  # ESM-C adds BOS and EOS
        hidden = torch.zeros(self.n_layers + 1, 1, n_tokens, self.dim)
        for layer in range(self.n_layers + 1):
            for token in range(n_tokens):
                hidden[layer, 0, token, :] = layer * 100 + token
        return _FakeLogitsOutput(hidden)


def _install_fake_esm_for_embed(
    monkeypatch: pytest.MonkeyPatch,
    model: torch.nn.Module,
) -> None:
    """Install stub ``esm.models.esmc`` and ``esm.sdk.api`` for embedding."""

    class ESMC:
        @classmethod
        def from_pretrained(cls, model_id: str) -> torch.nn.Module:
            return model

    class ESMProtein:
        def __init__(self, sequence: str) -> None:
            self.sequence = sequence

    class LogitsConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    esm_pkg = types.ModuleType('esm')
    models_pkg = types.ModuleType('esm.models')
    esmc_pkg = types.ModuleType('esm.models.esmc')
    esmc_pkg.ESMC = ESMC  # type: ignore[attr-defined]
    sdk_pkg = types.ModuleType('esm.sdk')
    api_pkg = types.ModuleType('esm.sdk.api')
    api_pkg.ESMProtein = ESMProtein  # type: ignore[attr-defined]
    api_pkg.LogitsConfig = LogitsConfig  # type: ignore[attr-defined]
    for name, module in {
        'esm': esm_pkg,
        'esm.models': models_pkg,
        'esm.models.esmc': esmc_pkg,
        'esm.sdk': sdk_pkg,
        'esm.sdk.api': api_pkg,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_embed_pools_requested_layer_excluding_bos_eos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    _install_fake_esm_for_embed(monkeypatch, _FakeEsmcModel(n_layers=30, dim=4))

    backend = EsmcEmbeddingBackend(Settings())
    # 'AAA' -> tokens [BOS, r1, r2, r3, EOS]; residues are tokens 1,2,3.
    # Layer 5 values are 500+token, so pooled = 500 + mean(1,2,3) = 502.
    vectors = backend.embed('esmc_300m', ['AAA'], transformer_layer=5)

    assert vectors == [[502.0, 502.0, 502.0, 502.0]]


def test_embed_final_layer_and_multiple_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    _install_fake_esm_for_embed(monkeypatch, _FakeEsmcModel(n_layers=36, dim=2))

    backend = EsmcEmbeddingBackend(Settings())
    # esmc_600m final layer = 36. 'MK' residues are tokens 1,2:
    # 3600 + mean(1,2) = 3601.5. 'MTP' residues tokens 1,2,3: 3600 + 2 = 3602.
    vectors = backend.embed('esmc_600m', ['MK', 'MTP'], transformer_layer=36)

    assert vectors == [[3601.5, 3601.5], [3602.0, 3602.0]]


def test_embed_rejects_unexpected_layer_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    # Model reports 10 layers but esmc_300m expects 30 -> guard must fire.
    _install_fake_esm_for_embed(monkeypatch, _FakeEsmcModel(n_layers=10, dim=4))

    backend = EsmcEmbeddingBackend(Settings())
    with pytest.raises(RuntimeError, match='expected 31 hidden layers'):
        backend.embed('esmc_300m', ['AAA'], transformer_layer=5)
