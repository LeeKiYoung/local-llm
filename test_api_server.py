"""
llm-api-server.py 테스트
실행: .venv/bin/python -m pytest test_api_server.py -v
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

# mlx.core를 mock으로 대체 (GPU 없는 환경에서도 테스트 가능)
mock_mx = MagicMock()
mock_mx.metal.clear_cache = MagicMock()
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = mock_mx

# mlx_lm mock
mock_mlx_lm = MagicMock()
mock_sample_utils = MagicMock()
mock_sample_utils.make_sampler.return_value = lambda x: x
sys.modules["mlx_lm"] = mock_mlx_lm
sys.modules["mlx_lm.sample_utils"] = mock_sample_utils

import importlib.util

# 하이픈 파일명이라 importlib으로 로드
_spec = importlib.util.spec_from_file_location(
    "llm_api_server",
    os.path.join(os.path.dirname(__file__), "llm-api-server.py"),
)
server_module = importlib.util.module_from_spec(_spec)
sys.modules["llm_api_server"] = server_module
_spec.loader.exec_module(server_module)

import pytest
from fastapi.testclient import TestClient


@dataclass
class MockResponse:
    text: str
    token: int = 0
    logprobs: object = None
    from_draft: bool = False
    prompt_tokens: int = 10
    generation_tokens: int = 0
    prompt_tps: float = 100.0
    generation_tps: float = 50.0
    peak_memory: float = 20.0
    finish_reason: str = None


def mock_stream_generate(model, tokenizer, prompt, max_tokens=256, **kwargs):
    yield MockResponse("Hello", generation_tokens=1)
    yield MockResponse(" world", generation_tokens=2)
    yield MockResponse("!", generation_tokens=3, finish_reason="stop")


@pytest.fixture(autouse=True)
def setup():
    server_module.model = MagicMock()
    server_module.tokenizer = MagicMock()
    server_module.tokenizer.apply_chat_template.return_value = "formatted prompt"
    server_module.model_id = "test-model"
    server_module.gpu_semaphore = asyncio.Semaphore(1)
    server_module.pending = 0
    server_module.MAX_QUEUE = 5
    server_module.DEFAULT_THINKING = False
    server_module.LOG_DIR = "/tmp/llm-test-logs"
    os.makedirs("/tmp/llm-test-logs", exist_ok=True)

    # stream_generate mock 패치
    server_module.stream_generate = mock_stream_generate
    yield


@pytest.fixture
def client():
    return TestClient(server_module.app)


# ── GET /v1/models ───────────────────────────────
class TestModels:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"
        assert data["data"][0]["object"] == "model"
        assert data["data"][0]["owned_by"] == "local"


# ── POST /v1/chat/completions (non-streaming) ───
class TestChatCompletions:
    def test_basic_request(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "안녕!"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["id"].startswith("chatcmpl-")
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Hello world!"
        assert data["choices"][0]["finish_reason"] in ("stop", "length")
        assert "usage" in data
        assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]

    def test_enable_thinking_true(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "enable_thinking": True,
        })
        assert resp.status_code == 200
        call_kwargs = server_module.tokenizer.apply_chat_template.call_args
        assert call_kwargs.kwargs.get("enable_thinking") is True

    def test_enable_thinking_default_false(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        })
        assert resp.status_code == 200
        call_kwargs = server_module.tokenizer.apply_chat_template.call_args
        assert call_kwargs.kwargs.get("enable_thinking") is False

    def test_custom_parameters(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "max_tokens": 100,
        })
        assert resp.status_code == 200

    def test_max_completion_tokens(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "max_completion_tokens": 500,
        })
        assert resp.status_code == 200

    def test_response_has_created_timestamp(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        })
        data = resp.json()
        assert isinstance(data["created"], int)
        assert data["created"] > 0


# ── POST /v1/chat/completions (streaming) ────────
class TestStreaming:
    def test_stream_response_format(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_stream_chunks(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        })
        lines = [l for l in resp.text.strip().split("\n\n") if l.strip()]

        # 첫 청크: role
        first = json.loads(lines[0].replace("data: ", ""))
        assert first["object"] == "chat.completion.chunk"
        assert first["choices"][0]["delta"].get("role") == "assistant"

        # 마지막: [DONE]
        assert lines[-1].strip() == "data: [DONE]"

        # finish_reason 청크
        last_chunk = json.loads(lines[-2].replace("data: ", ""))
        assert last_chunk["choices"][0]["finish_reason"] is not None

    def test_stream_with_thinking(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
            "enable_thinking": True,
        })
        assert resp.status_code == 200
        call_kwargs = server_module.tokenizer.apply_chat_template.call_args
        assert call_kwargs.kwargs.get("enable_thinking") is True


# ── 큐 제한 (429) ────────────────────────────────
class TestRateLimit:
    def test_queue_full_returns_429(self, client):
        server_module.pending = server_module.MAX_QUEUE
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        })
        assert resp.status_code == 429
        data = resp.json()
        assert "error" in data
        server_module.pending = 0

    def test_queue_not_full_returns_200(self, client):
        server_module.pending = server_module.MAX_QUEUE - 1
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        })
        assert resp.status_code == 200
        server_module.pending = 0


# ── 로그 파일 ────────────────────────────────────
class TestLogging:
    def test_log_file_created(self, client):
        client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "로그 테스트"}],
        })
        from datetime import datetime
        log_file = os.path.join("/tmp/llm-test-logs", f"{datetime.now().strftime('%Y-%m-%d')}.jsonl")
        assert os.path.exists(log_file)

        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) > 0

        entry = json.loads(lines[-1])
        assert "timestamp" in entry
        assert "ip" in entry
        assert "duration_ms" in entry
        assert "usage" in entry
        assert "prompt_preview" in entry
