"""
llm-api-server.py 테스트
실행: .venv/bin/python -m pytest test_api_server.py -v
"""

import asyncio
import base64
import json
import io
import os
import sys
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, call

# mlx.core를 mock으로 대체 (GPU 없는 환경에서도 테스트 가능)
mock_mx = MagicMock()
mock_mx.metal.clear_cache = MagicMock()
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = mock_mx

# mlx_vlm mock
mock_mlx_vlm = MagicMock()
mock_mlx_vlm_prompt_utils = MagicMock()
sys.modules["mlx_vlm"] = mock_mlx_vlm
sys.modules["mlx_vlm.prompt_utils"] = mock_mlx_vlm_prompt_utils

# PIL mock (Pillow 없는 환경에서도 테스트 가능)
mock_pil = MagicMock()
sys.modules["PIL"] = mock_pil
sys.modules["PIL.Image"] = mock_pil.Image

import importlib.util

# 하이픈 파일명이라 importlib으로 로드
_spec = importlib.util.spec_from_file_location(
    "llm_api_server",
    os.path.join(os.path.dirname(__file__), "llm-api-server.py"),
)
server_module = importlib.util.module_from_spec(_spec)
sys.modules["llm_api_server"] = server_module
_spec.loader.exec_module(server_module)

# Capture real function references before autouse fixture overrides them.
# The fixture sets server_module.extract_images = MagicMock(return_value=[])
# so we must capture the real implementation here, at import time.
_REAL_EXTRACT_IMAGES = server_module.extract_images
_REAL_STRIP_THINKING = server_module.strip_thinking
# Capture DEFAULT_THINKING at module load time, before the fixture overrides it to False.
_MODULE_DEFAULT_THINKING_AT_LOAD = server_module.DEFAULT_THINKING

import pytest
from fastapi.testclient import TestClient


@dataclass
class MockResponse:
    text: str
    token: int = 0
    logprobs: object = None
    prompt_tokens: int = 10
    generation_tokens: int = 0
    prompt_tps: float = 100.0
    generation_tps: float = 50.0
    peak_memory: float = 20.0
    finish_reason: str = None


def mock_stream_generate(model, processor, prompt, image=None, max_tokens=256, **kwargs):
    yield MockResponse("Hello", generation_tokens=1)
    yield MockResponse(" world", generation_tokens=2)
    yield MockResponse("!", generation_tokens=3, finish_reason="stop")


@pytest.fixture(autouse=True)
def setup():
    server_module.model = MagicMock()
    server_module.model.config = MagicMock()   # apply_chat_template(processor, model.config, ...)용
    server_module.processor = MagicMock()
    # apply_chat_template은 모듈 레벨 독립 함수 — server_module에서 직접 mock
    server_module.apply_chat_template = MagicMock(return_value="formatted prompt")
    server_module.model_id = "test-model"
    server_module.gpu_semaphore = asyncio.Semaphore(1)
    server_module.pending = 0
    server_module.MAX_QUEUE = 5
    server_module.DEFAULT_THINKING = False
    server_module.LOG_DIR = "/tmp/llm-test-logs"
    os.makedirs("/tmp/llm-test-logs", exist_ok=True)

    # run_inference()는 generate()를 호출 — 결과 mock (MockResponse 재사용으로 finish_reason=None 보장)
    mock_generate_result = MockResponse(text="Hello world!", prompt_tokens=10, generation_tokens=3)
    server_module.generate = MagicMock(return_value=mock_generate_result)
    # extract_images mock — 기본적으로 빈 리스트 반환 (이미지 없는 요청)
    server_module.extract_images = MagicMock(return_value=[])

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
        # apply_chat_template은 모듈 레벨 함수: apply_chat_template(processor, config, messages, num_images, chat_template_kwargs)
        assert server_module.apply_chat_template.called
        call_args = server_module.apply_chat_template.call_args
        # chat_template_kwargs 파라미터 (keyword 인자)
        template_kwargs = call_args.kwargs.get("chat_template_kwargs", {})
        assert template_kwargs.get("enable_thinking") is True

    def test_enable_thinking_default_false(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}],
        })
        assert resp.status_code == 200
        # DEFAULT_THINKING=False이면 enable_thinking=False가 chat_template_kwargs로 전달됨
        assert server_module.apply_chat_template.called
        call_args = server_module.apply_chat_template.call_args
        template_kwargs = call_args.kwargs.get("chat_template_kwargs", {})
        # enable_thinking이 False로 전달되거나 absent
        assert not template_kwargs.get("enable_thinking", False)

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
        assert server_module.apply_chat_template.called
        call_args = server_module.apply_chat_template.call_args
        template_kwargs = call_args.kwargs.get("chat_template_kwargs", {})
        assert template_kwargs.get("enable_thinking") is True


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


# ── extract_images() 유닛 테스트 ─────────────────────────────────────────────
# autouse fixture가 server_module.extract_images를 MagicMock으로 교체하므로
# 실제 함수는 _REAL_EXTRACT_IMAGES로 직접 호출한다.
class TestExtractImages:
    def test_base64_image_decoded_to_pil(self):
        """CORE-02: base64 data URI가 PIL Image로 변환된다"""
        # PNG 1×1 픽셀의 최소 유효 base64
        tiny_png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        url = f"data:image/png;base64,{tiny_png_b64}"
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
        ]

        # PIL mock이 설정되어 있으므로 Image.open 호출 여부와 반환값을 검사한다.
        fake_img = MagicMock()
        fake_img.convert.return_value = fake_img
        server_module.Image.open.return_value = fake_img
        server_module.Image.open.reset_mock()

        result = _REAL_EXTRACT_IMAGES(messages)

        # Image.open이 BytesIO 인수와 함께 호출되었다 (디코딩 성공 증거)
        assert server_module.Image.open.called
        call_arg = server_module.Image.open.call_args[0][0]
        assert isinstance(call_arg, io.BytesIO)

        # 결과 목록에 이미지가 1개 들어 있다
        assert len(result) == 1

    def test_url_image_downloaded(self):
        """CORE-02: HTTP URL 이미지가 urllib.request로 다운로드되어 PIL Image로 변환된다"""
        fake_img = MagicMock()
        fake_img.convert.return_value = fake_img
        server_module.Image.open.return_value = fake_img
        server_module.Image.open.reset_mock()

        # urllib.request.urlopen을 patch해서 실제 네트워크 요청을 막는다
        fake_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # 짧은 PNG-like 바이트
        fake_resp = MagicMock()
        fake_resp.__enter__ = MagicMock(return_value=fake_resp)
        fake_resp.__exit__ = MagicMock(return_value=False)
        fake_resp.read.return_value = fake_bytes

        url = "http://example.com/image.png"
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
        ]

        with patch("urllib.request.urlopen", return_value=fake_resp):
            result = _REAL_EXTRACT_IMAGES(messages)

        # Image.open이 호출되어 다운로드된 바이트를 처리했다
        assert server_module.Image.open.called
        # 결과 목록에 이미지가 1개 들어 있다
        assert len(result) == 1

    def test_large_image_resized_to_1120(self):
        """CORE-03: 이미지에 thumbnail((1120, 1120))이 적용된다 (1120px 리사이즈)"""
        tiny_png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        url = f"data:image/png;base64,{tiny_png_b64}"
        messages = [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
        ]

        fake_img = MagicMock()
        fake_img.convert.return_value = fake_img
        server_module.Image.open.return_value = fake_img

        _REAL_EXTRACT_IMAGES(messages)

        # thumbnail((1120, 1120))이 반드시 호출되어야 한다
        fake_img.thumbnail.assert_called_with((1120, 1120))


# ── strip_thinking() 유닛 테스트 ─────────────────────────────────────────────
class TestStripThinking:
    def test_strip_thinking_removes_think_block(self):
        """CORE-06: strip_thinking()이 <think>...</think> 블록을 제거한다"""
        input_text = "<think>내부 추론 과정</think>최종 답변"
        result = _REAL_STRIP_THINKING(input_text)
        assert result == "최종 답변"

    def test_strip_thinking_multiline(self):
        """CORE-06: strip_thinking()이 멀티라인 <think> 블록도 제거한다"""
        input_text = "<think>\n여러 줄\n추론\n</think>\n답변 내용"
        result = _REAL_STRIP_THINKING(input_text)
        assert result == "답변 내용"

    def test_strip_thinking_no_block(self):
        """CORE-06: <think> 블록이 없으면 원문 그대로 반환한다"""
        input_text = "일반 텍스트 응답"
        result = _REAL_STRIP_THINKING(input_text)
        assert result == "일반 텍스트 응답"


# ── preserve_thinking / DEFAULT_THINKING 통합 테스트 ─────────────────────────
class TestPreserveThinking:
    def test_default_thinking_is_true_at_module_level(self):
        """CORE-05: DEFAULT_THINKING 모듈 레벨 기본값이 True이다
        (autouse fixture가 False로 override하기 전의 실제 모듈 초기값 검증)"""
        assert _MODULE_DEFAULT_THINKING_AT_LOAD is True

    def test_preserve_thinking_false_strips_think_block(self, client):
        """CORE-06: preserve_thinking=False(기본)일 때 응답 content에서 <think> 블록이 제거된다"""
        # generate()가 <think> 블록을 포함한 텍스트를 반환하도록 설정
        thinking_response = MockResponse(
            text="<think>추론 과정</think>최종 답변",
            prompt_tokens=10,
            generation_tokens=5,
        )
        server_module.generate = MagicMock(return_value=thinking_response)

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "질문"}],
            # preserve_thinking은 명시하지 않으면 False (기본값)
        })

        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        # <think>...</think> 블록이 제거되고 최종 답변만 남아야 한다
        assert "<think>" not in content
        assert "최종 답변" in content

    def test_bad_image_returns_400(self, client):
        """CORE-02: 이미지 디코딩 실패 시 HTTP 400이 반환된다"""
        # extract_images가 ValueError를 raise하도록 설정 (autouse fixture의 mock을 override)
        server_module.extract_images = MagicMock(
            side_effect=ValueError("이미지 디코딩 실패: data:image/png;base64,INVALID")
        )

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,INVALID"}},
                        {"type": "text", "text": "이 이미지는?"},
                    ],
                }
            ],
        })

        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["type"] == "invalid_request_error"
