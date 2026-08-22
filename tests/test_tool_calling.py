"""
tool calling 테스트 (parse_tool_calls, parse_request tools 처리)
실행: .venv/bin/python -m pytest tests/test_tool_calling.py -v
"""

import json
import os
import sys
from unittest.mock import MagicMock

# mlx 계열 mock (GPU 없는 환경에서도 테스트 가능)
sys.modules.setdefault("mlx", MagicMock())
sys.modules.setdefault("mlx.core", MagicMock())
sys.modules.setdefault("mlx_vlm", MagicMock())
sys.modules.setdefault("mlx_vlm.prompt_utils", MagicMock())
sys.modules.setdefault("mlx_vlm.generate", MagicMock())
sys.modules.setdefault("mlx_vlm.vision_cache", MagicMock())
sys.modules.setdefault("mlx_vlm.apc", MagicMock())
sys.modules.setdefault("mlx_vlm.speculative", MagicMock())
_mock_pil = MagicMock()
sys.modules.setdefault("PIL", _mock_pil)
sys.modules.setdefault("PIL.Image", _mock_pil.Image)

import importlib.util

if "llm_api_server" in sys.modules:
    server_module = sys.modules["llm_api_server"]
else:
    _spec = importlib.util.spec_from_file_location(
        "llm_api_server",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "llm-api-server.py"),
    )
    server_module = importlib.util.module_from_spec(_spec)
    sys.modules["llm_api_server"] = server_module
    _spec.loader.exec_module(server_module)

parse_tool_calls = server_module.parse_tool_calls
parse_request = server_module.parse_request

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
        },
    },
}


class TestParseToolCalls:
    def test_no_tool_call_returns_text(self):
        content, calls = parse_tool_calls("그냥 답변", [WEATHER_TOOL])
        assert content == "그냥 답변"
        assert calls == []

    def test_single_call(self):
        text = "<tool_call>\n<function=get_weather>\n<parameter=city>\nSeoul\n</parameter>\n</function>\n</tool_call>"
        content, calls = parse_tool_calls(text, [WEATHER_TOOL])
        assert content is None
        assert len(calls) == 1
        tc = calls[0]
        assert tc["type"] == "function"
        assert tc["id"].startswith("call_")
        assert tc["function"]["name"] == "get_weather"
        # OpenAI 포맷: arguments는 JSON 문자열
        assert isinstance(tc["function"]["arguments"], str)
        assert json.loads(tc["function"]["arguments"]) == {"city": "Seoul"}

    def test_schema_type_casting(self):
        text = ("<tool_call>\n<function=get_weather>\n"
                "<parameter=city>\nSeoul\n</parameter>\n"
                "<parameter=days>\n3\n</parameter>\n"
                "</function>\n</tool_call>")
        _, calls = parse_tool_calls(text, [WEATHER_TOOL])
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"city": "Seoul", "days": 3}

    def test_multiline_string_value_kept_raw(self):
        text = ("<tool_call>\n<function=get_weather>\n"
                "<parameter=city>\nline1\nline2\n</parameter>\n"
                "</function>\n</tool_call>")
        _, calls = parse_tool_calls(text, [WEATHER_TOOL])
        assert json.loads(calls[0]["function"]["arguments"])["city"] == "line1\nline2"

    def test_leading_content_preserved(self):
        text = "날씨를 확인할게요.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nSeoul\n</parameter>\n</function>\n</tool_call>"
        content, calls = parse_tool_calls(text, [WEATHER_TOOL])
        assert content == "날씨를 확인할게요."
        assert len(calls) == 1

    def test_multiple_calls(self):
        block = "<tool_call>\n<function=get_weather>\n<parameter=city>\n{c}\n</parameter>\n</function>\n</tool_call>"
        text = block.format(c="Seoul") + "\n" + block.format(c="Busan")
        _, calls = parse_tool_calls(text, [WEATHER_TOOL])
        assert len(calls) == 2
        assert [json.loads(c["function"]["arguments"])["city"] for c in calls] == ["Seoul", "Busan"]
        assert calls[0]["id"] != calls[1]["id"]

    def test_malformed_block_ignored(self):
        content, calls = parse_tool_calls("<tool_call>깨진 블록</tool_call>", [WEATHER_TOOL])
        assert calls == []

    def test_no_schema_keeps_strings(self):
        text = "<tool_call>\n<function=unknown_fn>\n<parameter=x>\n42\n</parameter>\n</function>\n</tool_call>"
        _, calls = parse_tool_calls(text, None)
        assert json.loads(calls[0]["function"]["arguments"]) == {"x": "42"}


class TestCacheTrimBugRetry:
    """mlx-vlm ArraysCache.trim() 업스트림 버그 방어 경로"""

    TRIM_ERR = AttributeError("'ArraysCache' object has no attribute 'trim'")

    def _patch_cache(self, monkeypatch):
        monkeypatch.setattr(server_module, "prompt_cache_state", MagicMock())
        monkeypatch.setattr(server_module, "mx", MagicMock())

    def test_non_streaming_retries_on_trim_bug(self, monkeypatch):
        self._patch_cache(monkeypatch)
        inner = MagicMock(side_effect=[self.TRIM_ERR, ("답변", "stop", 10, 5)])
        monkeypatch.setattr(server_module, "_run_inference_inner", inner)
        assert server_module.run_inference({}) == ("답변", "stop", 10, 5)
        assert inner.call_count == 2

    def test_non_streaming_other_attribute_error_propagates(self, monkeypatch):
        self._patch_cache(monkeypatch)
        err = AttributeError("something else")
        monkeypatch.setattr(server_module, "_run_inference_inner", MagicMock(side_effect=err))
        import pytest
        with pytest.raises(AttributeError, match="something else"):
            server_module.run_inference({})

    def test_streaming_retries_before_first_yield(self, monkeypatch):
        self._patch_cache(monkeypatch)
        calls = {"n": 0}

        def fake_inner(params):
            calls["n"] += 1
            if calls["n"] == 1:
                raise self.TRIM_ERR
            yield "tok1"
            yield "tok2"

        monkeypatch.setattr(server_module, "_run_inference_streaming_inner", fake_inner)
        assert list(server_module.run_inference_streaming({})) == ["tok1", "tok2"]
        assert calls["n"] == 2

    def test_streaming_passthrough_when_no_error(self, monkeypatch):
        self._patch_cache(monkeypatch)

        def fake_inner(params):
            yield "a"
            yield "b"

        monkeypatch.setattr(server_module, "_run_inference_streaming_inner", fake_inner)
        assert list(server_module.run_inference_streaming({})) == ["a", "b"]


class TestStripThinkingWithToolCall:
    def test_thinking_on_direct_tool_call_kept(self):
        # thinking ON인데 모델이 <think> 없이 바로 tool call → 원문 유지
        text = "<tool_call>\n<function=f>\n</function>\n</tool_call>"
        assert server_module.strip_thinking(text, enable_thinking=True) == text

    def test_thinking_on_truncated_still_hidden(self):
        # 잘린 thinking (tool_call 없음)은 기존대로 숨김
        assert server_module.strip_thinking("생각 중인 내용...", enable_thinking=True) == ""

    def test_thinking_block_then_tool_call(self):
        text = "고민...</think>\n<tool_call>X</tool_call>"
        assert server_module.strip_thinking(text, enable_thinking=True) == "<tool_call>X</tool_call>"


class TestDeveloperRole:
    def test_developer_role_mapped_to_system(self):
        out = server_module.normalize_messages([
            {"role": "developer", "content": "지시문"},
            {"role": "user", "content": "질문"},
        ])
        assert [m["role"] for m in out] == ["system", "user"]


class TestParseRequestTools:
    def test_tools_passthrough(self):
        params = parse_request({"tools": [WEATHER_TOOL]})
        assert params["tools"] == [WEATHER_TOOL]

    def test_tool_choice_none_drops_tools(self):
        params = parse_request({"tools": [WEATHER_TOOL], "tool_choice": "none"})
        assert params["tools"] is None

    def test_no_tools_default(self):
        assert parse_request({})["tools"] is None
