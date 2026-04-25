"""
llm-proxy.py 순수 함수 유닛 테스트
실행: .venv/bin/python -m pytest test_proxy_units.py -v
"""

import importlib.util
import json
import os
import sys

# llm-proxy.py 로드 (하이픈 파일명)
_spec = importlib.util.spec_from_file_location(
    "llm_proxy",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "llm-proxy.py"),
)
proxy_module = importlib.util.module_from_spec(_spec)
sys.modules["llm_proxy"] = proxy_module
_spec.loader.exec_module(proxy_module)

import pytest


# ── strip_thinking() 유닛 테스트 ─────────────────────────────────────────────
class TestProxyStripThinking:
    def test_strip_removes_think_from_content(self):
        """content의 <think>...</think> 블록 제거"""
        resp = {
            "choices": [
                {"message": {"content": "<think>추론 과정</think>최종 답변", "role": "assistant"}}
            ]
        }
        result = proxy_module.strip_thinking(resp)
        assert result["choices"][0]["message"]["content"] == "최종 답변"
        assert "<think>" not in result["choices"][0]["message"]["content"]

    def test_empty_choices_returns_unchanged(self):
        """choices가 빈 리스트면 그대로 반환"""
        resp = {"choices": []}
        result = proxy_module.strip_thinking(resp)
        assert result == {"choices": []}

    def test_reasoning_extracted_when_content_empty(self):
        """content가 비어있고 reasoning에 내용이 있을 때 reasoning에서 답변 추출"""
        resp = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning": "<think>내부 추론</think>실제 답변",
                        "role": "assistant",
                    }
                }
            ]
        }
        result = proxy_module.strip_thinking(resp)
        content = result["choices"][0]["message"]["content"]
        assert "실제 답변" in content

    def test_reasoning_removed_from_message(self):
        """strip 후 reasoning 키가 message에서 제거된다"""
        resp = {
            "choices": [
                {
                    "message": {
                        "content": "일반 답변",
                        "reasoning": "추론 과정",
                        "role": "assistant",
                    }
                }
            ]
        }
        result = proxy_module.strip_thinking(resp)
        assert "reasoning" not in result["choices"][0]["message"]


# ── log_entry() 유닛 테스트 ──────────────────────────────────────────────────
class TestProxyLogEntry:
    def test_log_entry_writes_jsonl(self, tmp_path):
        """log_entry()가 LOG_DIR에 JSONL 파일을 생성하고 항목을 기록한다"""
        original_log_dir = proxy_module.LOG_DIR
        proxy_module.LOG_DIR = str(tmp_path)
        try:
            entry = {
                "timestamp": "2026-04-25T00:00:00",
                "ip": "127.0.0.1",
                "duration_ms": 100,
                "prompt_preview": "테스트 프롬프트",
                "strip_think": True,
                "response": {"usage": {"total_tokens": 42}},
            }
            proxy_module.log_entry(entry)

            log_files = list(tmp_path.glob("*.jsonl"))
            assert len(log_files) == 1

            lines = log_files[0].read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 1

            saved = json.loads(lines[0])
            assert saved["ip"] == "127.0.0.1"
            assert saved["duration_ms"] == 100
        finally:
            proxy_module.LOG_DIR = original_log_dir
