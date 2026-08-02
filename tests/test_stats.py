"""
llm-api-server.py 모니터링 대시보드 (aggregate_stats / GET /api/stats / GET /dashboard) 테스트
실행: .venv/bin/python -m pytest tests/test_stats.py -v
"""

import json
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# mlx.core를 mock으로 대체 (GPU 없는 환경에서도 테스트 가능)
mock_mx = MagicMock()
mock_mx.metal.clear_cache = MagicMock()
sys.modules["mlx"] = MagicMock()
sys.modules["mlx.core"] = mock_mx

# mlx_vlm mock
mock_mlx_vlm = MagicMock()
mock_mlx_vlm_prompt_utils = MagicMock()
mock_mlx_vlm_generate = MagicMock()
mock_mlx_vlm_vision_cache = MagicMock()
mock_mlx_vlm_apc = MagicMock()
sys.modules["mlx_vlm"] = mock_mlx_vlm
sys.modules["mlx_vlm.prompt_utils"] = mock_mlx_vlm_prompt_utils
sys.modules["mlx_vlm.generate"] = mock_mlx_vlm_generate
sys.modules["mlx_vlm.vision_cache"] = mock_mlx_vlm_vision_cache
sys.modules["mlx_vlm.apc"] = mock_mlx_vlm_apc

# PIL mock (Pillow 없는 환경에서도 테스트 가능)
mock_pil = MagicMock()
sys.modules["PIL"] = mock_pil
sys.modules["PIL.Image"] = mock_pil.Image

import importlib.util

# 하이픈 파일명이라 importlib으로 로드 (이미 다른 테스트 모듈이 로드했다면 재사용)
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

import pytest
from fastapi.testclient import TestClient


def _write_log(log_dir, date, entries):
    """date: datetime.date, entries: list[dict] — 각 dict가 한 줄의 JSON이 된다"""
    path = os.path.join(log_dir, f"{date.strftime('%Y-%m-%d')}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            if isinstance(e, str):
                f.write(e + "\n")
            else:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return path


def _entry(**overrides):
    base = {
        "timestamp": datetime.now().isoformat(),
        "ip": "127.0.0.1",
        "enable_thinking": False,
        "stream": False,
        "duration_ms": 1000,
        "prompt_preview": "hello",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "finish_reason": "stop",
        "content_preview": "world",
    }
    base.update(overrides)
    return base


# ── aggregate_stats() 유닛 테스트 ────────────────────────────────────────────
class TestAggregateStats:
    def test_empty_dir_returns_full_schema_with_zeros(self, tmp_path):
        result = server_module.aggregate_stats(str(tmp_path), days=7)
        assert result["total_requests"] == 0
        assert result["daily"] == []
        assert result["tokens"] == {"prompt": 0, "completion": 0, "total": 0}
        assert result["duration_ms"] == {"avg": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
        assert result["tokens_per_sec"] == {"avg": 0, "p50": 0, "max": 0}
        assert result["thinking"] == {"on": 0, "off": 0}
        assert result["streaming"] == {"on": 0, "off": 0}
        assert result["finish_reason"] == {}
        assert result["recent"] == []

    def test_nonexistent_dir_returns_full_schema_with_zeros(self, tmp_path):
        missing = str(tmp_path / "does-not-exist")
        result = server_module.aggregate_stats(missing, days=7)
        assert result["total_requests"] == 0
        assert result["daily"] == []

    def test_happy_path_across_two_day_files(self, tmp_path):
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        _write_log(str(tmp_path), yesterday, [_entry(duration_ms=1000), _entry(duration_ms=2000)])
        _write_log(str(tmp_path), today, [_entry(duration_ms=500)])

        result = server_module.aggregate_stats(str(tmp_path), days=7)
        assert result["total_requests"] == 3
        assert len(result["daily"]) == 2
        # ascending by date
        assert result["daily"][0]["date"] == yesterday.strftime("%Y-%m-%d")
        assert result["daily"][1]["date"] == today.strftime("%Y-%m-%d")
        assert result["daily"][0]["requests"] == 2
        assert result["daily"][1]["requests"] == 1
        assert result["tokens"]["prompt"] == 30
        assert result["tokens"]["completion"] == 60
        assert result["tokens"]["total"] == 90
        assert result["thinking"] == {"on": 0, "off": 3}
        assert result["streaming"] == {"on": 0, "off": 3}
        assert result["finish_reason"] == {"stop": 3}
        assert len(result["recent"]) == 3

    def test_malformed_line_skipped_surrounding_lines_counted(self, tmp_path):
        today = datetime.now().date()
        path = _write_log(str(tmp_path), today, [])
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_entry()) + "\n")
            f.write("not valid json {{{\n")
            f.write(json.dumps(_entry()) + "\n")
            f.write('{"timestamp": "2024-01-01T00:00:00", "incomplete": tr\n')  # torn trailing line

        result = server_module.aggregate_stats(str(tmp_path), days=7)
        assert result["total_requests"] == 2

    def test_missing_usage_and_duration_ms_counts_as_zeroed_request(self, tmp_path):
        today = datetime.now().date()
        _write_log(str(tmp_path), today, [{"timestamp": datetime.now().isoformat(), "ip": "1.2.3.4"}])
        result = server_module.aggregate_stats(str(tmp_path), days=7)
        assert result["total_requests"] == 1
        assert result["tokens"] == {"prompt": 0, "completion": 0, "total": 0}
        assert result["duration_ms"]["avg"] == 0
        assert result["duration_ms"]["max"] == 0

    def test_duration_ms_zero_yields_zero_tokens_per_sec_no_zerodivision(self, tmp_path):
        today = datetime.now().date()
        _write_log(str(tmp_path), today, [_entry(duration_ms=0, usage={
            "prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15
        })])
        result = server_module.aggregate_stats(str(tmp_path), days=7)
        assert result["total_requests"] == 1
        assert result["tokens_per_sec"]["avg"] == 0.0
        assert result["tokens_per_sec"]["max"] == 0.0

    def test_file_outside_days_window_excluded(self, tmp_path):
        today = datetime.now().date()
        old = today - timedelta(days=10)
        _write_log(str(tmp_path), old, [_entry()])
        _write_log(str(tmp_path), today, [_entry()])

        result = server_module.aggregate_stats(str(tmp_path), days=7)
        assert result["total_requests"] == 1
        assert len(result["daily"]) == 1
        assert result["daily"][0]["date"] == today.strftime("%Y-%m-%d")

    @pytest.mark.parametrize("days", [0, -1, 999, "abc", None])
    def test_days_clamping_never_raises(self, tmp_path, days):
        today = datetime.now().date()
        _write_log(str(tmp_path), today, [_entry()])
        result = server_module.aggregate_stats(str(tmp_path), days=days)
        assert result["total_requests"] >= 0
        assert isinstance(result["daily"], list)

    def test_recent_capped_at_limit_newest_first(self, tmp_path):
        today = datetime.now().date()
        entries = [
            _entry(timestamp=(datetime.now() - timedelta(minutes=i)).isoformat(), prompt_preview=f"p{i}")
            for i in range(server_module.RECENT_REQUESTS_LIMIT + 10)
        ]
        _write_log(str(tmp_path), today, entries)
        result = server_module.aggregate_stats(str(tmp_path), days=7)
        assert len(result["recent"]) == server_module.RECENT_REQUESTS_LIMIT
        # newest first: p0 has the newest timestamp (i=0 → now)
        assert result["recent"][0]["prompt_preview"] == "p0"


# ── GET /api/stats + GET /dashboard 스모크 테스트 ────────────────────────────
@pytest.fixture
def client():
    return TestClient(server_module.app)


@pytest.fixture
def stats_log_dir(tmp_path):
    """server_module.LOG_DIR을 tmp_path로 임시 override, 테스트 후 원복"""
    original = server_module.LOG_DIR
    server_module.LOG_DIR = str(tmp_path)
    yield tmp_path
    server_module.LOG_DIR = original


class TestApiStatsEndpoint:
    def test_returns_200_with_expected_keys(self, client, stats_log_dir):
        today = datetime.now().date()
        _write_log(str(stats_log_dir), today, [_entry()])
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_requests" in data
        assert "daily" in data
        assert "tokens" in data
        assert "recent" in data

    def test_days_param_narrows_window(self, client, stats_log_dir):
        today = datetime.now().date()
        old = today - timedelta(days=10)
        _write_log(str(stats_log_dir), old, [_entry()])
        _write_log(str(stats_log_dir), today, [_entry()])

        resp30 = client.get("/api/stats", params={"days": 30})
        resp1 = client.get("/api/stats", params={"days": 1})
        assert resp30.status_code == 200
        assert resp1.status_code == 200
        assert resp30.json()["total_requests"] >= resp1.json()["total_requests"]
        assert resp1.json()["total_requests"] == 1

    def test_days_zero_and_large_clamped_not_500(self, client, stats_log_dir):
        today = datetime.now().date()
        _write_log(str(stats_log_dir), today, [_entry()])
        assert client.get("/api/stats", params={"days": 0}).status_code == 200
        assert client.get("/api/stats", params={"days": 999}).status_code == 200

    def test_days_non_int_returns_422(self, client, stats_log_dir):
        resp = client.get("/api/stats", params={"days": "abc"})
        assert resp.status_code == 422


class TestDashboardEndpoint:
    def test_returns_html(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")

    def test_no_cdn_references_only_api_stats(self, client):
        body = client.get("/dashboard").text
        import re
        # src=/href= 속성 값이 http://, https://, // 로 시작하면 CDN/원격 참조
        matches = re.findall(r'(?:src|href)\s*=\s*["\'](https?://|//)', body)
        assert matches == []
        assert "/api/stats" in body
