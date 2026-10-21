"""
/v1/systemone 판단 엔드포인트 테스트 (요청 검증, 선택지 확률, llm/kev 엔진)
실행: .venv/bin/python -m pytest tests/test_decisions.py -v
"""

import asyncio
import json
import math
import os
import sys
from dataclasses import dataclass
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

import pytest
from fastapi.testclient import TestClient

# 토큰 id: "A"=10 " A"=11, "B"=20, "C"=30 (가짜 토크나이저)
_TOKENS = {"A": [10], " A": [11], "B": [20], " B": [20, 99], "C": [30], " C": [31]}


class _Scalar:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


class FakeLogprobs:
    """mx.array 대신 — 인덱싱 결과가 .item()을 지원하는 어휘 로그확률 벡터"""

    def __init__(self, probs, floor=1e-9):
        self.probs, self.floor = probs, floor

    def __getitem__(self, i):
        return _Scalar(math.log(self.probs.get(i, self.floor)))


@dataclass
class FirstToken:
    logprobs: object
    prompt_tokens: int = 42


def _generator_yielding(logprobs, calls):
    def fake_stream_generate(model, processor, prompt, **kwargs):
        calls.append({"prompt": prompt, **kwargs})
        yield FirstToken(logprobs)
    return fake_stream_generate


@pytest.fixture(autouse=True)
def setup():
    tokenizer = MagicMock()
    tokenizer.encode = lambda text, add_special_tokens=False: _TOKENS.get(text, [1, 2])
    server_module.processor = MagicMock(tokenizer=tokenizer)
    server_module.model = MagicMock()
    server_module.model_id = "test-model"
    server_module.apply_chat_template = MagicMock(side_effect=lambda p, c, messages, **kw: json.dumps(messages, ensure_ascii=False))
    server_module.gpu_semaphore = asyncio.Semaphore(1)
    server_module.pending = 0
    server_module.MAX_QUEUE = 5
    server_module.PREFILL_STEP_SIZE = 512
    server_module.DECISION_ENGINE = "llm"
    server_module.DECISION_TEMPERATURE = 1.0
    server_module.KEV_URL = "http://127.0.0.1:8009"
    server_module.LOG_DIR = "/tmp/llm-test-logs"
    os.makedirs("/tmp/llm-test-logs", exist_ok=True)
    yield
    server_module.DECISION_ENGINE = "off"


@pytest.fixture
def client():
    return TestClient(server_module.app)


class TestParseDecisionRequest:
    def test_choice_dict_criteria(self):
        p = server_module.parse_decision_request({
            "state": {"msg": "hi"},
            "questions": {"tone": {"type": "choice", "instructions": "Tone?",
                                   "criteria": {"calm": "neutral", "angry": "upset"}}},
        })
        assert p["questions"]["tone"]["options"] == [("calm", "neutral"), ("angry", "upset")]
        assert '"msg": "hi"' in p["state"]

    def test_choice_none_description_uses_key(self):
        # TypeSafe SDK의 Choice(criteria={"calm": None, ...}) 형식
        p = server_module.parse_decision_request({
            "state": "x", "questions": {"tone": {"type": "choice", "criteria": {"calm": None, "angry": None}}},
        })
        assert p["questions"]["tone"]["options"] == [("calm", "calm"), ("angry", "angry")]
        prompt = server_module._decision_messages("x", p["questions"]["tone"])[1]["content"]
        assert "A) calm\nB) angry" in prompt and "None" not in prompt

    def test_noul_has_yes_no_and_appends_criteria(self):
        p = server_module.parse_decision_request({
            "state": "x", "questions": {"q": {"type": "noul", "instructions": "Billing?", "criteria": "mentions money"}},
        })
        q = p["questions"]["q"]
        assert q["options"] == [("yes", "yes"), ("no", "no")]
        assert "mentions money" in q["instructions"]

    def test_score_levels_are_indexed(self):
        p = server_module.parse_decision_request({
            "state": "x", "questions": {"u": {"type": "score", "criteria": ["low", "mid", "high"]}},
        })
        assert p["questions"]["u"]["options"] == [("0", "low"), ("1", "mid"), ("2", "high")]

    @pytest.mark.parametrize("body, needle", [
        ({"questions": {"q": {"type": "noul"}}}, "state"),
        ({"state": "x", "questions": {}}, "non-empty"),
        ({"state": "x", "questions": {"q": {"type": "box"}}}, "unsupported"),
        ({"state": "x", "questions": {"q": {"type": "choice"}}}, "criteria"),
        ({"state": "x", "questions": {"q": {"type": "score", "criteria": ["only"]}}}, "2+"),
        ({"state": "x", "questions": {"q": {"type": "noul"}}, "assets": {"a": {}}}, "assets"),
    ])
    def test_invalid_requests(self, body, needle):
        with pytest.raises(ValueError, match=needle):
            server_module.parse_decision_request(body)


class TestOptionProbs:
    def test_variant_tokens_are_summed_and_renormalized(self):
        # A = "A"(0.3) + " A"(0.1) = 0.4, B = 0.1 → 0.8 / 0.2, coverage 0.5
        lp = FakeLogprobs({10: 0.3, 11: 0.1, 20: 0.1})
        probs, coverage = server_module.option_probs(lp, [[10, 11], [20]])
        assert probs == pytest.approx([0.8, 0.2])
        assert coverage == pytest.approx(0.5)

    def test_temperature_flattens(self):
        lp = FakeLogprobs({10: 0.8, 20: 0.2})
        sharp, _ = server_module.option_probs(lp, [[10], [20]], temperature=1.0)
        flat, _ = server_module.option_probs(lp, [[10], [20]], temperature=2.0)
        assert flat[0] < sharp[0] and flat[0] > 0.5

    def test_letter_token_ids_skip_multi_token_variants(self):
        tok = server_module.processor.tokenizer
        assert server_module._letter_token_ids(tok, 3) == [[10, 11], [20], [30, 31]]


class TestSystemOneEndpoint:
    def test_engine_off_returns_404(self, client):
        server_module.DECISION_ENGINE = "off"
        r = client.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "noul"}}})
        assert r.status_code == 404
        assert "--decision-engine" in r.json()["detail"]

    def test_invalid_request_returns_422(self, client):
        r = client.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "box"}}})
        assert r.status_code == 422

    def test_llm_engine_answers_all_question_types(self, client):
        calls = []
        server_module.stream_generate = _generator_yielding(FakeLogprobs({10: 0.6, 20: 0.3, 30: 0.1}), calls)
        r = client.post("/v1/systemone", json={
            "state": {"request": "PR 리뷰만 해줘", "command": "gh pr merge 250"},
            "questions": {
                "requested": {"type": "noul", "instructions": "Did the user ask for this?"},
                "action": {"type": "choice", "instructions": "How to handle?",
                           "criteria": {"run": "run", "ask": "ask first", "reply": "answer in words"}},
                "risk": {"type": "score", "criteria": ["low", "mid", "high"]},
            },
        })
        assert r.status_code == 200
        body = r.json()
        a = body["answers"]
        assert a["requested"] == {"type": "noul", "noul": pytest.approx(0.6667, abs=1e-3), "coverage": pytest.approx(0.9)}
        assert a["action"]["choice"] == "run"
        assert a["action"]["probabilities"] == pytest.approx({"run": 0.6, "ask": 0.3, "reply": 0.1}, abs=1e-3)
        assert a["risk"]["score"] == pytest.approx(0.3 + 0.2, abs=1e-3)
        assert a["risk"]["legend"] == {"0": "low", "1": "mid", "2": "high"}
        assert body["usage"] == {"input_tokens": 126, "output_tokens": 0}
        # 질문마다 프리필 1회, 생성은 1토큰만, speculative(draft) 없이
        assert len(calls) == 3
        assert all(c["max_tokens"] == 1 and "draft_model" not in c for c in calls)
        assert all(c["logits_processors"] == [server_module._logits_to_float32] for c in calls)
        assert "PR 리뷰만 해줘" in calls[0]["prompt"] and "A) yes" in calls[0]["prompt"]

    def test_queue_full_returns_429(self, client):
        server_module.pending = 5
        r = client.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "noul"}}})
        assert r.status_code == 429

    def test_kev_engine_forwards_request_and_status(self, client, monkeypatch):
        server_module.DECISION_ENGINE = "kev"
        seen = {}

        def fake_post(url, body, headers, timeout=180):
            seen.update(url=url, body=body, headers=headers)
            return 200, {"model": "kev-latest", "answers": {"q": {"type": "noul", "noul": 0.1}}}

        monkeypatch.setattr(server_module, "_post_json", fake_post)
        body = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}}
        r = client.post("/v1/systemone", json=body, headers={"Authorization": "Bearer k"})
        assert r.status_code == 200 and r.json()["answers"]["q"]["noul"] == 0.1
        assert seen == {"url": "http://127.0.0.1:8009/v1/systemone", "body": body,
                        "headers": {"Authorization": "Bearer k"}}

    def test_kev_engine_unreachable_returns_502(self, client, monkeypatch):
        server_module.DECISION_ENGINE = "kev"

        def refuse(*a, **kw):
            raise server_module.urllib.error.URLError("connection refused")

        monkeypatch.setattr(server_module, "_post_json", refuse)
        r = client.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "noul"}}})
        assert r.status_code == 502
        assert "decision-server.sh" in r.json()["detail"]
