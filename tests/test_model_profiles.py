"""프로필명 → 모델 ID 일치 회귀 테스트 (#21)

setup.sh, llm-server.sh, README.md에서 동일 프로필명이 동일 모델을 가리키는지 검증.
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

TEXT_MODEL = "Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2"
VLM_MODEL = "Jiunsong/supergemma4-26b-abliterated-multimodal-mlx-4bit"


def _server_profile_model(profile):
    """llm-server.sh의 case 분기에서 profile이 매핑하는 MODEL 값을 추출"""
    src = (ROOT / "llm-server.sh").read_text()
    # case 패턴 라인(예: "supergemma4|supergemma4-text)")을 찾고 그 아래 MODEL= 라인을 읽는다
    lines = src.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^[\w|4-]+\)$", stripped) and profile in stripped.rstrip(")").split("|"):
            for follow in lines[i + 1:i + 5]:
                m = re.search(r'MODEL="([^"]+)"', follow)
                if m:
                    return m.group(1)
    return None


def test_supergemma4_matches_setup_choice():
    """setup.sh 선택 2와 ./llm-server.sh supergemma4가 같은 모델을 가리켜야 함"""
    setup = (ROOT / "setup.sh").read_text()
    assert TEXT_MODEL in setup
    assert _server_profile_model("supergemma4") == TEXT_MODEL


def test_supergemma4_text_alias():
    assert _server_profile_model("supergemma4-text") == TEXT_MODEL


def test_supergemma4_vlm_profile():
    assert _server_profile_model("supergemma4-vlm") == VLM_MODEL


def test_readme_documents_both_profiles():
    readme = (ROOT / "README.md").read_text()
    assert "supergemma4-vlm" in readme
    assert TEXT_MODEL in readme
    assert VLM_MODEL in readme
