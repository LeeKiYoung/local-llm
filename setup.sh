#!/bin/bash
# Local LLM 자동 셋업 스크립트
#
# 사용법:
#   ./setup.sh              # 전체 셋업 (venv + mlx-lm + 모델 다운로드)
#   ./setup.sh --no-model   # 모델 다운로드 없이 환경만 셋업
#
# 요구사항:
#   - Apple Silicon Mac (M1/M2/M3/M4/M5)
#   - Python 3.10+
#   - ~20GB 디스크 여유 (모델 다운로드)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="mlx-community/Qwen3.5-35B-A3B-4bit"

echo "============================================"
echo "  Local LLM Setup — Qwen3.5-35B-A3B (MLX)"
echo "============================================"
echo ""

# 1. Apple Silicon 체크
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
  echo "❌ Apple Silicon(arm64)이 필요합니다. 현재: $ARCH"
  exit 1
fi
echo "✅ Apple Silicon 확인 ($ARCH)"

# 2. Python 버전 체크
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3; do
  if command -v "$cmd" &>/dev/null; then
    VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
    MAJOR=$(echo "$VER" | cut -d. -f1)
    MINOR=$(echo "$VER" | cut -d. -f2)
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
      PYTHON="$cmd"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "❌ Python 3.10+ 이 필요합니다."
  echo "   설치: brew install python@3.12"
  exit 1
fi
echo "✅ Python $VER ($PYTHON)"

# 3. 메모리 체크
TOTAL_MEM=$(sysctl -n hw.memsize 2>/dev/null)
TOTAL_GB=$((TOTAL_MEM / 1073741824))
echo "✅ 메모리 ${TOTAL_GB}GB"

if [ "$TOTAL_GB" -lt 16 ]; then
  echo "⚠️  16GB 이상 권장. 현재 ${TOTAL_GB}GB로 모델 실행이 어려울 수 있습니다."
fi

# 4. venv 생성
echo ""
if [ -d "$SCRIPT_DIR/.venv" ]; then
  echo "📦 기존 가상환경 발견 — 건너뜀"
else
  echo "📦 가상환경 생성 중..."
  "$PYTHON" -m venv "$SCRIPT_DIR/.venv"
  echo "✅ 가상환경 생성 완료"
fi

# 5. mlx-lm 설치
echo ""
echo "📦 mlx-lm 설치 중..."
"$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip -q
"$SCRIPT_DIR/.venv/bin/pip" install mlx-lm -q
echo "✅ mlx-lm 설치 완료"

# 6. HuggingFace 캐시 경로 안내
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
echo ""
echo "📂 모델 캐시 경로: $HF_CACHE"
if [ -n "$HF_HOME" ]; then
  echo "   (HF_HOME 환경변수 사용: $HF_HOME)"
else
  echo "   (기본 경로. 변경하려면 HF_HOME 환경변수 설정)"
  echo "   예: export HF_HOME=/Volumes/외장SSD/.huggingface"
fi

# 7. 모델 다운로드
if [ "$1" = "--no-model" ]; then
  echo ""
  echo "⏭️  모델 다운로드 건너뜀 (--no-model)"
  echo "   나중에 다운로드: ./llm-chat.sh (첫 실행 시 자동 다운로드)"
else
  echo ""
  echo "📥 모델 다운로드 중... (~19GB, 시간이 걸릴 수 있습니다)"
  "$SCRIPT_DIR/.venv/bin/mlx_lm.generate" \
    --model "$MODEL" \
    --prompt "Hello" \
    --max-tokens 1 2>&1 | tail -3
  echo "✅ 모델 다운로드 완료"
fi

# 8. 완료
echo ""
echo "============================================"
echo "  셋업 완료!"
echo "============================================"
echo ""
echo "  채팅 시작:    ./llm-chat.sh"
echo "  1M 컨텍스트:  ./llm-chat.sh 1m"
echo "  API 서버:     ./llm-server.sh"
echo "  상태 확인:    ./llm-chat.sh status"
echo ""
echo "  Alias 설정 (선택):"
echo "  echo 'alias llm-chat=\"$SCRIPT_DIR/llm-chat.sh\"' >> ~/.zshrc"
echo "  echo 'alias llm-server=\"$SCRIPT_DIR/llm-server.sh\"' >> ~/.zshrc"
echo ""
