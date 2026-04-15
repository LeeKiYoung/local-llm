#!/bin/bash
# Local LLM 자동 셋업 스크립트
#
# 사용법:
#   ./setup.sh              # 대화형 모델 선택 + 셋업
#   ./setup.sh --no-model   # 모델 다운로드 없이 환경만 셋업
#
# 요구사항:
#   - Apple Silicon Mac (M1/M2/M3/M4/M5)
#   - Python 3.10+
#   - 디스크 여유 (모델 크기에 따라 다름)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  Local LLM Setup (MLX)"
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
echo ""

# 4. 모델 선택
if [ "$1" != "--no-model" ]; then
  echo "--------------------------------------------"
  echo "  모델 선택"
  echo "--------------------------------------------"
  echo ""

  if [ "$TOTAL_GB" -ge 64 ]; then
    MEM_TAG="64GB+"
  elif [ "$TOTAL_GB" -ge 48 ]; then
    MEM_TAG="48GB"
  elif [ "$TOTAL_GB" -ge 32 ]; then
    MEM_TAG="32GB"
  elif [ "$TOTAL_GB" -ge 24 ]; then
    MEM_TAG="24GB"
  elif [ "$TOTAL_GB" -ge 16 ]; then
    MEM_TAG="16GB"
  else
    MEM_TAG="8GB"
  fi

  echo "  현재 메모리: ${TOTAL_GB}GB — 아래에서 적합한 모델을 선택하세요."
  echo ""
  echo "  ┌─────┬──────────────────────────┬────────┬───────────┬────────┬──────────────────────────────┐"
  echo "  │  #  │ 모델                     │ 메모리 │ 속도      │ 최소   │ 특징                         │"
  echo "  ├─────┼──────────────────────────┼────────┼───────────┼────────┼──────────────────────────────┤"
  echo "  │  1  │ Qwen3.5-35B-A3B    ⭐    │ ~20GB  │ 103 tok/s │ 24GB+  │ 한국어+코딩+비전 올라운더     │"
  echo "  │  2  │ Qwen3.5-9B               │ ~6GB   │ 40+ tok/s │ 16GB+  │ 가볍고 빠름                   │"
  echo "  │  3  │ Qwen3.5-27B              │ ~17GB  │ 15  tok/s │ 24GB+  │ Dense, 코딩 벤치마크 최강     │"
  echo "  │  4  │ Qwen3-Coder-Next-80B     │ ~15GB  │ 25+ tok/s │ 24GB+  │ 코딩 에이전트 특화            │"
  echo "  │  5  │ 직접 입력                │ -      │ -         │ -      │ Hugging Face 모델 ID         │"
  echo "  └─────┴──────────────────────────┴────────┴───────────┴────────┴──────────────────────────────┘"
  echo ""

  # 메모리 기반 추천 표시
  if [ "$TOTAL_GB" -lt 16 ]; then
    echo "  💡 ${TOTAL_GB}GB 메모리 — [2] Qwen3.5-9B를 권장합니다."
  elif [ "$TOTAL_GB" -lt 24 ]; then
    echo "  💡 ${TOTAL_GB}GB 메모리 — [2] Qwen3.5-9B를 권장합니다."
  elif [ "$TOTAL_GB" -lt 48 ]; then
    echo "  💡 ${TOTAL_GB}GB 메모리 — [1] Qwen3.5-35B-A3B를 권장합니다."
  else
    echo "  💡 ${TOTAL_GB}GB 메모리 — 모든 모델 실행 가능! [1] 추천."
  fi
  echo ""

  read -p "  선택 [1-5] (기본: 1): " MODEL_CHOICE
  MODEL_CHOICE=${MODEL_CHOICE:-1}

  case "$MODEL_CHOICE" in
    1)
      MODEL="mlx-community/Qwen3.5-35B-A3B-4bit"
      MODEL_NAME="Qwen3.5-35B-A3B-4bit"
      MODEL_SIZE="~19GB"
      ;;
    2)
      MODEL="mlx-community/Qwen3.5-9B-4bit"
      MODEL_NAME="Qwen3.5-9B-4bit"
      MODEL_SIZE="~6GB"
      ;;
    3)
      MODEL="mlx-community/Qwen3.5-27B-4bit"
      MODEL_NAME="Qwen3.5-27B-4bit"
      MODEL_SIZE="~17GB"
      ;;
    4)
      MODEL="mlx-community/Qwen3-Coder-Next-80B-A3B-4bit"
      MODEL_NAME="Qwen3-Coder-Next-4bit"
      MODEL_SIZE="~15GB"
      ;;
    5)
      read -p "  Hugging Face 모델 ID: " MODEL
      MODEL_NAME="$MODEL"
      MODEL_SIZE="알 수 없음"
      ;;
    *)
      echo "❌ 잘못된 선택입니다."
      exit 1
      ;;
  esac

  echo ""
  echo "  선택: $MODEL_NAME ($MODEL_SIZE)"
fi

# 5. venv 생성
echo ""
if [ -d "$SCRIPT_DIR/.venv" ]; then
  echo "📦 기존 가상환경 발견 — 건너뜀"
else
  echo "📦 가상환경 생성 중..."
  "$PYTHON" -m venv "$SCRIPT_DIR/.venv"
  echo "✅ 가상환경 생성 완료"
fi

# 6. mlx-lm 설치
echo ""
echo "📦 mlx-lm 설치 중..."
"$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip -q
"$SCRIPT_DIR/.venv/bin/pip" install "git+https://github.com/ml-explore/mlx-lm.git" -q
echo "✅ mlx-lm 설치 완료 (GitHub 최신 — gemma4 지원 포함)"

echo "📦 FastAPI + uvicorn 설치 중..."
"$SCRIPT_DIR/.venv/bin/pip" install fastapi uvicorn -q
echo "✅ FastAPI + uvicorn 설치 완료"

# 7. HuggingFace 캐시 경로 안내
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
echo ""
echo "📂 모델 캐시 경로: $HF_CACHE"
if [ -n "$HF_HOME" ]; then
  echo "   (HF_HOME 환경변수 사용: $HF_HOME)"
else
  echo "   (기본 경로. 변경하려면 HF_HOME 환경변수 설정)"
  echo "   예: export HF_HOME=/Volumes/외장SSD/.huggingface"
fi

# 8. 선택한 모델을 스크립트에 반영
save_model_config() {
  # llm-chat.sh와 llm-server.sh의 MODEL 변수를 업데이트
  for script in "$SCRIPT_DIR/llm-chat.sh" "$SCRIPT_DIR/llm-server.sh"; do
    if [ -f "$script" ]; then
      sed -i '' "s|^MODEL=.*|MODEL=\"$MODEL\"|" "$script"
    fi
  done
}

# 9. 모델 다운로드
if [ "$1" = "--no-model" ]; then
  echo ""
  echo "⏭️  모델 다운로드 건너뜀 (--no-model)"
  echo "   나중에 다운로드: ./llm-chat.sh (첫 실행 시 자동 다운로드)"
else
  save_model_config
  echo ""
  echo "📥 모델 다운로드 중... ($MODEL_SIZE, 시간이 걸릴 수 있습니다)"
  "$SCRIPT_DIR/.venv/bin/mlx_lm.generate" \
    --model "$MODEL" \
    --prompt "Hello" \
    --max-tokens 1 2>&1 | tail -3
  echo "✅ 모델 다운로드 완료"
fi

# 10. 완료
echo ""
echo "============================================"
echo "  셋업 완료! — $MODEL_NAME"
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
