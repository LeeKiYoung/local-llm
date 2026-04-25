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
    VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) || continue
    MAJOR=$(echo "$VER" | cut -d. -f1)
    MINOR=$(echo "$VER" | cut -d. -f2)
    if [ -n "$MAJOR" ] && [ -n "$MINOR" ] && [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
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

  echo "  현재 메모리: ${TOTAL_GB}GB — 아래에서 적합한 모델을 선택하세요."
  echo ""
  echo "  ┌─────┬──────────────────────────┬────────┬───────────┬────────┬──────────────────────────────────┐"
  echo "  │  #  │ 모델                     │ 메모리 │ 속도      │ 최소   │ 특징                             │"
  echo "  ├─────┼──────────────────────────┼────────┼───────────┼────────┼──────────────────────────────────┤"
  echo "  │  1  │ Qwen3.6-27B-6bit   ⭐    │ ~23GB  │ -         │ 24GB+  │ VLM, 텍스트+이미지, thinking     │"
  echo "  │  2  │ SuperGemma4-26B          │ ~16GB  │ -         │ 24GB+  │ 무검열 보조 모델 (텍스트 전용)   │"
  echo "  │  3  │ 직접 입력                │ -      │ -         │ -      │ Hugging Face 모델 ID             │"
  echo "  └─────┴──────────────────────────┴────────┴───────────┴────────┴──────────────────────────────────┘"
  echo ""

  # 메모리 기반 추천 표시
  if [ "$TOTAL_GB" -lt 24 ]; then
    echo "  💡 ${TOTAL_GB}GB 메모리 — 최소 24GB가 필요합니다. 직접 입력([3])으로 경량 모델을 선택하세요."
  else
    echo "  💡 ${TOTAL_GB}GB 메모리 — [1] Qwen3.6-27B-6bit를 권장합니다."
  fi
  echo ""

  read -p "  선택 [1-3] (기본: 1): " MODEL_CHOICE
  MODEL_CHOICE=${MODEL_CHOICE:-1}

  case "$MODEL_CHOICE" in
    1)
      MODEL="mlx-community/Qwen3.6-27B-6bit"
      MODEL_NAME="Qwen3.6-27B-6bit"
      MODEL_SIZE="~23GB"
      SERVER_ARG="qwen36"
      ;;
    2)
      MODEL="Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2"
      MODEL_NAME="SuperGemma4-26B-4bit"
      MODEL_SIZE="~16GB"
      SERVER_ARG="supergemma4"
      ;;
    3)
      read -p "  Hugging Face 모델 ID: " MODEL
      MODEL_NAME="$MODEL"
      MODEL_SIZE="알 수 없음"
      SERVER_ARG=""
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

# 6. mlx-vlm 설치
echo ""
echo "📦 mlx-vlm 설치 중..."
"$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip -q
"$SCRIPT_DIR/.venv/bin/pip" install mlx-vlm -q
echo "✅ mlx-vlm 설치 완료"

echo "📦 FastAPI + uvicorn 설치 중..."
"$SCRIPT_DIR/.venv/bin/pip" install fastapi uvicorn -q
echo "✅ FastAPI + uvicorn 설치 완료"

echo "📦 Pillow 설치 중 (이미지 처리)..."
"$SCRIPT_DIR/.venv/bin/pip" install Pillow -q
echo "✅ Pillow 설치 완료"

echo "📦 torch + torchvision 설치 중 (transformers 의존성)..."
"$SCRIPT_DIR/.venv/bin/pip" install torch torchvision -q
echo "✅ torch + torchvision 설치 완료"

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
  "$SCRIPT_DIR/.venv/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$MODEL', ignore_patterns=['*.bin', '*.pt'])
" 2>&1 | tail -5
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
if [ -n "$SERVER_ARG" ]; then
  echo "  API 서버:     ./llm-server.sh $SERVER_ARG"
else
  echo "  API 서버:     ./llm-server.sh"
fi
echo "  상태 확인:    ./llm-chat.sh status"
echo ""
echo "  Alias 설정 (선택):"
echo "  echo 'alias llm-chat=\"$SCRIPT_DIR/llm-chat.sh\"' >> ~/.zshrc"
echo "  echo 'alias llm-server=\"$SCRIPT_DIR/llm-server.sh\"' >> ~/.zshrc"
echo ""
