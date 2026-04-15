#!/bin/bash
# Qwen3.5-35B-A3B / SuperGemma4 API 서버 실행 스크립트
#
# 사용법:
#   ./llm-server.sh              # 기본 (Thinking OFF)
#   ./llm-server.sh 1m           # 1M 컨텍스트
#   ./llm-server.sh --think      # Thinking ON (수학/코딩 정확도 향상)
#   ./llm-server.sh 1m --think   # 1M + Thinking ON
#   ./llm-server.sh 262k 9090    # 포트 지정
#   ./llm-server.sh supergemma4    # SuperGemma4 모델
#   ./llm-server.sh supergemma4 --think  # SuperGemma4 + Thinking (모델이 지원하는 경우만)
#
# Thinking 제어:
#   기본 Thinking OFF. --think 옵션으로 기본 ON.
#   어느 쪽이든 요청별 enable_thinking 파라미터로 override 가능.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin"
MODEL="mlx-community/Qwen3.5-35B-A3B-4bit"
PROFILE_DIR="$SCRIPT_DIR/profiles"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
MODEL_CONFIG=$(find "$HF_CACHE/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots" -maxdepth 2 -name "config.json" 2>/dev/null | head -1)
PORT=8080
USE_THINK=false

switch_profile() {
  case "$1" in
    262k)
      cp "$PROFILE_DIR/config-262k.json" "$MODEL_CONFIG"
      echo "✅ 262K 컨텍스트 (기본) 적용"
      ;;
    1m)
      cp "$PROFILE_DIR/config-1m.json" "$MODEL_CONFIG"
      echo "✅ 1M 컨텍스트 (YaRN) 적용"
      ;;
  esac
}

show_status() {
  if grep -q '"rope_type": "yarn"' "$MODEL_CONFIG" 2>/dev/null; then
    echo "📍 현재: 1M 컨텍스트 (YaRN 활성)"
  else
    echo "📍 현재: 262K 컨텍스트 (기본)"
  fi
}

# 인자 파싱
SERVER_ARGS=()
for arg in "$@"; do
  case "$arg" in
    supergemma4)
      MODEL="Jiunsong/supergemma4-26b-abliterated-multimodal-mlx-4bit"
      ;;
    1m|long)
      switch_profile 1m
      ;;
    262k|default)
      switch_profile 262k
      ;;
    --think)
      USE_THINK=true
      ;;
    *)
      if [[ "$arg" =~ ^[0-9]+$ ]]; then
        PORT="$arg"
      fi
      ;;
  esac
done

# 서버 인자 구성
SERVER_ARGS+=(--model "$MODEL" --host 0.0.0.0 --port "$PORT")
if [ "$USE_THINK" = true ]; then
  SERVER_ARGS+=(--think)
fi

# 프로필 인자 없으면 상태 표시
if ! echo "$@" | grep -qE "1m|long|262k|default"; then
  show_status
fi

# 로컬 IP 확인
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)
TS_IP=$(tailscale ip -4 2>/dev/null)

echo ""
echo "🌐 API 서버 시작"
echo "   로컬:     http://localhost:$PORT"
echo "   네트워크: http://$LOCAL_IP:$PORT"
[ -n "$TS_IP" ] && echo "   Tailscale: http://$TS_IP:$PORT"
echo ""
echo "   엔드포인트: /v1/chat/completions"
echo "   스트리밍: stream=true 지원"
if [ "$USE_THINK" = true ]; then
  echo "   🧠 Thinking: ON (기본, 요청별 override 가능)"
else
  echo "   🧠 Thinking: OFF (기본, 요청별 override 가능)"
fi
echo "   📝 로깅: ON (logs/ 폴더에 저장)"
echo "   종료: Ctrl+C"
echo "   💤 덮개 닫아도 서버 유지됩니다 (caffeinate -dis, 전원 연결 필요)"
echo ""

exec caffeinate -dis "$VENV/python" "$SCRIPT_DIR/llm-api-server.py" "${SERVER_ARGS[@]}"
