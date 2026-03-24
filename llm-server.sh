#!/bin/bash
# Qwen3.5-35B-A3B API 서버 실행 스크립트
#
# 사용법:
#   ./llm-server.sh              # 기본 (262K, 포트 8080, 로깅 ON)
#   ./llm-server.sh 1m           # 1M 컨텍스트
#   ./llm-server.sh --no-log     # 로깅 없이 실행
#   ./llm-server.sh 1m --no-log  # 1M + 로깅 없이
#   ./llm-server.sh 262k 9090    # 포트 지정
#
# Thinking 제어:
#   서버는 항상 Thinking ON으로 실행됩니다.
#   요청에 "no_think": true를 포함하면 프록시가 thinking을 제거합니다.
#   (로깅 모드에서만 동작, --no-log 시에는 서버에 직접 전달)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin"
MODEL="mlx-community/Qwen3.5-35B-A3B-4bit"
PROFILE_DIR="$SCRIPT_DIR/profiles"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
MODEL_CONFIG=$(find "$HF_CACHE/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots" -maxdepth 2 -name "config.json" 2>/dev/null | head -1)
PORT=8080
USE_LOG=true

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
ARGS=()
for arg in "$@"; do
  case "$arg" in
    1m|long)
      switch_profile 1m
      ;;
    262k|default)
      switch_profile 262k
      ;;
    --no-log)
      USE_LOG=false
      ;;
    *)
      if [[ "$arg" =~ ^[0-9]+$ ]]; then
        PORT="$arg"
      else
        ARGS+=("$arg")
      fi
      ;;
  esac
done

# 프로필 인자 없으면 상태 표시
if ! echo "$@" | grep -qE "1m|long|262k|default"; then
  show_status
fi

# 로컬 IP 확인
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)
# Tailscale IP 확인
TS_IP=$(tailscale ip -4 2>/dev/null)

echo ""
echo "🌐 API 서버 시작"
echo "   로컬:     http://localhost:$PORT"
echo "   네트워크: http://$LOCAL_IP:$PORT"
[ -n "$TS_IP" ] && echo "   Tailscale: http://$TS_IP:$PORT"
echo ""
echo "   엔드포인트: /v1/chat/completions"
echo "   🧠 Thinking: 요청별 제어 (\"no_think\": true 로 OFF)"
echo "   종료: Ctrl+C"
echo "   💤 화면이 꺼져도 서버는 유지됩니다 (caffeinate)"

if [ "$USE_LOG" = true ]; then
  BACKEND_PORT=$((PORT + 1))
  echo "   📝 로깅: ON (logs/ 폴더에 저장)"
  echo ""

  # 백엔드 서버 시작 (내부 포트)
  caffeinate -di "$VENV/mlx_lm.server" --model "$MODEL" --host 127.0.0.1 --port "$BACKEND_PORT" "${ARGS[@]}" &
  BACKEND_PID=$!

  # 백엔드 시작 대기
  echo "   백엔드 로딩 중..."
  sleep 3

  # 프록시 서버 시작 (외부 포트)
  BACKEND_PORT="$BACKEND_PORT" PROXY_PORT="$PORT" "$VENV/python" "$SCRIPT_DIR/llm-proxy.py" &
  PROXY_PID=$!

  # 종료 시 둘 다 정리
  trap "kill $BACKEND_PID $PROXY_PID 2>/dev/null; exit" INT TERM
  wait $BACKEND_PID
else
  echo "   📝 로깅: OFF"
  echo ""
  exec caffeinate -di "$VENV/mlx_lm.server" --model "$MODEL" --host 0.0.0.0 --port "$PORT" "${ARGS[@]}"
fi
