#!/bin/bash
# Qwen3.5-35B-A3B API 서버 실행 스크립트
#
# 사용법:
#   ./llm-server.sh              # 기본 (262K, 포트 8080)
#   ./llm-server.sh 1m           # 1M 컨텍스트
#   ./llm-server.sh 262k 9090    # 262K + 포트 지정
#   ./llm-server.sh 1m 9090      # 1M + 포트 지정

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin"
MODEL="mlx-community/Qwen3.5-35B-A3B-4bit"
PROFILE_DIR="$SCRIPT_DIR/profiles"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
MODEL_CONFIG=$(find "$HF_CACHE/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots" -maxdepth 2 -name "config.json" 2>/dev/null | head -1)
PORT=8080

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

case "$1" in
  1m|long)
    switch_profile 1m
    shift
    ;;
  262k|default)
    switch_profile 262k
    shift
    ;;
  *)
    show_status
    ;;
esac

# 포트 지정 (두번째 인자)
if [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
  PORT="$1"
  shift
fi

# 로컬 IP 확인
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null)

echo ""
echo "🌐 API 서버 시작"
echo "   로컬:  http://localhost:$PORT"
echo "   네트워크: http://$LOCAL_IP:$PORT"
echo ""
echo "   엔드포인트: /v1/chat/completions"
echo "   종료: Ctrl+C"
echo "   💤 화면이 꺼져도 서버는 유지됩니다 (caffeinate)"
echo ""
echo "   테스트:"
echo "   curl http://$LOCAL_IP:$PORT/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"messages\":[{\"role\":\"user\",\"content\":\"안녕!\"}],\"max_tokens\":200}'"
echo ""

# caffeinate -di: 시스템 잠자기 방지 (화면은 꺼져도 됨)
# 서버 프로세스가 끝나면 caffeinate도 자동 종료
exec caffeinate -di "$VENV/mlx_lm.server" --model "$MODEL" --host 0.0.0.0 --port "$PORT" "$@"
