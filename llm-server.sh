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
PORT=8080

# 프로필 전환
case "$1" in
  1m|long)
    "$SCRIPT_DIR/llm-profile.sh" 1m
    shift
    ;;
  262k|default)
    "$SCRIPT_DIR/llm-profile.sh" 262k
    shift
    ;;
  *)
    "$SCRIPT_DIR/llm-profile.sh" status
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
echo ""
echo "   테스트:"
echo "   curl http://$LOCAL_IP:$PORT/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"messages\":[{\"role\":\"user\",\"content\":\"안녕!\"}],\"max_tokens\":200}'"
echo ""

"$VENV/mlx_lm.server" --model "$MODEL" --host 0.0.0.0 --port "$PORT" "$@"
