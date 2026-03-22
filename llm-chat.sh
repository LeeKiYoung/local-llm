#!/bin/bash
# Qwen3.5-35B-A3B 채팅 실행 스크립트
#
# 사용법:
#   ./llm-chat.sh           # 기본 262K로 채팅
#   ./llm-chat.sh 1m        # 1M 컨텍스트로 채팅
#   ./llm-chat.sh 262k      # 명시적으로 262K 채팅

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin"
MODEL="mlx-community/Qwen3.5-35B-A3B-4bit"

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

echo ""
echo "🚀 채팅 시작 (종료: Ctrl+C)"
echo ""

"$VENV/mlx_lm.chat" --model "$MODEL" "$@"
