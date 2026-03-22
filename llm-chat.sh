#!/bin/bash
# Qwen3.5-35B-A3B 채팅 실행 스크립트
#
# 사용법:
#   ./llm-chat.sh           # 기본 262K로 채팅
#   ./llm-chat.sh 1m        # 1M 컨텍스트로 채팅
#   ./llm-chat.sh 262k      # 명시적으로 262K 채팅
#   ./llm-chat.sh status    # 현재 프로필 확인

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin"
MODEL="mlx-community/Qwen3.5-35B-A3B-4bit"
PROFILE_DIR="$SCRIPT_DIR/profiles"
MODEL_CONFIG="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec/config.json"

switch_profile() {
  case "$1" in
    262k)
      cp "$PROFILE_DIR/config-262k.json" "$MODEL_CONFIG"
      echo "✅ 262K 컨텍스트 (기본) 적용"
      ;;
    1m)
      cp "$PROFILE_DIR/config-1m.json" "$MODEL_CONFIG"
      echo "✅ 1M 컨텍스트 (YaRN) 적용"
      echo "   ⚠️  짧은 프롬프트에서 품질 약간 저하 가능"
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
  status|s)
    show_status
    exit 0
    ;;
  *)
    show_status
    ;;
esac

echo ""
echo "🚀 채팅 시작 (종료: Ctrl+C)"
echo ""

"$VENV/mlx_lm.chat" --model "$MODEL" "$@"
