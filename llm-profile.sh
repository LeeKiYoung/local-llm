#!/bin/bash
# Qwen3.5-35B-A3B 컨텍스트 프로필 전환 스크립트
#
# 사용법:
#   ./llm-profile.sh 262k    # 기본 (262K 컨텍스트)
#   ./llm-profile.sh 1m      # 확장 (1M 컨텍스트)
#   ./llm-profile.sh status   # 현재 상태 확인

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROFILE_DIR="$SCRIPT_DIR/profiles"
MODEL_CONFIG="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/snapshots/1e20fd8d42056f870933bf98ca6211024744f7ec/config.json"

case "$1" in
  262k|default)
    cp "$PROFILE_DIR/config-262k.json" "$MODEL_CONFIG"
    echo "✅ 262K 컨텍스트 (기본) 적용 완료"
    echo "   - 일반 대화, 코딩에 최적"
    echo "   - 메모리: ~22-25GB"
    ;;
  1m|long)
    cp "$PROFILE_DIR/config-1m.json" "$MODEL_CONFIG"
    echo "✅ 1M 컨텍스트 (YaRN) 적용 완료"
    echo "   - 대형 문서/코드베이스 분석용"
    echo "   - 메모리: ~34GB"
    echo "   ⚠️  짧은 프롬프트에서 품질 약간 저하 가능"
    ;;
  status|s)
    if grep -q '"rope_type": "yarn"' "$MODEL_CONFIG" 2>/dev/null; then
      echo "📍 현재: 1M 컨텍스트 (YaRN 활성)"
    else
      echo "📍 현재: 262K 컨텍스트 (기본)"
    fi
    ;;
  *)
    echo "사용법: $0 {262k|1m|status}"
    echo ""
    echo "  262k (default) - 기본 컨텍스트 (일반용)"
    echo "  1m   (long)    - 1M 확장 컨텍스트 (긴 문서용)"
    echo "  status         - 현재 프로필 확인"
    ;;
esac
