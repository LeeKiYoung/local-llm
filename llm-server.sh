#!/bin/bash
# Qwen3.8-27B / Qwen3.6-27B / SuperGemma4 API 서버 실행 스크립트
#
# 사용법:
#   ./llm-server.sh              # 기본 (Thinking OFF, MTP ON)
#   ./llm-server.sh 1m           # 1M 컨텍스트
#   ./llm-server.sh --think      # Thinking ON (수학/코딩 정확도 향상)
#   ./llm-server.sh 1m --think   # 1M + Thinking ON
#   ./llm-server.sh 262k 9090    # 포트 지정
#   ./llm-server.sh qwen38       # Qwen3.8-27B-8bit (기본값과 동일)
#   ./llm-server.sh qwen36       # Qwen3.6-27B-6bit (이전 기본 모델)
#   ./llm-server.sh qwen36-fast  # Qwen3.6-35B-A3B (MoE, 3B active — 3~4배 빠름, 품질은 27B가 우위)
#   ./llm-server.sh supergemma4    # SuperGemma4 텍스트 전용 (uncensored v2, = supergemma4-text)
#   ./llm-server.sh supergemma4-vlm  # SuperGemma4 멀티모달 (abliterated)
#   ./llm-server.sh supergemma4 --think  # SuperGemma4 + Thinking (모델이 지원하는 경우만)
#   ./llm-server.sh --no-mtp     # MTP speculative decoding 비활성화
#   ./llm-server.sh --no-apc     # APC prefix caching 비활성화
#   ./llm-server.sh --no-dsh     # dsh 웹 UI 자동 실행 안 함
#
# Thinking 제어:
#   기본 Thinking OFF. --think 옵션으로 기본 ON.
#   어느 쪽이든 요청별 enable_thinking 파라미터로 override 가능.
#
# MTP 제어:
#   기본 MTP ON (mlx-vlm 소스 버전 필요). --no-mtp 옵션으로 비활성화.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin"
MODEL="mlx-community/Qwen3.8-27B-8bit"
PROFILE_DIR="$SCRIPT_DIR/profiles"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
# 262k/1m 프로필 전환은 기본 모델(Qwen3.8-27B-8bit)의 캐시 config를 대상으로 한다.
# YaRN 1M 프로필은 Qwen3.6에서 검증된 설정 — 3.8도 동일 아키텍처(qwen3_5)라 적용 가능.
MODEL_CONFIG=$(find "$HF_CACHE/models--mlx-community--Qwen3.8-27B-8bit/snapshots" -maxdepth 2 -name "config.json" 2>/dev/null | head -1)
PORT=8080
USE_THINK=false
NO_MTP=false
NO_APC=false
NO_DSH=false

switch_profile() {
  if [ -z "$MODEL_CONFIG" ]; then
    echo "⚠️  모델 캐시를 찾을 수 없습니다. 먼저 모델을 다운로드하세요."
    return 1
  fi
  local BACKUP="${MODEL_CONFIG}.original"
  # 원본 백업이 없으면 현재 파일을 백업 (심링크 대상 파일을 복사)
  if [ ! -f "$BACKUP" ]; then
    cp -L "$MODEL_CONFIG" "$BACKUP"
  fi
  case "$1" in
    262k)
      # 원본 복원 후 262K 설정만 덮어쓰기
      "$VENV/python" - "$BACKUP" "$MODEL_CONFIG" <<'EOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f: cfg = json.load(f)
tc = cfg.setdefault("text_config", {})
rp = tc.setdefault("rope_parameters", {})
rp.update({"rope_type": "default", "type": "default"})
rp.pop("factor", None)
rp.pop("original_max_position_embeddings", None)
tc["max_position_embeddings"] = 262144
cfg["max_position_embeddings"] = 262144
with open(dst, "w") as f: json.dump(cfg, f, indent=2)
EOF
      echo "✅ 262K 컨텍스트 (기본) 적용"
      ;;
    1m)
      # 원본 복원 후 1M YaRN 설정만 덮어쓰기
      "$VENV/python" - "$BACKUP" "$MODEL_CONFIG" <<'EOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f: cfg = json.load(f)
tc = cfg.setdefault("text_config", {})
rp = tc.setdefault("rope_parameters", {})
rp.update({
    "rope_type": "yarn", "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": 262144,
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "partial_rotary_factor": 0.25,
    "rope_theta": 10000000
})
tc["max_position_embeddings"] = 1048576
cfg["max_position_embeddings"] = 1048576
with open(dst, "w") as f: json.dump(cfg, f, indent=2)
EOF
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
    supergemma4|supergemma4-text)
      # 텍스트 전용 uncensored v2 — setup.sh 선택 2와 동일 모델 (#21)
      MODEL="Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2"
      ;;
    supergemma4-vlm)
      # 멀티모달 abliterated variant
      MODEL="Jiunsong/supergemma4-26b-abliterated-multimodal-mlx-4bit"
      ;;
    qwen38)
      MODEL="mlx-community/Qwen3.8-27B-8bit"
      ;;
    qwen36)
      MODEL="mlx-community/Qwen3.6-27B-6bit"
      ;;
    qwen36-fast|fast)
      # MoE 35B-A3B — 활성 파라미터 3B라 디코딩이 3~4배 빠르다.
      # 품질은 27B dense가 전 벤치에서 우위이므로 대량/반복 작업용 보조 프로필.
      MODEL="mlx-community/Qwen3.6-35B-A3B-8bit"
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
    --no-mtp)
      NO_MTP=true
      ;;
    --no-apc)
      NO_APC=true
      ;;
    --no-dsh)
      NO_DSH=true
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
if [ "$NO_MTP" = true ]; then
  SERVER_ARGS+=(--no-draft)
fi
if [ "$NO_APC" = true ]; then
  SERVER_ARGS+=(--no-apc)
fi
# --apc-dir는 넘기지 않는다 = 메모리 전용.
# Qwen3.6-27B는 hybrid(linear+full attention)라 APC가 block이 아닌 exact 모드로 동작하고,
# 실측 결과 히트 0회 / 디스크만 1.4GB 소모했다. 디스크 영속이 필요하면 llm-api-server.py에
# --apc-dir를 직접 지정할 것.

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
echo "   대시보드: http://localhost:$PORT/dashboard"
[ -n "$TS_IP" ] && echo "   Tailscale: http://$TS_IP:$PORT"
echo ""
echo "   엔드포인트: /v1/chat/completions"
echo "   스트리밍: stream=true 지원"
if [ "$USE_THINK" = true ]; then
  echo "   🧠 Thinking: ON (기본, 요청별 override 가능)"
else
  echo "   🧠 Thinking: OFF (기본, 요청별 override 가능)"
fi
if [ "$NO_MTP" = false ]; then
  echo "   🚀 MTP: ON (speculative decoding, block_size=6)"
else
  echo "   🚀 MTP: OFF"
fi
if [ "$NO_APC" = false ]; then
  echo "   ♻️  APC: ON (prefix caching, 메모리 전용)"
else
  echo "   ♻️  APC: OFF"
fi
echo "   📝 로깅: ON (logs/ 폴더에 저장)"

# dsh 웹 UI 자동 실행 (설치돼 있으면) — API 서버 종료 시 함께 종료
DSH_PID=""
if [ "$NO_DSH" = false ] && command -v dsh >/dev/null 2>&1; then
  mkdir -p "$SCRIPT_DIR/logs"
  LOCAL_LLM_API_KEY=local dsh web > "$SCRIPT_DIR/logs/dsh-web.log" 2>&1 &
  DSH_PID=$!
  echo "   🤖 dsh 웹 UI: http://127.0.0.1:3080 (에이전트 질문창, 서버와 함께 종료)"
fi

echo "   종료: Ctrl+C"
echo "   💤 덮개 닫아도 서버 유지됩니다 (caffeinate -dis, 전원 연결 필요)"
echo ""

# exec를 쓰면 trap이 실행되지 않아 dsh가 고아 프로세스로 남는다 — 일반 실행 + trap으로 정리
trap '[ -n "$DSH_PID" ] && kill "$DSH_PID" 2>/dev/null' EXIT
caffeinate -dis "$VENV/python" "$SCRIPT_DIR/llm-api-server.py" "${SERVER_ARGS[@]}"
