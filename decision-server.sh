#!/bin/bash
# Kev 판단모델(Jev 호환 System One API) 서버 실행 스크립트
#
# 글을 생성하는 llm-server.sh(:8080)와 별개인 "판단 전용" 사이드카다. state + 타입 있는 질문을 보내면
# 선택지별 확률을 돌려준다 (TypeSafe Jev와 같은 POST /v1/systemone — Jev 클라이언트 코드 그대로 사용).
# 모델: jaredpalmer/kev (Qwen3.5 베이스 + LoRA 어댑터 + 포인터 헤드, Apache-2.0), Apple Silicon에서는 MLX.
#
# 사용법:
#   ./decision-server.sh              # Kev-4B (기본), 127.0.0.1:8009
#   ./decision-server.sh 0.8b         # Kev-0.8B — 메모리 최소, 정확도 낮음
#   ./decision-server.sh 9b           # Kev-9B — 32GB+ 맥
#   ./decision-server.sh 4b 9009      # 포트 지정
#   KEV_API_KEY=… ./decision-server.sh --host 0.0.0.0   # 외부 바인딩 — KEV_API_KEY 없으면 거부
#   ./decision-server.sh --setup      # 설치만 (kev 클론 + uv sync), 서버는 안 띄움
#
# 환경변수:
#   KEV_DIR       kev 체크아웃 위치 (기본: <이 레포>/.kev)
#   KEV_API_KEY   설정하면 Authorization: Bearer <키> 요구. 루프백이 아닌 --host에는 필수
#
# 외부 노출: 서버는 평문 HTTP라 Bearer 키도 평문으로 오간다 — Tailscale 같은 암호화된 망 안에서만
#   열거나, 127.0.0.1에 두고 TLS 종단 프록시 뒤에 둘 것. 공용 인터넷에 직접 열지 말 것.
#
# 요구사항: uv (https://docs.astral.sh/uv/), 처음 실행 시 베이스 모델 다운로드 (4B ≈ 9GB)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KEV_DIR="${KEV_DIR:-$SCRIPT_DIR/.kev}"
KEV_REPO="https://github.com/jaredpalmer/kev.git"
KEV_TAG="v1.0"        # Hub 태그 — 같은 가중치를 고정해서 임계값이 흔들리지 않게
KEV_SRC_TAG="kev-1.0" # 서버 코드도 같은 릴리스로 고정 (설치 시점마다 다른 코드가 돌지 않게)

SIZE="4b"
PORT="8009"
HOST="127.0.0.1"
SETUP_ONLY=0

while [ $# -gt 0 ]; do
  case "$1" in
    0.8b|4b|9b) SIZE="$1" ;;
    --host) HOST="$2"; shift ;;
    --setup) SETUP_ONLY=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    [0-9]*) PORT="$1" ;;
    *) echo "❌ 알 수 없는 인자: $1 (./decision-server.sh --help)"; exit 1 ;;
  esac
  shift
done

if ! command -v uv >/dev/null 2>&1; then
  echo "❌ uv가 필요합니다: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

# 외부 바인딩은 인증 필수 — 설치 전에 먼저 거른다
if [ "$SETUP_ONLY" = 0 ]; then
  case "$HOST" in
    127.0.0.1|localhost|::1) ;;
    *)
      if [ -z "$KEV_API_KEY" ]; then
        echo "❌ $HOST 에 인증 없이 열 수 없습니다 — KEV_API_KEY=<키> 를 같이 지정하세요"
        exit 1
      fi
      echo "🔐 Bearer 인증 사용 — 평문 HTTP이므로 암호화된 망(Tailscale 등)이나 TLS 프록시 뒤에서만 쓰세요"
      ;;
  esac
fi

# 설치: kev 클론 + 의존성 (torch는 3.14 휠이 없어서 kev의 .python-version(3.13)을 uv가 맞춰줌)
if [ ! -d "$KEV_DIR/.git" ]; then
  echo "📦 kev $KEV_SRC_TAG 클론 → $KEV_DIR"
  git -c advice.detachedHead=false clone -q --depth 1 --branch "$KEV_SRC_TAG" "$KEV_REPO" "$KEV_DIR" || exit 1
elif [ "$(git -C "$KEV_DIR" rev-parse HEAD)" != "$(git -C "$KEV_DIR" rev-parse -q --verify "$KEV_SRC_TAG^{commit}" 2>/dev/null)" ]; then
  # 남이 만든 체크아웃(KEV_DIR 지정)은 함부로 바꾸지 않고 알리기만 한다
  echo "⚠️  $KEV_DIR 이 $KEV_SRC_TAG 가 아닙니다 — 고정하려면: git -C $KEV_DIR fetch --tags && git -C $KEV_DIR checkout $KEV_SRC_TAG"
fi
# 매번 실행 — 이미 맞으면 금방 끝나고, 이전 설치가 serve 의존성 없이 끝났어도 여기서 채워진다
(cd "$KEV_DIR" && uv sync -q --extra serve) || { echo "❌ uv sync --extra serve 실패"; exit 1; }
[ "$SETUP_ONLY" = 1 ] && { echo "✅ 설치 완료: $KEV_DIR"; exit 0; }

RUN="jaredpalmer/kev-$SIZE@$KEV_TAG"
echo "🧭 Kev 판단모델: $RUN → http://$HOST:$PORT/v1/systemone"
echo "   (처음 실행이면 베이스 모델을 내려받느라 몇 분 걸립니다)"
cd "$KEV_DIR" && exec uv run --extra serve python -m kev.serve --run "$RUN" --host "$HOST" --port "$PORT"
