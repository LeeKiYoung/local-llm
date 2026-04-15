# Local LLM Server (MLX)

> **Apple Silicon Mac 전용** (M1/M2/M3/M4/M5)
> MLX는 Apple의 네이티브 ML 프레임워크로, Apple Silicon에서만 동작합니다.

Apple Silicon Mac에서 로컬 LLM을 OpenAI 호환 API 서버로 실행하는 프로젝트.
openclaw, OpenAI SDK 등 기존 클라이언트를 그대로 연결해 완전히 로컬에서 추론합니다.

---

## 지원 모델

| 모델 | 실행 명령 | 메모리 | 속도 | 특징 |
|------|----------|------:|-----:|------|
| **Qwen3.5-35B-A3B** (기본) | `./llm-server.sh` | ~20GB | 103 tok/s | 한국어+코딩 올라운더, Thinking 모드, 1M 컨텍스트 |
| **SuperGemma4-26B** | `./llm-server.sh supergemma4` | ~15GB | 46 tok/s | 무검열, 툴콜 강화, 128K 컨텍스트 |

두 모델 동시 로드는 불가 (메모리 초과). 서버 재시작으로 전환.

---

## 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| Mac | Apple Silicon (M1+) | M3 Pro / M4 Pro 이상 |
| 메모리 | 24GB | 64GB |
| Python | 3.10+ | 3.11+ |
| 디스크 | 20GB 여유 | 40GB+ |

---

## 설치

```bash
git clone https://github.com/LeeKiYoung/local-llm.git
cd local-llm
./setup.sh
```

`setup.sh`가 자동으로:
1. Apple Silicon / Python / 메모리 확인
2. 가상환경 생성 + mlx-lm + FastAPI + uvicorn 설치
3. 선택한 모델 다운로드

### 환경만 셋업 (모델 나중에)

```bash
./setup.sh --no-model
```

### 모델 캐시 위치 변경 (외장 SSD 등)

```bash
# ~/.zshrc에 추가
export HF_HOME=/Volumes/MySSD/.huggingface
source ~/.zshrc && ./setup.sh
```

---

## 프로젝트 구조

```
local-llm/
├── llm-server.sh           # API 서버 실행 스크립트
├── llm-api-server.py       # FastAPI 커스텀 API 서버 (핵심)
├── llm-chat.sh             # 대화형 채팅 (Qwen3.5 전용)
├── setup.sh                # 자동 셋업
├── profiles/
│   ├── config-262k.json    # 기본 프로필 (262K, Qwen3.5 전용)
│   └── config-1m.json      # 확장 프로필 (1M YaRN, Qwen3.5 전용)
├── test_api_server.py      # API 서버 테스트 (13개)
├── .venv/                  # Python 가상환경
└── logs/                   # 요청/응답 JSONL 로그 (자동 생성)
```

---

## API 서버

### 시작

```bash
# Qwen3.5 (기본)
./llm-server.sh              # 262K 컨텍스트, Thinking OFF
./llm-server.sh 1m           # 1M 컨텍스트 (YaRN) ← 주로 이걸 씀
./llm-server.sh --think      # Thinking ON (수학/코딩 정확도 향상)
./llm-server.sh 1m --think   # 1M + Thinking ON

# SuperGemma4
./llm-server.sh supergemma4  # 첫 실행 시 ~15GB 자동 다운로드
```

> **모델별 지원 기능**
>
> | 기능 | Qwen3.5 | SuperGemma4 |
> |------|:-------:|:-----------:|
> | 컨텍스트 프로필 (1m/262k) | ✅ | ❌ (128K 고정) |
> | Thinking 모드 (`enable_thinking`) | ✅ | ❌ |

### 기본 호출

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "안녕!"}],
    "max_tokens": 200
  }'
```

### 스트리밍

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "안녕!"}],
    "stream": true,
    "max_tokens": 200
  }'
```

### 요청별 Thinking 제어 (Qwen3.5 전용)

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "123*456=?"}],
    "enable_thinking": true,
    "max_tokens": 500
  }'
```

### 지원 파라미터 (OpenAI 호환)

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `messages` | array | 필수 | 대화 메시지 |
| `stream` | bool | false | SSE 스트리밍 |
| `temperature` | float | 1.0 | 샘플링 온도 |
| `top_p` | float | 1.0 | Nucleus sampling |
| `max_tokens` | int | 2048 | 최대 생성 토큰 |
| `max_completion_tokens` | int | — | max_tokens 별칭 |
| `stop` | string/array | null | 정지 시퀀스 |
| `seed` | int | null | 결정적 샘플링 |
| `enable_thinking` | bool | false | Thinking 모드 (Qwen3.5 전용) |

### 웹 UI / 클라이언트 연결

Continue, Open WebUI 등에서 OpenAI endpoint로 연결:
- URL: `http://localhost:8080/v1` (또는 네트워크 IP)
- API Key: 아무 값 (인증 없음)

### 외부 네트워크 접속 (Tailscale)

```bash
brew install tailscale
# 양쪽 기기에서 로그인 후
curl http://<TAILSCALE_IP>:8080/v1/chat/completions ...
```

### 아키텍처

```
클라이언트(:8080) → FastAPI 서버 → mlx_lm Python API (직접 호출)
```

- 요청 완료 후 KV 캐시 자동 해제 (`mx.clear_cache()`)
- `asyncio.Semaphore(1)` — GPU 순차 처리, HTTP 동시 수신
- 대기 큐 5개 초과 시 429 응답 (OOM 방지)

---

## 컨텍스트 프로필 (Qwen3.5 전용)

262K(기본)와 1M(확장) 두 프로필을 `config.json` 교체 방식으로 전환.

```bash
./llm-server.sh 1m     # 1M으로 전환 후 서버 시작
./llm-server.sh 262k   # 262K으로 전환 후 서버 시작
./llm-server.sh        # 현재 프로필 그대로 시작
```

| 프로필 | 컨텍스트 | 메모리 | 용도 |
|-------|-------:|------:|------|
| 262K (기본) | ~52만 글자 | ~22-25GB | 일반 대화, 코딩 |
| 1M (YaRN) | ~200만 글자 | ~34GB | 대형 문서/코드베이스 |

- 전환 후 서버 재시작 필요
- 1M 모드는 짧은 프롬프트에서 품질 약간 저하 가능 (Static YaRN 특성)

---

## Thinking 모드 (Qwen3.5 전용)

서버 기본값은 `--think` 플래그로, 개별 요청은 `enable_thinking`으로 override.

| 상황 | 권장 | 이유 |
|------|:---:|------|
| 수학/논리 | ON | 정확도 크게 향상 |
| 코딩 | ON | 단계적 사고로 버그 감소 |
| 간단한 질문/번역/요약 | OFF | 빠른 응답 |

- Thinking 토큰도 `max_tokens`에 포함 → Thinking ON 시 8000+ 권장

---

## 대화형 채팅 (Qwen3.5 전용)

```bash
./llm-chat.sh           # 262K
./llm-chat.sh 1m        # 1M 컨텍스트
./llm-chat.sh --think   # Thinking ON
```

---

## 로깅

요청/응답이 `logs/` 폴더에 일별 JSONL 파일로 자동 저장.

```
logs/
├── 2026-04-15.jsonl
└── 2026-04-16.jsonl
```

| 항목 | 내용 |
|------|------|
| timestamp | 요청 시각 |
| ip | 호출한 기기 IP |
| duration_ms | 응답 시간 (ms) |
| enable_thinking | Thinking ON/OFF |
| stream | 스트리밍 여부 |
| prompt_preview | 프롬프트 미리보기 (200자) |
| content_preview | 응답 미리보기 (200자) |
| usage | 토큰 사용량 |

```bash
# 오늘 로그 보기
cat logs/$(date +%Y-%m-%d).jsonl | python3 -m json.tool

# 느린 요청 찾기 (3초 이상)
cat logs/*.jsonl | jq 'select(.duration_ms > 3000)'
```

---

## 테스트

```bash
.venv/bin/python -m pytest test_api_server.py -v
```

mock 모델 사용, GPU 불필요. 13개 테스트 (OpenAI 호환성, 스트리밍, Thinking, 429 처리 등).

---

## 메모리 관리

| 상태 | 메모리 |
|------|-------:|
| 서버 미실행 | ~21GB (시스템) |
| Qwen3.5 실행 중 | ~41GB |
| SuperGemma4 실행 중 | ~36GB |
| Ctrl+C 종료 후 | ~21GB (즉시 해제) |

```bash
# 프로세스 확인
ps aux | grep llm-api-server | grep -v grep

# 강제 종료
kill $(pgrep -f llm-api-server)
```

---

## 참고 링크

- [MLX-LM GitHub](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [Qwen3.5-35B-A3B (MLX 4bit)](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit)
- [Qwen3.5-122B-A10B (MLX 4bit)](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-4bit)
- [Qwen3.5 공식 GitHub](https://github.com/QwenLM/Qwen3.5)
- [Qwen3.5 로컬 가이드 (Unsloth)](https://unsloth.ai/docs/models/qwen3.5)
- [SuperGemma4 26B MLX 4bit](https://huggingface.co/Jiunsong/supergemma4-26b-abliterated-multimodal-mlx-4bit)
- [SuperGemma4 26B GGUF](https://huggingface.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2)
- [Gemma 4 공식 블로그 (Google)](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [TurboQuant (Google Research)](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [Gemma 4 on Ollama](https://ollama.com/library/gemma4)
- [Apple MLX + M5 리서치](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [M5 Pro/Max 로컬 LLM 가이드](https://modelfit.io/blog/m5-pro-max-local-llm-2026/)
- [Unsloth GGUF Benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
- [whatcani.run — Apple Silicon 실측 LLM 벤치마크](https://www.whatcani.run/)

---

## License

[MIT License](LICENSE)
