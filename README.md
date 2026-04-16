# Local LLM Server (MLX)

> **Apple Silicon Mac 전용** (M1/M2/M3/M4/M5)
> MLX는 Apple의 네이티브 ML 프레임워크로, Apple Silicon에서만 동작합니다.
> NVIDIA GPU / Intel Mac / Windows / Linux는 지원하지 않습니다.

Apple Silicon Mac에서 로컬 LLM을 OpenAI 호환 API 서버로 실행하는 프로젝트.
openclaw, OpenAI SDK 등 기존 클라이언트를 그대로 연결해 완전히 로컬에서 추론합니다.

## 지원 모델

| 모델 | 실행 명령 | 메모리 | 속도 | 특징 |
|------|----------|------:|-----:|------|
| **Qwen3.5-35B-A3B** (기본) | `./llm-server.sh 1m` | ~20GB | 103 tok/s | 한국어+코딩 올라운더, Thinking 모드, 1M 컨텍스트 |
| **SuperGemma4-26B uncensored-v2** | `./llm-server.sh supergemma4` | ~13GB | 46 tok/s | 무검열(파인튜닝), 툴콜·한국어·코드 강화, 텍스트 전용 |
| **SuperGemma4-26B abliterated-multimodal** | 직접 모델 ID 지정¹ | ~15GB | ~49 tok/s | 무검열(EGA), 이미지+텍스트 입력 지원 |

두 모델 동시 로드는 불가 (메모리 초과). 서버 재시작으로 전환.

> ¹ abliterated-multimodal은 `mlx_vlm` 런타임이 필요 (Phase 1 통합 예정). 현재는 `python llm-api-server.py --model Jiunsong/supergemma4-26b-abliterated-multimodal-mlx-4bit`로 직접 실행.

> **모델별 지원 기능**
>
> | 기능 | Qwen3.5 | SuperGemma4 uncensored-v2 | SuperGemma4 abliterated-multimodal |
> |------|:-------:|:------------------------:|:---------------------------------:|
> | 컨텍스트 프로필 (1m/262k) | ✅ | ❌ (128K 고정) | ❌ (256K 고정) |
> | Thinking 모드 (`enable_thinking`) | ✅ | ❌ | ❌ |
> | 대화형 채팅 (`llm-chat.sh`) | ✅ | ❌ | ❌ |
> | 이미지 입력 (멀티모달) | ❌ | ❌ | ✅ |

### SuperGemma4 모델 라인업

Jiunsong이 배포한 SuperGemma4 전체 variant. 런타임·용도에 맞게 선택.

| 모델 | 파라미터 | 포맷 | 런타임 | 용량 | 멀티모달 | 검열 해제 방식 |
|------|:-------:|------|--------|-----:|:-------:|:------------:|
| `uncensored-mlx-4bit-v2` | 26B MoE | MLX 4bit | `mlx_lm` | ~13GB | ❌ | Uncensored (파인튜닝), 기본값 |
| `abliterated-multimodal-mlx-4bit` | 26B MoE | MLX 4bit | `mlx_vlm` | ~15GB | ✅ | Abliterated+EGA, 2.2K 다운로드 |
| `abliterated-multimodal-mlx-8bit` | 26B MoE | MLX 8bit | `mlx_vlm` | ~24GB | ✅ | 위와 동일, 더 높은 품질 |
| `uncensored-gguf-v2` (Ollama) | 26B MoE | GGUF Q4_K_M | llama.cpp / Ollama | ~17GB | ❌ | Uncensored (파인튜닝), 42K+ 다운로드 |
| `SuperGemma4-31b-abliterated-mlx-4bit` | 31B **Dense** | MLX 4bit | `mlx_lm` | ~17.3GB | ❌ | Abliterated, 느림 |
| `SuperGemma4-31b-abliterated-GGUF` | 31B **Dense** | GGUF Q4_K_M | llama.cpp / Ollama | ~18.7GB | ❌ | Abliterated, 느림 |

#### 26B variant 비교 (MLX)

같은 26B 계열이지만 목적이 다름.

| 항목 | `uncensored-mlx-4bit-v2` | `abliterated-multimodal-mlx-4bit` |
|------|:------------------------:|:---------------------------------:|
| **검열 해제 방식** | Uncensored (파인튜닝) | Abliterated (가중치 벡터 제거) |
| **멀티모달** | ❌ 텍스트 전용 | ✅ 이미지+텍스트 |
| **서버 라이브러리** | `mlx_lm` | `mlx_vlm` (v0.4.3+, Day-0 지원) |
| **디스크 용량** | ~13GB | ~15GB |
| **생성 속도** | 46.2 tok/s | ~49.5 tok/s |
| **한국어/코드 강화** | ✅ 파인튜닝으로 향상 | 기본 수준 |
| **HF 다운로드** | ~8,908 | ~2,230 |

> **검열 해제 방식 차이**
> - **Uncensored**: 거부 없이 직접 답하도록 데이터로 재학습 → 코드·한국어 성능도 함께 향상
> - **Abliterated**: 모델 내부의 "거부" 방향 벡터를 수술적으로 제거 → 추가 학습 없이 검열 해제, 능력 향상은 없음
>   - Gemma 4 26B는 MoE 구조 특성상 표준 abliteration만으론 거부율 29% 잔존 → **EGA(Expert-Granular Abliteration)** 로 각 전문가(expert)에 개별 적용해 0.7%로 감소

#### uncensored-v2 Quick Bench 성능 (vs Gemma 4 26B IT 기준)

| 카테고리 | Gemma 4 26B IT | SuperGemma4 uncensored-v2 | 향상 |
|---------|:--------------:|:-------------------------:|:----:|
| 코드 | 92.3 | 98.6 | +6.3 |
| 로직/추론 | 86.9 | 95.2 | +8.3 |
| 한국어 | 90.7 | 95.0 | +4.3 |
| **전체** | **91.4** | **95.8** | **+4.4** |

v1 대비 v2 개선: tool-call 라우팅 버그 수정, 생성 속도 +8.7% (46.2 tok/s), 채팅 템플릿 중립화.

#### 31B 모델

26B 대비 파라미터 5B 증가. 텍스트 전용, Abliterated 방식.

| 항목 | MLX 4bit | GGUF Q4_K_M |
|------|:--------:|:-----------:|
| **런타임** | `mlx_lm` (Apple Silicon) | llama.cpp / Ollama |
| **용량** | ~17.3GB | ~18.7GB |
| **생성 속도 (참고)** | MLX 미측정 (RTX 3090 기준 ~30 tok/s) | - |
| **멀티모달** | ❌ | ❌ |
| **HF 다운로드** | ~1,302 | - |
| **HF** | [링크](https://huggingface.co/Jiunsong/SuperGemma4-31b-abliterated-mlx-4bit) | [링크](https://huggingface.co/Jiunsong/SuperGemma4-31b-abliterated-GGUF) |

> 31B는 26B uncensored-v2 대비 파인튜닝 없이 Abliterated만 적용 — 코드·한국어 강화 효과는 없음. Dense 구조라 26B MoE보다 생성 속도가 느림. 순수 파라미터 증가에 따른 베이스 성능 향상이 목적.

**결론**
- 이미지 처리 필요 → `abliterated-multimodal` (26B MLX, `mlx_vlm` v0.4.3+)
- 텍스트, 코딩·한국어 위주 → `uncensored-v2` (26B MLX, `mlx_lm`)
- llama.cpp / Ollama 선호 → `uncensored-gguf-v2` (26B, 89.4 tok/s, 커뮤니티 최다 사용)
- 더 큰 모델 원할 때 → `31b-abliterated` (MLX or GGUF, ~17-19GB, 단 Dense라 느림)

> **⚠️ 알려진 버그**: Gemma4 베이스 모델의 토큰 반복 붕괴(token repetition collapse) 버그가 [google-deepmind/gemma#622](https://github.com/google-deepmind/gemma/issues/622)로 공식 확인됨. Ollama/LMStudio 환경에서 재현. **MLX 환경에서는 미재현** — serving template 문제로 추정.

> **⚠️ MLX 주의사항**: `--chat-template` 옵션에 파일 경로 문자열을 직접 전달하지 말 것. 경로 문자열이 템플릿 본문 대신 그대로 주입되어 응답이 손상됨. 모델 내부 번들 템플릿을 자동 감지에 맡기는 것이 올바른 방법.

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
2. 메모리에 맞는 모델 선택 메뉴 표시
3. 가상환경 생성 + mlx-lm + FastAPI + uvicorn 설치
4. 선택한 모델 다운로드 + 스크립트에 자동 반영

| # | 모델 | 메모리 | 속도 | 최소 RAM | 특징 |
|:-:|------|------:|-----:|--------:|------|
| 1 | **Qwen3.5-35B-A3B** | ~20GB | 103 tok/s | 24GB+ | 한국어+코딩 올라운더, Thinking 모드 |
| 2 | Qwen3.5-9B | ~6GB | 40+ tok/s | 16GB+ | 가볍고 빠름 |
| 3 | Qwen3.5-27B | ~17GB | 15 tok/s | 24GB+ | Dense, 코딩 벤치마크 최강 |
| 4 | Qwen3-Coder-Next-80B | ~15GB | 25+ tok/s | 24GB+ | 코딩 에이전트 특화 |
| 5 | Gemma 4 26B MoE | ~15GB | ~42 tok/s | 24GB+ | TurboQuant, 멀티모달, 256K 컨텍스트 |
| 6 | **SuperGemma4 26B** (🔥) | ~15.6GB | ~46 tok/s | 24GB+ | 무검열+툴콜 강화, Gemma4 기반 |
| 7 | 직접 입력 | - | - | - | Hugging Face 모델 ID |

메모리에 따라 자동 추천이 표시됩니다. Enter만 누르면 추천 모델(Qwen3.5-35B-A3B)로 설치됩니다.

### 환경만 셋업 (모델 나중에)

```bash
./setup.sh --no-model
```

### 모델은 어디에 저장되나요?

첫 실행 시 모델이 자동 다운로드되며, 기본 경로에 저장됩니다:

```
~/.cache/huggingface/hub/    (macOS/Linux 공통)
```

경로를 바꾸고 싶다면 (외장 SSD 등):

```bash
# ~/.zshrc에 추가
export HF_HOME=/Volumes/MySSD/.huggingface

source ~/.zshrc && ./setup.sh
```

---

## 프로젝트 구조

```
local-llm/
├── setup.sh                        # 자동 셋업 (환경 + 모델 + 의존성)
├── llm-chat.sh                     # 대화형 채팅 (Qwen3.5 전용)
├── llm-server.sh                   # API 서버 실행
├── llm-api-server.py               # FastAPI 커스텀 API 서버 (핵심)
├── profiles/
│   ├── config-262k.json            # 기본 프로필 (262K, Qwen3.5 전용)
│   └── config-1m.json              # 확장 프로필 (1M YaRN, Qwen3.5 전용)
├── test_api_server.py              # API 서버 테스트 (13개)
├── local-llm-guide-2026.md         # 모델 비교 가이드 문서
├── .venv/                          # Python 가상환경
└── logs/                           # 요청/응답 JSONL 로그 (자동 생성)
```

---

## 빠른 시작 (Quick Start)

### 1단계: 셋업

```bash
git clone https://github.com/LeeKiYoung/local-llm.git
cd local-llm
./setup.sh
```

### 2단계: 서버 시작

```bash
# Qwen3.5 (기본, 긴 문서 모드)
./llm-server.sh 1m

# SuperGemma4
./llm-server.sh supergemma4
```

### 3단계: 요청 보내기

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "안녕!"}],
    "max_tokens": 200
  }'
```

### Qwen3.5 대화형 채팅

```bash
./llm-chat.sh 1m
```

```
✅ 1M 컨텍스트 (YaRN) 적용 완료

🚀 채팅 시작 (종료: Ctrl+C)

>> 파이썬으로 피보나치 함수 짜줘
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```

---

## 사용 모드별 가이드

### 1. 대화형 채팅 (Qwen3.5 전용)

```bash
./llm-chat.sh           # 262K
./llm-chat.sh 1m        # 1M 컨텍스트
./llm-chat.sh 262k      # 명시적 262K
```

추가 옵션:

```bash
./llm-chat.sh 1m --temp 0.3
./llm-chat.sh --max-tokens 4000
./llm-chat.sh --system-prompt "한국어로만 답해줘"
```

### 2. 단발 생성 (Qwen3.5)

```bash
source .venv/bin/activate

mlx_lm.generate \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt "Python으로 퀵소트 구현해줘" \
  --max-tokens 500

# Thinking 끄기
mlx_lm.generate \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt "안녕! /no_think" \
  --max-tokens 200
```

### 3. API 서버

OpenAI 호환 API 서버. FastAPI + mlx_lm Python API로 직접 추론.
같은 네트워크의 다른 기기(맥미니 등)에서 접속 가능.

```bash
# Qwen3.5
./llm-server.sh              # 262K 컨텍스트, Thinking OFF
./llm-server.sh 1m           # 1M 컨텍스트 (YaRN) ← 주로 이걸 씀
./llm-server.sh --think      # Thinking ON (수학/코딩 정확도 향상)
./llm-server.sh 1m --think   # 1M + Thinking ON
./llm-server.sh 262k 9090    # 포트 지정

# SuperGemma4
./llm-server.sh supergemma4          # 첫 실행 시 ~15GB 자동 다운로드
./llm-server.sh supergemma4 9090     # 포트 지정
```

실행하면:

```
🌐 API 서버 시작
   로컬:     http://localhost:8080
   네트워크: http://<YOUR_LOCAL_IP>:8080

   엔드포인트: /v1/chat/completions
   스트리밍: stream=true 지원
```

#### 기본 호출

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "안녕!"}],
    "max_tokens": 200
  }'
```

#### 스트리밍

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "안녕!"}],
    "stream": true,
    "max_tokens": 200
  }'
```

#### 요청별 Thinking 제어 (Qwen3.5 전용)

```bash
# 서버 기본값 OFF → 이 요청만 ON
curl http://localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"123*456=?"}],"enable_thinking":true,"max_tokens":500}'

# 서버 기본값 ON (--think) → 이 요청만 OFF
curl http://localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"안녕!"}],"enable_thinking":false}'
```

#### 지원 파라미터 (OpenAI 호환)

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `model` | string | 서버 모델 | 모델 ID |
| `messages` | array | 필수 | 대화 메시지 |
| `stream` | bool | false | SSE 스트리밍 |
| `temperature` | float | 1.0 | 샘플링 온도 |
| `top_p` | float | 1.0 | Nucleus sampling |
| `max_tokens` | int | 2048 | 최대 생성 토큰 |
| `max_completion_tokens` | int | - | max_tokens 별칭 |
| `stop` | string/array | null | 정지 시퀀스 |
| `seed` | int | null | 결정적 샘플링 |
| `presence_penalty` | float | 0 | 존재 패널티 |
| `frequency_penalty` | float | 0 | 빈도 패널티 |
| `enable_thinking` | bool | false | Thinking 모드 (Qwen3.5 전용) |

#### 웹 UI 연동

Continue, Open WebUI 등에서 OpenAI endpoint로 연결:
- URL: `http://<YOUR_LOCAL_IP>:8080/v1`
- API Key: 아무 값 (인증 없음)

#### 외부 네트워크에서 접속 (Tailscale)

```bash
brew install tailscale
# 양쪽 기기에서 Tailscale 로그인 후
curl http://<TAILSCALE_IP>:8080/v1/chat/completions ...
```

#### 아키텍처

```
클라이언트(:8080) → FastAPI 서버 (mlx_lm API 직접 호출)
```

- 요청 완료 후 KV 캐시 자동 해제 (`mx.clear_cache()`)
- `asyncio.Semaphore(1)` — GPU 순차 처리, HTTP는 동시 수신
- 대기 큐 5개 초과 시 429 응답 (OOM 방지)

### 4. 로깅

서버에 로깅이 **내장**되어 있습니다. 두 모델 모두 동일하게 기록.

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
| usage | 토큰 사용량 (prompt / completion / total) |

```bash
# 오늘 로그 보기
cat logs/$(date +%Y-%m-%d).jsonl | python3 -m json.tool

# IP별 호출 횟수
cat logs/*.jsonl | jq -r '.ip' | sort | uniq -c | sort -rn

# 느린 요청 찾기 (3초 이상)
cat logs/*.jsonl | jq 'select(.duration_ms > 3000)'
```

### 5. 벤치마크 (Qwen3.5)

```bash
source .venv/bin/activate

mlx_lm.benchmark \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt-tokens 256 \
  --generation-tokens 100 \
  --num-trials 3
```

### 커뮤니티 벤치마크 공유 (whatcani.run)

[whatcani.run](https://www.whatcani.run/)은 실사용자 실측 LLM 벤치마크 공유 플랫폼. MLX/llama.cpp 런타임 공식 지원.

```bash
bunx whatcanirun run \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --runtime mlx \
  --submit
```

---

## 컨텍스트 프로필 시스템 (Qwen3.5 전용)

> SuperGemma4는 128K 고정, 이 시스템 해당 없음.

262K(기본)와 1M(확장) 두 가지 프로필. `config.json` 교체 방식으로 전환.

```bash
# 채팅에서 전환
./llm-chat.sh 1m       # 1M 전환 후 채팅 시작
./llm-chat.sh 262k     # 262K 전환 후 채팅 시작

# API 서버에서 전환
./llm-server.sh 1m     # 1M 전환 후 서버 시작
./llm-server.sh 262k   # 262K 전환 후 서버 시작
```

| 프로필 | 컨텍스트 | 메모리 | 용도 |
|-------|-------:|------:|------|
| 262K (기본) | ~52만 글자 | ~22-25GB | 일반 대화, 코딩 |
| 1M (YaRN) | ~200만 글자 | ~34GB | 대형 문서/코드베이스 분석 |

- 전환 후 **재시작 필요** (Ctrl+C → 다시 실행)
- 1M 모드는 짧은 프롬프트에서 품질 약간 저하 가능 (Static YaRN 특성)
- 평소에는 262K로 충분. 정말 긴 문서 작업할 때만 1M 권장

### 동작 원리

YaRN(Yet another RoPE extensioN)으로 위치 인코딩을 스케일링.

| 설정 | 262K | 1M |
|------|------|-----|
| rope_type | `"default"` | `"yarn"` |
| factor | (없음) | `4.0` |
| original_max_position_embeddings | (없음) | `262144` |

---

## Thinking 모드 (Qwen3.5 전용)

> SuperGemma4에서 `enable_thinking`을 보내도 오류는 없지만 무시됩니다. 서버가 자동 감지 처리.

### Thinking ON

```
>> 123 * 456은?
<think>123 × 456을 계산해보겠습니다.
123 × 400 = 49,200
123 × 56 = 6,888
49,200 + 6,888 = 56,088</think>

123 × 456 = 56,088 입니다.
```

### 모드별 제어 방법

| 모드 | 방법 | 동작 |
|------|------|:---:|
| `llm-chat.sh` (대화형) | 프롬프트에 `/no_think` 추가 | O |
| `llm-server.sh` (API) | 기본값 Thinking OFF | O |
| `llm-server.sh --think` (API) | 기본값 Thinking ON | O |
| API 요청 `enable_thinking` | **요청별 제어 가능** | **O** |

### 언제 켜고 끌까?

| 상황 | Thinking | 이유 |
|------|:-------:|------|
| 수학/논리 | **ON** | 정확도 크게 향상 |
| 코딩 | **ON** | 단계적 사고로 버그 감소 |
| 간단한 질문 | OFF | 빠른 응답 |
| 번역/요약 | OFF | 생각 과정 불필요 |
| 창작/글쓰기 | OFF | 자연스러운 흐름 |

- Thinking 토큰도 `max_tokens`에 포함 → Thinking ON 시 8000+ 권장

---

## 파라미터 가이드

### Temperature

| 값 | 효과 | 용도 |
|---|------|------|
| 0.0 | 결정적 (항상 같은 답) | 코딩, 수학 |
| 0.3 | 약간의 변화 | 일반 대화 |
| 0.7 | 창의적 | 글쓰기, 브레인스토밍 |
| 1.0+ | 매우 랜덤 | 실험용 |

### Max Tokens

| 용도 | 권장 값 |
|------|------:|
| 짧은 답변 | 200 |
| 일반 대화 | 500 |
| 코드 생성 | 1000-2000 |
| 긴 문서 | 4000+ |
| Thinking ON | 8000+ |

---

## 편의 Alias (선택)

`~/.zshrc`에 추가하면 어디서든 실행 가능:

```bash
# local-llm alias
alias llm-chat='/path/to/local-llm/llm-chat.sh'
alias llm-server='/path/to/local-llm/llm-server.sh'
alias llm-gemma='/path/to/local-llm/llm-server.sh supergemma4'
alias llm-gen='/path/to/local-llm/.venv/bin/mlx_lm.generate --model mlx-community/Qwen3.5-35B-A3B-4bit'
alias llm-bench='/path/to/local-llm/.venv/bin/mlx_lm.benchmark --model mlx-community/Qwen3.5-35B-A3B-4bit'
```

```bash
source ~/.zshrc

llm-chat 1m              # Qwen3.5 1M 채팅
llm-server 1m            # Qwen3.5 API 서버
llm-gemma                # SuperGemma4 API 서버
```

---

## 메모리 관리

| 상태 | 메모리 사용 |
|------|--------:|
| 미실행 | ~21GB (시스템) |
| Qwen3.5 실행 중 | ~41GB |
| SuperGemma4 실행 중 | ~36GB |
| Ctrl+C 종료 후 | ~21GB (**즉시 해제**) |

- Apple Silicon Unified Memory — Ctrl+C로 종료하면 모델 메모리 즉시 반환
- 두 모델 전환 시: 반드시 Ctrl+C로 종료 후 재시작

```bash
# 프로세스 확인
ps aux | grep llm-api-server | grep -v grep

# 강제 종료
kill $(pgrep -f llm-api-server)

# 시스템 메모리 확인
top -l 1 -s 0 | grep PhysMem
```

---

## llmfit — 하드웨어 기반 모델 추천 도구

내 하드웨어에 맞는 LLM을 자동 추천해주는 도구.

```bash
brew install llmfit

llmfit system                                    # 시스템 사양 확인
llmfit fit                                       # 호환 모델 전체 추천
llmfit search qwen3.5                            # 특정 모델 검색
llmfit diff Qwen/Qwen3.5-35B-A3B Qwen/Qwen3.5-27B  # 두 모델 비교
```

### M5 Pro 64GB 추천 결과 (2026-03-22)

| 상태 | 모델 | Score | tok/s | 메모리% |
|:---:|------|:---:|------:|------:|
| Good | Qwen3-Coder-Next 80B-A3B | 99 | 105 | 64% |
| Perfect | Qwen3.5-122B-A10B (NVFP4) | 96 | 62 | 52% |
| **Perfect** | **Qwen3.5-35B-A3B** | **92** | **105** | **29%** |
| Perfect | GPT-OSS 20B | 91 | 64 | 17% |

---

## 성능 테스트 결과

### Qwen3.5-35B-A3B (2026-03-22, M5 Pro 64GB)

| 테스트 | 프롬프트 처리 | 생성 속도 | 피크 메모리 |
|-------|----------:|--------:|--------:|
| 한국어 인사 | 171 tok/s | 103 tok/s | 19.6GB |
| 영어 질문 | 122 tok/s | 104 tok/s | 19.6GB |

### SuperGemma4-26B (공개 벤치마크 기준)

| 항목 | 값 |
|------|-----|
| 생성 속도 (MLX 4bit) | ~46 tok/s |
| 프롬프트 처리 | ~328 tok/s |
| 런타임 메모리 | ~15GB |

---

## 메모리별 추천 모델 가이드

### MoE vs Dense

로컬 환경에서 4-bit 양자화 기준:

| | MoE | Dense |
|---|---|---|
| **속도** | 빠름 (활성 파라미터만 추론) | 느림 (전체 파라미터 추론) |
| **메모리 효율** | 높음 | 낮음 |
| **64GB 최적** | ✅ **Qwen3.5-35B-A3B (29%)** | Llama-3.3-70B (~70%, 빡빡) |

### Apple Silicon 메모리별 추천

| 메모리 | 추천 모델 | 메모리 사용 | 예상 속도 |
|------:|---------|--------:|--------:|
| **16GB** | Qwen3.5-9B (Q4) | ~6GB | 40+ tok/s |
| **24GB** | Qwen3.5-35B-A3B (Q3) | ~18GB | 80+ tok/s |
| **32GB** | Qwen3.5-35B-A3B (Q4) | ~22GB | 100+ tok/s |
| **64GB** | **Qwen3.5-35B-A3B (Q4)** — MoE 가성비 최적 | ~22GB | **103 tok/s** |
| **128GB** | Qwen3.5-122B-A10B (Q4) | ~70GB | ~15 tok/s |

### 용도별 추천

| 용도 | 추천 모델 | 이유 |
|------|---------|------|
| 한국어 + 코딩 올라운더 | **Qwen3.5-35B-A3B** | 속도+품질+메모리 밸런스 |
| 무검열 + 툴콜 강화 | **SuperGemma4 26B** 🔥 | 완전 무검열, 128K, 15GB |
| 긴 컨텍스트 메모리 절약 | **Qwen3.5-35B-A3B 1M** | YaRN으로 200만 글자 |
| 코딩 에이전트 | Qwen3-Coder-Next 80B-A3B | SWE-bench 최강 |
| 코딩 품질 최우선 | Qwen3.5-27B (Dense) | SWE 72.4, LiveCode 80.7 |
| 에이전트/도구 호출 | Qwen3.5-122B-A10B | BFCL 72.2 (128GB 필요) |
| 가볍고 빠르게 | Qwen3.5-9B | 6GB, 40+ tok/s |

### 벤치마크 비교

| 벤치마크 | 35B-A3B | SuperGemma4¹ | 27B | 122B-A10B | GPT-5 mini | Claude Sonnet 4.5 |
|---------|:-------:|:------------:|:---:|:---------:|:---------:|:-----------------:|
| MMLU-Pro | 85.3 | 82.6 | 86.1 | **86.7** | 83.7 | 80.8 |
| SWE-bench | 69.2 | - | **72.4** | 72.0 | 72.0 | 62.0 |
| LiveCodeBench | 74.6 | 77.1 | **80.7** | 78.9 | 80.5 | 82.7 |
| BFCL-V4 (도구) | 67.3 | 툴콜 2배↑² | 68.5 | **72.2** | 55.5 | 54.8 |
| GPQA Diamond | - | **82.3** | - | - | - | - |
| 생성 속도 (MLX) | **103 tok/s** | ~46 tok/s | ~15 tok/s | - | - | - |

> ¹ Gemma 4 26B-it 공식 벤치마크 기준. ² 자체 측정, 독립 검증 미완료.

---

## 테스트

```bash
# API 서버 테스트 (13개, mock 모델 — GPU 불필요)
.venv/bin/python -m pytest test_api_server.py -v
```

| 카테고리 | 테스트 | 검증 내용 |
|---------|-------|---------|
| Models | list_models | GET /v1/models 응답 형식 |
| Chat | basic_request | OpenAI 호환 응답 (id, choices, usage) |
| Chat | enable_thinking | Qwen3.5 Thinking ON/OFF 조건부 처리 |
| Chat | custom_parameters | temperature, top_p, max_tokens |
| Chat | max_completion_tokens | OpenAI 신규 파라미터 호환 |
| Stream | stream_format | SSE content-type, chunk 형식 |
| Stream | stream_chunks | role → content → finish_reason → [DONE] |
| RateLimit | queue_full | 큐 초과 시 429 응답 |
| Logging | log_file_created | JSONL 로그 파일 생성 + 필드 검증 |

---

## 참고 링크

- [MLX-LM GitHub](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [Qwen3.5-35B-A3B (MLX 4bit)](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit)
- [Qwen3.5-122B-A10B (MLX 4bit)](https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-4bit)
- [Qwen3.5 공식 GitHub](https://github.com/QwenLM/Qwen3.5)
- [Qwen3.5 로컬 가이드 (Unsloth)](https://unsloth.ai/docs/models/qwen3.5)
- [Unsloth GGUF Benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)
- [SuperGemma4 26B uncensored MLX 4bit (v2)](https://huggingface.co/Jiunsong/supergemma4-26b-uncensored-mlx-4bit-v2)
- [SuperGemma4 26B abliterated multimodal MLX 4bit](https://huggingface.co/Jiunsong/supergemma4-26b-abliterated-multimodal-mlx-4bit)
- [SuperGemma4 26B uncensored GGUF v2](https://huggingface.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2)
- [SuperGemma4 26B uncensored GGUF v2 (Ollama)](https://ollama.com/0xIbra/supergemma4-26b-uncensored-gguf-v2)
- [SuperGemma4 31B abliterated MLX 4bit](https://huggingface.co/Jiunsong/SuperGemma4-31b-abliterated-mlx-4bit)
- [SuperGemma4 31B abliterated GGUF](https://huggingface.co/Jiunsong/SuperGemma4-31b-abliterated-GGUF)
- [Gemma 4 공식 블로그 (Google)](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [TurboQuant (Google Research)](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [Gemma 4 on Ollama](https://ollama.com/library/gemma4)
- [Apple MLX + M5 리서치](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [M5 Pro/Max 로컬 LLM 가이드](https://modelfit.io/blog/m5-pro-max-local-llm-2026/)
- [whatcani.run — Apple Silicon 실측 LLM 벤치마크](https://www.whatcani.run/)

---

## License

[MIT License](LICENSE)
