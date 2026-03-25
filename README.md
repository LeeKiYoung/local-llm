# Local LLM — Qwen3.5-35B-A3B (MLX)

> **Apple Silicon Mac 전용** (M1/M2/M3/M4/M5)
> MLX는 Apple의 네이티브 ML 프레임워크로, Apple Silicon에서만 동작합니다.
> NVIDIA GPU / Intel Mac / Windows / Linux는 지원하지 않습니다.

Apple Silicon Mac에서 Qwen3.5-35B-A3B를 MLX로 돌리기 위한 셋업 가이드.

## 요구사항

| 항목 | 최소 | 권장 |
|------|------|------|
| Mac | Apple Silicon (M1+) | M3 Pro / M4 Pro 이상 |
| 메모리 | 24GB | 64GB |
| Python | 3.10+ | 3.11+ |
| 디스크 | 20GB 여유 | 40GB+ |

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
| 1 | **Qwen3.5-35B-A3B** | ~20GB | 103 tok/s | 24GB+ | 한국어+코딩+비전 올라운더 |
| 2 | Qwen3.5-9B | ~6GB | 40+ tok/s | 16GB+ | 가볍고 빠름 |
| 3 | Qwen3.5-27B | ~17GB | 15 tok/s | 24GB+ | Dense, 코딩 벤치마크 최강 |
| 4 | Qwen3-Coder-Next-80B | ~15GB | 25+ tok/s | 24GB+ | 코딩 에이전트 특화 |
| 5 | 직접 입력 | - | - | - | Hugging Face 모델 ID |

메모리에 따라 자동 추천이 표시됩니다.

Enter만 누르면 추천 모델(Qwen3.5-35B-A3B)로 설치됩니다.

### 환경만 셋업 (모델 나중에)

```bash
./setup.sh --no-model
```

### 모델은 어디에 저장되나요?

첫 실행 시 모델(~19GB)이 자동 다운로드되며, 기본적으로 아래 경로에 저장됩니다:

```
~/.cache/huggingface/hub/    (macOS/Linux 공통)
```

이 경로를 바꾸고 싶다면 (예: 내장 디스크 용량이 부족해서 외장 SSD에 저장하고 싶은 경우):

```bash
# ~/.zshrc에 추가
export HF_HOME=/Volumes/MySSD/.huggingface

# 터미널 재시작 후 셋업
source ~/.zshrc
./setup.sh
```

`HF_HOME`은 Hugging Face에서 제공하는 공식 환경변수로,
모델/토크나이저 등 모든 캐시 파일의 저장 위치를 지정합니다.
설정하지 않으면 기본 경로(`~/.cache/huggingface`)를 사용합니다.

---

## 프로젝트 구조

```
local-llm/
├── setup.sh                        # 자동 셋업 (환경 + 모델 + 의존성)
├── llm-chat.sh                     # 대화형 채팅 (프로필 전환 포함)
├── llm-server.sh                   # API 서버 실행 (프로필 전환 포함)
├── llm-api-server.py               # FastAPI 커스텀 API 서버 (핵심)
├── profiles/
│   ├── config-262k.json            # 기본 프로필 (262K 컨텍스트)
│   └── config-1m.json              # 확장 프로필 (1M 컨텍스트, YaRN)
├── test_api_server.py              # API 서버 테스트 (13개)
├── local-llm-guide-2026.md         # 모델 비교 가이드 문서
├── .venv/                          # Python 가상환경 (mlx-lm, fastapi, uvicorn)
└── logs/                           # 요청/응답 JSONL 로그 (자동 생성)

# 모델 캐시 (자동 다운로드됨)
~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/  (~19GB)
```

## 현재 모델 스펙

| 항목 | 값 |
|------|-----|
| 모델 | Qwen3.5-35B-A3B |
| 양자화 | 4-bit (MLX 네이티브) |
| 디스크 크기 | ~19GB |
| 런타임 메모리 | ~19.6GB |
| 구조 | MoE (256 전문가, 토큰당 3B 활성) |
| 컨텍스트 | 262K (1M 확장 가능) |
| 생성 속도 | ~103 tok/s |
| 프롬프트 처리 | ~170 tok/s |

---

## 빠른 시작 (Quick Start)

### 1단계: 셋업

```bash
git clone https://github.com/LeeKiYoung/local-llm.git
cd local-llm
./setup.sh
```

### 2단계: 채팅 시작

```bash
./llm-chat.sh
```

### 3단계: 대화하기

```
📍 현재: 262K 컨텍스트 (기본)

🚀 채팅 시작 (종료: Ctrl+C)

>> 안녕! 자기소개 해줘
안녕하세요! 저는 Qwen3.5입니다. 다양한 질문에 답하고
코딩, 번역, 문서 분석 등을 도와드릴 수 있어요.

>> 파이썬으로 피보나치 함수 짜줘
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

>> (Ctrl+C로 종료)
```

### 4단계: 긴 문서 분석이 필요할 때

```bash
./llm-chat.sh 1m
```

```
✅ 1M 컨텍스트 (YaRN) 적용 완료

🚀 채팅 시작 (종료: Ctrl+C)

>> (긴 코드나 문서를 붙여넣고 질문)
```

### 5단계: 다시 평소 모드로

```bash
./llm-chat.sh          # 또는 ./llm-chat.sh 262k
```

---

## 사용 모드별 가이드

### 1. 대화형 채팅 (llm-chat.sh)

가장 기본적인 사용법. 스크립트가 프로필 전환 + 채팅 시작을 한번에 처리.

```bash
./llm-chat.sh           # 평소 (262K)
./llm-chat.sh 1m        # 긴 문서 모드 (1M)
./llm-chat.sh 262k      # 명시적 262K
```

추가 옵션도 뒤에 붙일 수 있음:

```bash
./llm-chat.sh 1m --temp 0.3              # 1M + 낮은 temperature
./llm-chat.sh --max-tokens 4000          # 긴 응답 허용
./llm-chat.sh --system-prompt "한국어로만 답해줘"  # 시스템 프롬프트
```

### 2. 단발 생성 (mlx_lm.generate)

한 번의 프롬프트 → 한 번의 응답. 스크립트나 자동화에 유용.

```bash
# 가상환경 활성화 후
source .venv/bin/activate

mlx_lm.generate \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt "Python으로 퀵소트 구현해줘" \
  --max-tokens 500

# Thinking 모드 끄고 싶으면 프롬프트에 /no_think 추가
mlx_lm.generate \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt "안녕! /no_think" \
  --max-tokens 200
```

### 3. API 서버 (llm-server.sh)

OpenAI 호환 API 서버. FastAPI + mlx_lm Python API로 직접 추론.
같은 네트워크의 다른 기기(맥미니 등)에서 접속 가능.

```bash
./llm-server.sh              # 기본 (Thinking OFF)
./llm-server.sh --think      # Thinking ON (수학/코딩 정확도 향상)
./llm-server.sh 1m           # 1M 컨텍스트
./llm-server.sh 1m --think   # 1M + Thinking ON
./llm-server.sh 262k 9090    # 포트 지정
```

실행하면:

```
📍 현재: 262K 컨텍스트 (기본)

🌐 API 서버 시작
   로컬:     http://localhost:8080
   네트워크: http://<YOUR_LOCAL_IP>:8080

   엔드포인트: /v1/chat/completions
   스트리밍: stream=true 지원
   🧠 Thinking: OFF (기본, 요청별 override 가능)
```

#### 기본 호출

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-35B-A3B-4bit",
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

#### 요청별 Thinking 제어

```bash
# Thinking ON (기본 OFF라도 이 요청만 ON)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "123*456=?"}],
    "enable_thinking": true,
    "max_tokens": 500
  }'
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
| `enable_thinking` | bool | false | Thinking 모드 (확장) |

#### 웹 UI 연동

Continue, Open WebUI 등에서 OpenAI endpoint로 연결:
- URL: `http://<YOUR_LOCAL_IP>:8080/v1`
- API Key: 아무 값 (인증 없음)

#### 외부 네트워크에서 접속 (Tailscale)

[Tailscale](https://tailscale.com)을 사용하면 외부 네트워크(카페, 회사 등)에서도 LLM 서버에 접속할 수 있습니다.
포트포워딩 없이 VPN 기반으로 안전하게 연결됩니다.

```bash
brew install tailscale
# 양쪽 기기에서 Tailscale 로그인 후
curl http://<TAILSCALE_IP>:8080/v1/chat/completions ...
```

#### 아키텍처

```
이전: 클라이언트(:8080) → 로깅 프록시 → mlx_lm.server(:8081)  (KV 캐시 누적)
현재: 클라이언트(:8080) → FastAPI 서버 (mlx_lm API 직접 호출)  (요청 후 자동 해제)
```

- 요청 완료 후 KV 캐시 자동 해제 (메모리 누적 없음)
- `asyncio.Semaphore(1)` — GPU 순차 처리, HTTP는 동시 수신
- 대기 큐 5개 초과 시 429 응답 (OOM 방지)

### 4. 로깅

서버에 로깅이 **내장**되어 있습니다. 별도 프록시 없이 요청/응답을 자동 기록.

로그는 `logs/` 폴더에 일별 JSONL 파일로 저장:

```
logs/
├── 2026-03-23.jsonl
├── 2026-03-24.jsonl
└── ...
```

각 로그 항목에 포함되는 정보:

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

로그 분석 예시:

```bash
# 오늘 로그 보기
cat logs/$(date +%Y-%m-%d).jsonl | python3 -m json.tool

# IP별 호출 횟수
cat logs/*.jsonl | jq -r '.ip' | sort | uniq -c | sort -rn

# 느린 요청 찾기 (3초 이상)
cat logs/*.jsonl | jq 'select(.duration_ms > 3000)'
```

### 5. 벤치마크 (mlx_lm.benchmark)

```bash
source .venv/bin/activate

mlx_lm.benchmark \
  --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt-tokens 256 \
  --generation-tokens 100 \
  --num-trials 3
```

---

## 컨텍스트 프로필 시스템

262K(기본)와 1M(확장) 두 가지 프로필을 미리 준비해둠.
config.json을 교체하는 방식으로 전환.

### 전환 방법

```bash
# 채팅에서 전환
./llm-chat.sh 1m       # 1M 전환 후 채팅 시작
./llm-chat.sh 262k     # 262K 전환 후 채팅 시작

# API 서버에서 전환
./llm-server.sh 1m     # 1M 전환 후 서버 시작
./llm-server.sh 262k   # 262K 전환 후 서버 시작

# 현재 프로필 확인
./llm-chat.sh status
```

### 프로필 비교

| 프로필 | 컨텍스트 | 메모리 | 용도 |
|-------|-------:|------:|------|
| 262K (기본) | ~52만 글자 | ~22-25GB | 일반 대화, 코딩 |
| 1M (YaRN) | ~200만 글자 | ~34GB | 대형 문서/코드베이스 분석 |

### 주의사항

- 전환 후 **재시작 필요** (Ctrl+C → 다시 실행)
- 1M 모드는 짧은 프롬프트에서 품질 약간 저하 가능 (Static YaRN 특성)
- 평소에는 262K로 충분. 정말 긴 문서 작업할 때만 1M 사용 권장

### 동작 원리 (참고)

YaRN(Yet another RoPE extensioN)으로 위치 인코딩을 스케일링.
`profiles/` 폴더의 config.json에서 `rope_type`과 `factor`만 다름:

| 설정 | 262K | 1M |
|------|------|-----|
| rope_type | `"default"` | `"yarn"` |
| factor | (없음) | `4.0` |
| original_max_position_embeddings | (없음) | `262144` |

---

## Thinking 모드

Qwen3.5는 기본적으로 **Thinking 모드 ON**. 답변 전에 추론 과정을 거침.

### Thinking ON (기본)

```
>> 123 * 456은?
<think>123 × 456을 계산해보겠습니다.
123 × 400 = 49,200
123 × 56 = 6,888
49,200 + 6,888 = 56,088</think>

123 × 456 = 56,088 입니다.
```

### Thinking OFF

끄는 방법이 **사용 모드에 따라 다릅니다:**

| 모드 | 방법 | 동작 |
|------|------|:---:|
| `llm-chat.sh` (대화형) | 프롬프트에 `/no_think` 추가 | O |
| `llm-server.sh` (API) | 기본값 Thinking OFF | O |
| `llm-server.sh --think` (API) | 기본값 Thinking ON | O |
| API 요청 `enable_thinking` | **요청별 제어 가능** | **O** |

#### 대화형 채팅에서 끄기

```
>> 안녕! /no_think
안녕하세요! 무엇을 도와드릴까요?
```

#### API 서버에서 제어 (요청별)

커스텀 서버에서는 **요청별 Thinking 제어**가 가능합니다:

```bash
# 서버 기본값 OFF → 이 요청만 ON
curl http://localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"1+1=?"}],"enable_thinking":true}'

# 서버 기본값 ON (--think) → 이 요청만 OFF
curl http://localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"안녕!"}],"enable_thinking":false}'
```

서버 기본값은 `--think` 플래그로 설정하고, 개별 요청에서 `enable_thinking`으로 override 가능.

### 언제 켜고 끌까?

| 상황 | Thinking | 이유 |
|------|:-------:|------|
| 수학/논리 | **ON** | 정확도 크게 향상 |
| 코딩 | **ON** | 단계적 사고로 버그 감소 |
| 간단한 질문 | OFF | 빠른 응답 |
| 번역/요약 | OFF | 생각 과정 불필요 |
| 창작/글쓰기 | OFF | 자연스러운 흐름 |

### 주의

- Thinking 토큰도 max-tokens에 포함됨 → 넉넉하게 설정 (Thinking ON일 때 8000+ 권장)
- Q4 양자화에서는 안정적으로 동작
- Q2/Q3에서는 Thinking이 길어져 32K 제한에 걸릴 수 있음

---

## 파라미터 가이드

### Temperature (--temp)

| 값 | 효과 | 용도 |
|---|------|------|
| 0.0 | 결정적 (항상 같은 답) | 코딩, 수학 |
| 0.3 | 약간의 변화 | 일반 대화 |
| 0.7 | 창의적 (기본값) | 글쓰기, 브레인스토밍 |
| 1.0+ | 매우 랜덤 | 실험용 |

### Top-p (--top-p)

확률 상위 p%의 토큰만 고려. 기본값 1.0. 0.9로 낮추면 집중된 답변.

### Max Tokens (--max-tokens / -m)

| 용도 | 권장 값 |
|------|------:|
| 짧은 답변 | 200 |
| 일반 대화 | 500 |
| 코드 생성 | 1000-2000 |
| 긴 문서 | 4000+ |

### KV Cache (--max-kv-size)

컨텍스트 윈도우 크기 제한. 메모리 절약이 필요하면:

```bash
mlx_lm.chat --model mlx-community/Qwen3.5-35B-A3B-4bit \
  --max-kv-size 32768   # 32K로 제한
```

---

## 편의 Alias (선택)

`~/.zshrc`에 추가하면 어디서든 실행 가능:

```bash
# local-llm alias
alias llm-chat='/path/to/local-llm/llm-chat.sh'
alias llm-gen='/path/to/local-llm/.venv/bin/mlx_lm.generate --model mlx-community/Qwen3.5-35B-A3B-4bit'
alias llm-server='/path/to/local-llm/llm-server.sh'
alias llm-bench='/path/to/local-llm/.venv/bin/mlx_lm.benchmark --model mlx-community/Qwen3.5-35B-A3B-4bit'
```

설정 후:

```bash
source ~/.zshrc

llm-chat                 # 평소 채팅
llm-chat 1m              # 1M 컨텍스트 채팅
llm-chat 1m --temp 0.3   # 1M + 낮은 temperature
llm-gen --prompt "Hello" --max-tokens 100
llm-server               # API 서버 시작
llm-chat status          # 프로필 확인
```

---

## 메모리 관리

### 모델 로드/해제

| 상태 | 메모리 사용 | 설명 |
|------|--------:|------|
| 모델 미실행 | ~21GB | 시스템 기본 |
| 채팅/서버 실행 중 | ~41GB | 시스템 21GB + 모델 19.6GB |
| Ctrl+C 종료 후 | ~21GB | **즉시 해제** |

- **Ctrl+C로 종료하면 모델 메모리(~19.6GB)가 즉시 반환됨**
- Apple Silicon Unified Memory라 GPU VRAM 별도 해제 걱정 없음
- 프로세스 종료 = 메모리 즉시 반환

### 모델이 메모리에 남아있는지 확인

```bash
# 프로세스 확인
ps aux | grep llm-api-server | grep -v grep

# 아무것도 안 나오면 → 메모리 해제된 상태
# 프로세스가 보이면 → 아직 실행 중 (kill로 강제 종료 가능)
kill $(pgrep -f llm-api-server)
```

### 시스템 메모리 확인

```bash
top -l 1 -s 0 | grep PhysMem
```

---

## llmfit — 하드웨어 기반 모델 추천 도구

내 하드웨어에 맞는 LLM을 자동 추천해주는 도구. 모델 실행은 안 하고 분석/추천만 함.

### 설치

```bash
brew install llmfit
```

### 사용법

```bash
# 시스템 사양 확인
llmfit system

# 내 하드웨어에 맞는 모델 추천 (전체 목록)
llmfit fit

# 특정 모델 검색
llmfit search qwen3.5

# 모델 상세 정보
llmfit info Qwen/Qwen3.5-35B-A3B

# 두 모델 비교
llmfit diff Qwen/Qwen3.5-35B-A3B Qwen/Qwen3.5-27B
```

### M5 Pro 64GB 추천 결과 (2026-03-22)

501개 호환 모델 중 상위:

| 상태 | 모델 | Score | tok/s | 메모리% |
|:---:|------|:---:|------:|------:|
| Good | Qwen3-Coder-Next 80B-A3B | 99 | 105 | 64% |
| Perfect | Qwen3.5-122B-A10B (NVFP4) | 96 | 62 | 52% |
| Perfect | Qwen3-Coder 30B-A3B | 95 | 70 | 24% |
| **Perfect** | **Qwen3.5-35B-A3B (현재 사용)** | **92** | **105** | **29%** |
| Perfect | GPT-OSS 20B | 91 | 64 | 17% |
| Marginal | GPT-OSS 120B | 83 | 33 | 93% |

- Perfect = 메모리 여유 충분 / Good = 가능하지만 빠듯 / Marginal = 타이트
- 우리 모델(35B-A3B): Perfect 등급, 메모리 29%, 예측 105 tok/s (실측 103 tok/s와 일치)

---

## 성능 테스트 결과 (2026-03-22)

| 테스트 | 프롬프트 처리 | 생성 속도 | 피크 메모리 |
|-------|----------:|--------:|--------:|
| 한국어 인사 | 171 tok/s | 103 tok/s | 19.6GB |
| 영어 질문 | 122 tok/s | 104 tok/s | 19.6GB |

---

## 메모리별 추천 모델 가이드

### Apple Silicon 메모리별 추천

| 메모리 | 추천 모델 | 양자화 | 메모리 사용 | 예상 속도 |
|------:|---------|:-----:|--------:|--------:|
| **8GB** | Qwen3.5-0.8B | Q4 | ~1GB | 200+ tok/s |
| **16GB** | Qwen3.5-9B | Q4 | ~6GB | 40+ tok/s |
| **24GB** | Qwen3.5-35B-A3B | Q3 | ~18GB | 80+ tok/s |
| **32GB** | Qwen3.5-35B-A3B | Q4 | ~22GB | 100+ tok/s |
| **48GB** | Qwen3.5-35B-A3B | Q6 | ~30GB | 80+ tok/s |
| **64GB** | Qwen3.5-35B-A3B (Q4) + 여유 | Q4 | ~22GB | **103 tok/s** |
| **64GB** | Qwen3.5-122B-A10B | Q4 | ~40GB | ~10 tok/s |
| **128GB** | Qwen3.5-122B-A10B | Q8 | ~70GB | ~15 tok/s |
| **128GB+** | Qwen3.5-397B (flash-moe) | Q4 | 5.5GB+SSD | ~4 tok/s |

### 용도별 추천

| 용도 | 추천 모델 | 이유 |
|------|---------|------|
| 코딩 에이전트 | Qwen3-Coder-Next 80B-A3B | SWE-bench, 에이전트 최강 |
| 한국어 + 코딩 올라운더 | **Qwen3.5-35B-A3B** | 속도+품질+메모리 밸런스 최고 |
| 코딩 품질 최우선 | Qwen3.5-27B (Dense) | SWE 72.4, LiveCode 80.7 |
| 에이전트/도구 호출 | Qwen3.5-122B-A10B | BFCL 72.2, 도구 사용 압도적 |
| 최고 지능 | Qwen3.5-397B (flash-moe) | SSD 스트리밍, 빌드 필요 |
| 가볍고 빠르게 | Qwen3.5-9B | 6GB, 40+ tok/s |

### Qwen3.5-35B-A3B 양자화별 비교

| 양자화 | 디스크 | 메모리 (4K) | 메모리 (262K) | PPL | 품질 손실 |
|-------|------:|--------:|----------:|----:|------:|
| Q2_K | ~12GB | ~14GB | ~17GB | 7.04 | 큼 |
| Q3_K_M | ~16GB | ~18GB | ~21GB | 6.73 | 보통 |
| **Q4_K_M** | **~19GB** | **~22GB** | **~25GB** | **6.61** | **미미** |
| Q6_K | ~28GB | ~30GB | ~33GB | 6.54 | 거의 없음 |
| Q8 | ~36GB | ~38GB | ~41GB | 6.54 | 무시 가능 |

### 벤치마크 비교 (주요 모델)

| 벤치마크 | 35B-A3B | 27B | 122B-A10B | GPT-5 mini | Claude Sonnet 4.5 |
|---------|:-------:|:---:|:---------:|:---------:|:-----------------:|
| MMLU-Pro | 85.3 | 86.1 | **86.7** | 83.7 | 80.8 |
| SWE-bench | 69.2 | **72.4** | 72.0 | 72.0 | 62.0 |
| LiveCodeBench | 74.6 | **80.7** | 78.9 | 80.5 | 82.7 |
| BFCL-V4 (도구) | 67.3 | 68.5 | **72.2** | 55.5 | 54.8 |
| MMMU-Pro (비전) | 68.4 | 67.3 | **76.9** | 67.3 | 75.0 |

---

## 테스트

```bash
# API 서버 테스트 (13개)
.venv/bin/python -m pytest test_api_server.py -v

# 레거시 프록시 테스트
.venv/bin/python test_proxy.py
```

API 서버 테스트 (mock 모델, GPU 불필요):

| 카테고리 | 테스트 | 검증 내용 |
|---------|-------|---------|
| Models | list_models | GET /v1/models 응답 형식 |
| Chat | basic_request | OpenAI 호환 응답 (id, choices, usage) |
| Chat | enable_thinking | 요청별 Thinking ON/OFF |
| Chat | custom_parameters | temperature, top_p, max_tokens |
| Chat | max_completion_tokens | OpenAI 신규 파라미터 호환 |
| Stream | stream_format | SSE content-type, chunk 형식 |
| Stream | stream_chunks | role → content → finish_reason → [DONE] |
| RateLimit | queue_full | 큐 초과 시 429 응답 |
| Logging | log_file_created | JSONL 로그 파일 생성 + 필드 검증 |

---

## 참고 링크

- [MLX-LM GitHub](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [mlx-community/Qwen3.5-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit)
- [Qwen3.5 공식 GitHub](https://github.com/QwenLM/Qwen3.5)
- [Unsloth GGUF Benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)

---

## License

[MIT License](LICENSE)
