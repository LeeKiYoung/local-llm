# TODO — Local LLM 개선 계획

## 자체 API 서버 (요청별 Thinking 제어)

### 현재 vs 커스텀 서버

| | 현재 (mlx_lm.server) | 커스텀 서버 |
|---|:---:|:---:|
| 요청별 Thinking ON/OFF | **X (재시작 필요)** | **O** |
| 프록시 없이 로깅 | X (프록시 경유) | O (내장) |
| 병렬 추론 | X | **X (GPU 1개라 동일)** |
| 구현 난이도 | 이미 완료 | 추가 개발 필요 |

> **결론:** 차이는 "요청별 Thinking 제어" 하나.
> Thinking 전환이 자주 필요해지면 그때 구현.
> 지금은 `./llm-server.sh` / `./llm-server.sh --think` 재시작 방식으로 충분.

### 배경
- mlx_lm.server는 요청별 `enable_thinking` 파라미터를 무시함
- 서버 시작 시 `--chat-template-args`로만 Thinking ON/OFF 가능
- API 요청의 `enable_thinking: false`는 thinking을 content로 옮기기만 할 뿐 실제로 끄지 않음

### 설계

```
클라이언트 → FastAPI 서버(:8080) → mlx_lm Python API (직접 호출)
```

#### 핵심 원리
- Thinking ON/OFF는 **채팅 템플릿 포맷팅 단계**에서 결정됨 (모델 가중치 문제 아님)
- 모델은 한 번만 로드하고, 요청마다 다른 템플릿을 적용하면 됨
- 모델 재로드 불필요

#### 동시성 전략
- Apple Silicon GPU는 순차 처리 → 병렬 추론 이점 없음
- `asyncio.Semaphore(1)` + 비동기 큐가 최적
- HTTP 요청은 동시에 받되, GPU 추론은 하나씩 순서대로
- 대기 큐 제한 (MAX_QUEUE=5)으로 OOM 방지

#### 메모리 구조
```
고정: 모델 가중치 ~19.6GB
변동: KV 캐시 (요청당 ~수십MB ~ 수GB, 컨텍스트 길이에 따라)
총 메모리 = 19.6GB + (동시 요청 수 × KV 캐시 크기)
```

#### 핵심 코드

```python
import asyncio
from fastapi import FastAPI
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3.5-35B-A3B-4bit")
gpu_semaphore = asyncio.Semaphore(1)
MAX_QUEUE = 5
pending = 0

@app.post("/v1/chat/completions")
async def chat(request):
    if pending >= MAX_QUEUE:
        return JSONResponse(status_code=429, content={"error": "서버 바쁨"})

    pending += 1
    async with gpu_semaphore:
        response = await run_inference(
            model, tokenizer, request,
            enable_thinking=request.get("enable_thinking", False)
        )
    pending -= 1
    return response
```

#### 필요 작업
1. FastAPI 서버 (`llm-api-server.py`)
   - OpenAI 호환 `/v1/chat/completions` 엔드포인트
   - `/v1/models` 엔드포인트
   - 스트리밍 지원 (SSE)
2. 요청별 Thinking 제어
   - `enable_thinking` 파라미터를 채팅 템플릿에 동적 주입
   - 모델 재로드 없이 템플릿만 변경
3. 기존 기능 유지
   - 로깅 (JSONL)
   - caffeinate (절전 방지)
   - 컨텍스트 프로필 (262K/1M)

#### 참고
- mlx_lm Python API: https://github.com/ml-explore/mlx-examples/tree/main/llms
- FastAPI: https://fastapi.tiangolo.com
