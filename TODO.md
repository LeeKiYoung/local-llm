# TODO — Local LLM 개선 계획

## 다음 작업 후보 (2026-08-23 실측 기반)

전제: MTP·프리픽스 캐시·스트리밍은 이미 동작한다. 상세 수치는
`dflash2-benchmark-2026-08-23.md` 참조. 아래는 **실측으로 확인된 병목**만 남긴 것.

계측이 이미 로그에 있으므로 추측하지 말고 먼저 로그를 볼 것:
`logs/YYYY-MM-DD.jsonl`의 `queue_wait_ms` / `prefill_tokens` / `cached_tokens` / `decode_tps`.

---

### 1. 프리픽스 캐시 슬롯 다중화 (효과 가장 큼)

**문제:** `prompt_cache_state`가 전역 단일 슬롯이라, 다른 대화나 부수 요청(dsh 세션 타이틀
생성 등)이 끼면 캐시가 덮어써지고 다음 턴이 전체 재프리필된다.

**실측:** 정상 히트 시 34K 컨텍스트가 `prefill=156 cached=34478` → **8.9초**.
슬롯을 뺏긴 직후 같은 대화가 `cached=0`으로 38K 전체 재프리필 → **130초**. 14배 차이.

**접근:** 대화별 LRU dict로 전환. 키는 messages 프리픽스 해시.
- KV 캐시는 크다(38K면 수 GB) → 슬롯 수 상한 + 메모리 상한 필수
- **KV 캐시 배열은 MLX 스트림에 묶인다** → 생성·해제 모두 `_gpu_executor` 스레드에서
  (`project_mlx_gotchas` 함정 2와 같은 이유). `_eval_kv_cache()` 패턴 유지
- APC(`apc_manager`)가 원래 이 역할이지만 이 모델 계열에선 `exact` 모드로 떨어져 무용

**검증:** 두 대화를 번갈아 요청 → 양쪽 모두 `cached_tokens > 0` 유지되는지

**코드:** `llm-api-server.py` 전역 `prompt_cache_state`,
`_run_inference_inner` / `_run_inference_streaming_inner`의 `prompt_cache_state=` 인자

---

### 2. 동시 요청 직렬화 완화

**문제:** `gpu_semaphore = asyncio.Semaphore(1)`로 GPU를 하나씩만 쓴다. 짧은 요청이
앞선 큰 요청을 통째로 기다린다.

**실측:** `ctx=133 gen=8`이 큐 대기만 **56초**(조용할 때 동일 크기는 1.7초). 최대 30배 지연.
dsh는 본 응답과 부수 요청을 병렬로 던지므로 상시 발생.

**접근 A (근본):** mlx-vlm의 continuous batching 사용. `mlx_vlm.server`는 `max_num_seqs`로
여러 시퀀스를 동시 디코딩한다. 단 우리는 `stream_generate`를 직접 호출하는 구조라
엔진 교체급 변경이 된다 (프록시화 또는 generation 루프 이식).
**접근 B (완화):** 짧은 요청(예상 토큰 수 기준) 우선 큐. 간단하지만 근본 해결은 아님.

**검증:** 긴 요청 진행 중 짧은 요청을 던져 `queue_wait_ms` 비교

---

### 3. 실사용 드래프터 적중률 (원인 미해명)

**문제:** 합성 프롬프트에서는 MTP가 2.24배(4.7K)인데, dsh 실사용 38K 구간은
`decode_tps` 7.0~8.3으로 **MTP를 끈 값(8.1)과 동일**했다. 실제 대화에서는 드래프터가
거의 기여하지 못한다.

**가설:** 합성 프롬프트는 같은 문장 반복이라 예측이 쉽다. 실제 대화는 도구 출력·JSON·
에러 로그·파일 목록이 섞여 다음 토큰 예측이 무너진다.

**접근:** 먼저 acceptance rate를 계측에 노출한다.
`mlx_vlm/speculative/common.py`의 `speculative_stats_snapshot()` /
`speculative_stats_since()`가 있으니 이걸 `_record_metrics()`에 추가.
그 값이 실제로 낮은지 확인한 뒤에야 대책(draft_block_size 튜닝, 도구 출력 비중이 높은
요청에서 MTP 비활성화 등)을 논의할 수 있다.

**주의:** README의 2.64× / 3.16×는 합성 프롬프트 기준값이다. 실사용 상한이 아니다.

---

### 이미 확인된 것 (재조사 불필요)

| 항목 | 결과 |
|---|---|
| MTP 동작 | 정상. 36K에서도 1.85배 기여 |
| 컨텍스트 길이 영향 | 작다. MTP OFF 기준 4.7K→36K에서 16% 감소뿐 |
| 프리픽스 캐시 자체 | 매우 효과적 (14배), 부분 히트도 정상 |
| 스트리밍 | thinking OFF 버퍼링 버그 수정 완료 (청크 3→33) |
| 이미지 + MTP | 정상 동작, 함께 가속됨 |
| dFlash2 (mlx-dspark) | 텍스트 전용, 이미지에서 무음 환각 → 미채택 |

---

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
