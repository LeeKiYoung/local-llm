# TODO — Local LLM 개선 계획

## 자체 API 서버 (요청별 Thinking 제어)

### 배경
- mlx_lm.server는 요청별 `enable_thinking` 파라미터를 무시함
- 서버 시작 시 `--chat-template-args`로만 Thinking ON/OFF 가능
- 용도별로 서버를 재시작해야 하는 불편함

### 목표
- OpenAI 호환 API 서버를 직접 구현
- 요청별로 `enable_thinking: true/false` 동적 제어
- 하나의 서버에서 Thinking ON/OFF 모두 지원

### 설계

```
클라이언트 → FastAPI 서버(:8080) → mlx_lm Python API (직접 호출)
```

#### 핵심 구조
```python
from mlx_lm import load, generate

# 모델은 한 번만 로드 (메모리 19.6GB)
model, tokenizer = load("mlx-community/Qwen3.5-35B-A3B-4bit")

@app.post("/v1/chat/completions")
async def chat(request):
    enable_thinking = request.get("enable_thinking", False)

    # 요청별로 채팅 템플릿 동적 변경
    template_config = {"enable_thinking": enable_thinking}

    # mlx_lm.generate() 직접 호출
    response = generate(model, tokenizer, prompt,
                       chat_template_config=template_config)

    # OpenAI 호환 응답 형식으로 반환
    return {"choices": [{"message": {"content": response}}]}
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
4. llm-server.sh 업데이트
   - 기존: mlx_lm.server 실행
   - 변경: FastAPI 서버 실행

#### 장점
- 서버 재시작 없이 요청별 Thinking ON/OFF
- 프록시 불필요 (직접 서버가 모든 기능 처리)
- 스트리밍 등 추가 기능 구현 자유도

#### 참고
- mlx_lm Python API: https://github.com/ml-explore/mlx-examples/tree/main/llms
- FastAPI: https://fastapi.tiangolo.com
