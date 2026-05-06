"""
커스텀 LLM API 서버
FastAPI + mlx_vlm Python API 직접 호출

특징:
- OpenAI API 호환 (/v1/chat/completions, /v1/models)
- 멀티모달 이미지 처리 (base64 data URI, HTTP URL)
- 요청별 enable_thinking / preserve_thinking 제어
- 요청 완료 후 KV 캐시 자동 해제
- 내장 JSONL 로깅
"""

import argparse
import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from functools import partial

import mlx.core as mx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import base64
import io
from PIL import Image
from mlx_vlm import load, generate, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.generate import PromptCacheState
from mlx_vlm.vision_cache import VisionFeatureCache

# 압축 폭탄(decompression bomb) 방지 — 50MP 이상 이미지 거부 (PIL 기본값 약 178MP)
Image.MAX_IMAGE_PIXELS = 50_000_000

# ── 글로벌 ────────────────────────────────────────
app = FastAPI()
model = None
processor = None
model_id = ""
gpu_semaphore = None
pending = 0
MAX_QUEUE = 5
LOG_DIR = ""
DEFAULT_THINKING = False
prompt_cache_state = None   # KV-캐시 재사용 (PromptCacheState, 턴 간 프리필 절감)
vision_cache = None         # 비전 피처 캐시 (VisionFeatureCache, 동일 이미지 재인코딩 방지)
PREFILL_STEP_SIZE = 512     # 청크 프리필 크기 (토큰) — 낮을수록 OOM 위험 감소

# 대형 프리필 경고 임계값 (토큰 수 추정치 기준, 실제 토큰화 전 문자 수 / 3.5 근사)
# 100K 토큰 이상이면 Metal OOM 위험을 콘솔에 경고한다
_LARGE_PREFILL_CHAR_THRESHOLD = 350_000  # ~100K 토큰


def _eval_kv_cache(cache_state):
    """prompt_cache_state의 KV 배열을 mx.clear_cache() 전에 evaluate해서 Metal에 고정.

    lazy 상태의 KV 배열이 clear_cache() 후 재사용될 때 command buffer 충돌을 방지한다.
    """
    if cache_state is None or cache_state.cache is None:
        return
    try:
        arrays = []
        for c in cache_state.cache:
            if hasattr(c, 'keys') and c.keys is not None:
                arrays.append(c.keys)
            if hasattr(c, 'values') and c.values is not None:
                arrays.append(c.values)
            if hasattr(c, 'state'):
                arrays.append(c.state)
        if arrays:
            mx.eval(*arrays)
    except Exception:
        pass


# ── 로깅 ─────────────────────────────────────────
def get_log_file():
    date = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"{date}.jsonl")


def log_entry(entry):
    with open(get_log_file(), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    ip = entry.get("ip", "?")
    prompt = entry.get("prompt_preview", "")[:60]
    tokens = entry.get("usage", {}).get("total_tokens", "?")
    duration = entry.get("duration_ms", "?")
    thinking = "🧠ON" if entry.get("enable_thinking") else "🧠OFF"
    stream = " 📡" if entry.get("stream") else ""
    print(f"  [{entry['timestamp']}] {ip} | {thinking}{stream} | {tokens} tokens | {duration}ms | {prompt}...")


# ── 요청 파싱 ─────────────────────────────────────
def parse_request(data: dict) -> dict:
    return {
        "model": data.get("model", model_id),
        "messages": data.get("messages", []),
        "temperature": data.get("temperature", 1.0),
        "top_p": data.get("top_p", 1.0),
        "max_tokens": data.get("max_tokens") or data.get("max_completion_tokens") or 2048,
        "stop": data.get("stop"),
        "stream": data.get("stream", False),
        "seed": data.get("seed"),
        "repetition_penalty": data.get("repetition_penalty"),
        "presence_penalty": data.get("presence_penalty", 0),
        "frequency_penalty": data.get("frequency_penalty", 0),
        "enable_thinking": data.get("enable_thinking", DEFAULT_THINKING),
        "preserve_thinking": data.get("preserve_thinking", False),
    }


def normalize_messages(messages):
    """OpenAI 포맷을 Qwen3.5 채팅 템플릿 호환으로 정규화"""
    normalized = []
    for msg in messages:
        m = {**msg}

        # content: 배열 → 문자열 (멀티모달 포맷)
        # image_url 파트는 extract_images()가 원본 messages에서 직접 읽으므로
        # 여기서 제거하지 않는다 — normalize 이전에 extract_images() 호출 필요
        content = m.get("content", "")
        if isinstance(content, list):
            parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            m["content"] = "\n".join(parts)

        # tool_calls: arguments가 JSON 문자열이면 dict로 파싱
        # (OpenAI는 문자열, Qwen3.5 템플릿은 dict를 기대)
        if "tool_calls" in m:
            tool_calls = []
            for tc in m["tool_calls"]:
                tc = {**tc}
                if "function" in tc:
                    func = {**tc["function"]}
                    args = func.get("arguments", "")
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            func["arguments"] = {}
                    tc["function"] = func
                tool_calls.append(tc)
            m["tool_calls"] = tool_calls

        normalized.append(m)
    return normalized


ALLOWED_URL_SCHEMES = ("http://", "https://")
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20MB raw bytes 상한 (압축 폭탄 방지)


def extract_images(messages):
    """OpenAI vision 포맷에서 PIL Image 리스트 추출 (CORE-02, CORE-03)"""
    images = []
    for msg in messages:
        content = msg.get("content", "")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue
            url = part.get("image_url", {}).get("url", "")
            try:
                if url.startswith("data:"):
                    # base64 data URI: "data:image/jpeg;base64,<data>"
                    header, data = url.split(",", 1)
                    decoded = base64.b64decode(data)
                    if len(decoded) > MAX_IMAGE_BYTES:
                        raise ValueError("이미지 크기 초과 (최대 20MB)")
                    img = Image.open(io.BytesIO(decoded))
                elif url.startswith(ALLOWED_URL_SCHEMES):
                    # HTTP/HTTPS URL만 허용 (SSRF 방지 — file://, ftp://, 내부 IP 등 차단)
                    import urllib.request
                    with urllib.request.urlopen(url, timeout=10) as resp:
                        raw = resp.read(MAX_IMAGE_BYTES + 1)
                        if len(raw) > MAX_IMAGE_BYTES:
                            raise ValueError("원격 이미지 크기 초과 (최대 20MB)")
                        img = Image.open(io.BytesIO(raw))
                else:
                    raise ValueError(f"허용되지 않는 URL 스킴: {url[:80]}")
                # 최대 1120px 리사이즈 — 비율 유지 (OOM 방지, CORE-03)
                img.thumbnail((1120, 1120))
                images.append(img.convert("RGB"))
            except ValueError:
                raise
            except Exception:
                raise ValueError(f"이미지 디코딩 실패: {url[:80]}")
    return images


def get_prompt_preview(messages):
    if not messages:
        return ""
    content = messages[-1].get("content", "")
    if isinstance(content, str):
        return content[:200]
    if isinstance(content, list):
        return str(content)[:200]
    return ""


def strip_thinking(text):
    """<think>...</think> 블록 제거 — preserve_thinking=False일 때만 호출 (CORE-06)"""
    import re
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _warn_large_prefill(messages):
    """메시지 전체 문자 수 기준으로 대형 프리필 위험을 콘솔에 경고한다.

    Metal GPU는 대형 프리필 도중 OOM이 발생하면 C-level abort()를 일으킨다.
    Python try/except로 잡을 수 없으므로 사전 경고만 제공한다.
    """
    total_chars = sum(
        len(m.get("content", "") if isinstance(m.get("content"), str) else str(m.get("content", "")))
        for m in messages
    )
    if total_chars >= _LARGE_PREFILL_CHAR_THRESHOLD:
        approx_tokens = total_chars // 3
        print(f"  [WARN] 대형 프리필 감지: ~{approx_tokens:,}토큰 추정 "
              f"({total_chars:,}자). Metal OOM 위험 — 컨텍스트 크기를 줄이거나 "
              f"서버 재시작 후 요청하세요.")


# ── 응답 생성 ─────────────────────────────────────
def make_completion_response(req_id, model_name, content, finish_reason, prompt_tokens, completion_tokens):
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def make_chunk(req_id, model_name, delta, finish_reason=None):
    return {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }


# ── 추론 ─────────────────────────────────────────
def run_inference(params):
    # _images가 사전 검증에서 이미 추출된 경우 재사용 (이중 fetch 방지, WR-02)
    images = params.get("_images") if "_images" in params else extract_images(params["messages"])
    messages = normalize_messages(params["messages"])

    # 대형 프리필 경고 (Metal OOM 사전 감지)
    _warn_large_prefill(messages)

    # apply_chat_template은 processor의 메서드가 아닌 독립 함수 (per D-12, RESEARCH Pattern 2)
    formatted = apply_chat_template(
        processor,
        model.config,
        messages,
        num_images=len(images),
        chat_template_kwargs={"enable_thinking": params["enable_thinking"]},
    )

    # 추론 전 Metal 동기화: inflight 커맨드 버퍼가 모두 완료될 때까지 대기.
    # clear_cache()는 여기서 호출하지 않음 — prompt_cache_state.cache의 KV 배열이
    # lazy 상태일 때 Metal 버퍼가 해제되면 다음 재사용 시 command buffer 충돌이 발생.
    # (이전 요청 finally 블록에서 이미 clear_cache() 처리됨)
    mx.synchronize()

    # mlx_vlm.generate()는 sampler 오브젝트 불필요 — temp/top_p 직접 전달 (per RESEARCH Pitfall 2)
    result = generate(
        model,
        processor,
        formatted,
        image=images if images else None,   # 이미지 없을 때 빈 리스트 아닌 None (per RESEARCH)
        max_tokens=params["max_tokens"],
        temp=params["temperature"],
        top_p=params["top_p"],
        prompt_cache_state=None if images else prompt_cache_state,
        vision_cache=vision_cache,
        prefill_step_size=PREFILL_STEP_SIZE,
    )

    full_text = result.text
    prompt_tokens = result.prompt_tokens
    completion_tokens = result.generation_tokens
    # finish_reason 방어 처리 (GenerationResult에 필드 없을 수 있음, per RESEARCH Pitfall 3)
    finish_reason = getattr(result, "finish_reason", None)
    if finish_reason is None:
        finish_reason = "length" if completion_tokens >= params["max_tokens"] else "stop"

    # preserve_thinking=False일 때만 <think> 블록 제거 (per D-19)
    if not params.get("preserve_thinking"):
        full_text = strip_thinking(full_text)

    if images:
        prompt_cache_state.cache = None
        prompt_cache_state.token_ids = None
    _eval_kv_cache(prompt_cache_state)
    mx.synchronize()
    mx.clear_cache()
    return full_text, finish_reason, prompt_tokens, completion_tokens


def run_inference_streaming(params):
    # _images가 사전 검증에서 이미 추출된 경우 재사용 (이중 fetch 방지, WR-02)
    images = params.get("_images") if "_images" in params else extract_images(params["messages"])
    messages = normalize_messages(params["messages"])

    # 대형 프리필 경고 (Metal OOM 사전 감지)
    _warn_large_prefill(messages)

    formatted = apply_chat_template(
        processor,
        model.config,
        messages,
        num_images=len(images),
        chat_template_kwargs={"enable_thinking": params["enable_thinking"]},
    )

    # 추론 전 Metal 동기화 (clear_cache는 생략 — KV 캐시 보존 필요, Change 1 주석 참조)
    mx.synchronize()

    try:
        for response in stream_generate(
            model,
            processor,
            formatted,
            image=images if images else None,
            max_tokens=params["max_tokens"],
            temp=params["temperature"],
            top_p=params["top_p"],
            prompt_cache_state=None if images else prompt_cache_state,
            vision_cache=vision_cache,
            prefill_step_size=PREFILL_STEP_SIZE,
        ):
            yield response
    finally:
        # stream_generate() 완료 후 KV 캐시 고정 → Metal 동기화 → 임시 버퍼 해제
        if images:
            prompt_cache_state.cache = None
            prompt_cache_state.token_ids = None
        _eval_kv_cache(prompt_cache_state)
        mx.synchronize()
        mx.clear_cache()


# ── 엔드포인트 ────────────────────────────────────
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global pending

    if pending >= MAX_QUEUE:
        return JSONResponse(
            status_code=429,
            content={"error": {"message": "Server busy", "type": "rate_limit_error"}},
        )

    data = await request.json()
    params = parse_request(data)
    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    ip = request.client.host if request.client else "?"
    start = time.time()
    last_msg = get_prompt_preview(params["messages"])

    # 이미지 파싱 사전 검증 (디코딩 실패 시 즉시 400 반환, per D-09)
    # 결과를 params["_images"]에 저장해 run_inference/run_inference_streaming에서 재사용 (이중 fetch 방지)
    try:
        params["_images"] = extract_images(params["messages"])
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": str(e), "type": "invalid_request_error"}},
        )

    if params["stream"]:
        # WR-03: 스트리밍에서 preserve_thinking=False는 SSE 청크에 적용되지 않음
        # (이미 yield된 청크는 수정 불가). 클라이언트에 <think> 블록이 노출될 수 있음.
        # 완전한 필터링이 필요하면 stream=false 사용 권장.
        if not params.get("preserve_thinking") and params.get("enable_thinking"):
            print(f"  [WARN] {req_id}: stream=true + preserve_thinking=false — "
                  "<think> 블록이 SSE 청크에 포함됩니다. 비스트리밍 사용 권장.")
        return _stream_response(req_id, params, ip, start, last_msg)

    # 비스트리밍
    pending += 1
    try:
        async with gpu_semaphore:
            loop = asyncio.get_running_loop()
            text, finish_reason, p_tokens, c_tokens = await loop.run_in_executor(
                None, partial(run_inference, params)
            )
    finally:
        pending -= 1

    duration_ms = int((time.time() - start) * 1000)
    resp = make_completion_response(req_id, params["model"], text, finish_reason, p_tokens, c_tokens)

    log_entry({
        "timestamp": datetime.now().isoformat(),
        "ip": ip,
        "enable_thinking": params["enable_thinking"],
        "stream": False,
        "duration_ms": duration_ms,
        "prompt_preview": last_msg,
        "usage": resp["usage"],
        "finish_reason": finish_reason,
        "content_preview": text[:200],
    })

    return resp


# ── 스트리밍 ──────────────────────────────────────
_SENTINEL = object()


def _stream_response(req_id, params, ip, start, last_msg):
    async def event_generator():
        global pending
        pending += 1
        prompt_tokens = 0
        completion_tokens = 0
        finish_reason = "stop"
        full_text = ""

        try:
            async with gpu_semaphore:
                # 첫 청크: role
                yield f"data: {json.dumps(make_chunk(req_id, params['model'], {'role': 'assistant', 'content': ''}))}\n\n"

                # 동기 제너레이터 → async Queue 브릿지 (진짜 토큰별 스트리밍)
                queue = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def _produce():
                    # 성공: _SENTINEL으로 종료, 실패: 예외 객체를 큐에 전달 (상호 배타적 터미네이터)
                    try:
                        for response in run_inference_streaming(params):
                            loop.call_soon_threadsafe(queue.put_nowait, response)
                        loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)
                    except Exception as e:
                        # 예외를 삼키지 않고 소비 측에 전파 (WR-04)
                        loop.call_soon_threadsafe(queue.put_nowait, e)

                loop.run_in_executor(None, _produce)

                while True:
                    item = await queue.get()
                    if isinstance(item, Exception):
                        # _produce에서 전파된 추론 오류 — 클라이언트에 500 반환
                        raise item
                    if item is _SENTINEL:
                        break

                    full_text += item.text
                    prompt_tokens = item.prompt_tokens
                    completion_tokens = item.generation_tokens

                    # finish_reason 방어 처리 (GenerationResult 필드 미보장, per RESEARCH Pitfall 3)
                    item_finish = getattr(item, "finish_reason", None)
                    if item_finish == "length" or completion_tokens >= params["max_tokens"]:
                        finish_reason = "length"

                    yield f"data: {json.dumps(make_chunk(req_id, params['model'], {'content': item.text}))}\n\n"

                # preserve_thinking=False일 때 full_text에서 <think> 블록 제거 (per D-19)
                # 주의: SSE 청크는 이미 yield됐으므로 content_preview 로그에만 영향 (알려진 제한)
                if not params.get("preserve_thinking"):
                    full_text = strip_thinking(full_text)

                # 최종 청크
                yield f"data: {json.dumps(make_chunk(req_id, params['model'], {}, finish_reason))}\n\n"
                yield "data: [DONE]\n\n"

        finally:
            pending -= 1

            duration_ms = int((time.time() - start) * 1000)
            log_entry({
                "timestamp": datetime.now().isoformat(),
                "ip": ip,
                "enable_thinking": params["enable_thinking"],
                "stream": True,
                "duration_ms": duration_ms,
                "prompt_preview": last_msg,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "finish_reason": finish_reason,
                "content_preview": full_text[:200],
            })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── 메인 ─────────────────────────────────────────
def main():
    global model, processor, model_id, gpu_semaphore, MAX_QUEUE, LOG_DIR, DEFAULT_THINKING, \
           prompt_cache_state, vision_cache

    parser = argparse.ArgumentParser(description="Local LLM API Server")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3.6-27B-6bit")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-queue", type=int, default=5)
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=False,
                        help="Thinking ON/OFF (기본값: OFF, --think으로 활성화)")
    args = parser.parse_args()

    model_id = args.model
    MAX_QUEUE = args.max_queue
    DEFAULT_THINKING = args.think
    gpu_semaphore = asyncio.Semaphore(1)
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"📥 모델 로딩: {model_id}")
    model, processor = load(model_id)
    prompt_cache_state = PromptCacheState()
    vision_cache = VisionFeatureCache()
    print(f"✅ 모델 로드 완료")
    print()
    print(f"🌐 API 서버: http://{args.host}:{args.port}")
    print(f"   엔드포인트: /v1/chat/completions")
    print(f"   스트리밍: stream=true 지원")
    if DEFAULT_THINKING:
        print(f"   🧠 Thinking: ON (기본, 요청별 override 가능)")
    else:
        print(f"   🧠 Thinking: OFF (기본, 요청별 override 가능)")
    print(f"   📝 로깅: {LOG_DIR}/")
    print(f"   종료: Ctrl+C")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
