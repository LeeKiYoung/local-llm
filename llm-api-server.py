"""
커스텀 LLM API 서버
FastAPI + mlx_lm Python API 직접 호출

특징:
- OpenAI API 호환 (/v1/chat/completions, /v1/models)
- 요청별 enable_thinking 제어
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
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

# ── 글로벌 ────────────────────────────────────────
app = FastAPI()
model = None
tokenizer = None
model_id = ""
gpu_semaphore = None
pending = 0
MAX_QUEUE = 5
LOG_DIR = ""
DEFAULT_THINKING = False


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
    }


def get_prompt_preview(messages):
    if not messages:
        return ""
    content = messages[-1].get("content", "")
    if isinstance(content, str):
        return content[:200]
    if isinstance(content, list):
        return str(content)[:200]
    return ""


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
    prompt = tokenizer.apply_chat_template(
        params["messages"],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=params["enable_thinking"],
    )

    sampler = make_sampler(
        temp=params["temperature"],
        top_p=params["top_p"],
    )

    full_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    finish_reason = "stop"

    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=params["max_tokens"],
        sampler=sampler,
    ):
        full_text += response.text
        prompt_tokens = response.prompt_tokens
        completion_tokens = response.generation_tokens

        if response.finish_reason == "length":
            finish_reason = "length"

    mx.metal.clear_cache()
    return full_text, finish_reason, prompt_tokens, completion_tokens


def run_inference_streaming(params):
    prompt = tokenizer.apply_chat_template(
        params["messages"],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=params["enable_thinking"],
    )

    sampler = make_sampler(
        temp=params["temperature"],
        top_p=params["top_p"],
    )

    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=params["max_tokens"],
        sampler=sampler,
    ):
        yield response

    mx.metal.clear_cache()


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

    if params["stream"]:
        return _stream_response(req_id, params, ip, start, last_msg)

    # 비스트리밍
    pending += 1
    try:
        async with gpu_semaphore:
            loop = asyncio.get_event_loop()
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

                loop = asyncio.get_event_loop()
                gen = partial(run_inference_streaming, params)

                for response in await loop.run_in_executor(None, lambda: list(gen())):
                    full_text += response.text
                    prompt_tokens = response.prompt_tokens
                    completion_tokens = response.generation_tokens

                    if response.finish_reason == "length":
                        finish_reason = "length"

                    yield f"data: {json.dumps(make_chunk(req_id, params['model'], {'content': response.text}))}\n\n"

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
    global model, tokenizer, model_id, gpu_semaphore, MAX_QUEUE, LOG_DIR, DEFAULT_THINKING

    parser = argparse.ArgumentParser(description="Local LLM API Server")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3.5-35B-A3B-4bit")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-queue", type=int, default=5)
    parser.add_argument("--think", action="store_true", help="기본 Thinking ON (요청별 override 가능)")
    args = parser.parse_args()

    model_id = args.model
    MAX_QUEUE = args.max_queue
    DEFAULT_THINKING = args.think
    gpu_semaphore = asyncio.Semaphore(1)
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"📥 모델 로딩: {model_id}")
    model, tokenizer = load(model_id)
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
