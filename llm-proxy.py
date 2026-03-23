"""
LLM 로깅 프록시 서버
클라이언트 → 프록시(8080) → mlx_lm.server(8081)
요청/응답을 logs/ 폴더에 기록
"""

import json
import os
import sys
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError

BACKEND_PORT = int(os.environ.get("BACKEND_PORT", 8081))
PROXY_PORT = int(os.environ.get("PROXY_PORT", 8080))
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

os.makedirs(LOG_DIR, exist_ok=True)


def get_log_file():
    """일별 로그 파일"""
    date = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(LOG_DIR, f"{date}.jsonl")


def log_entry(entry):
    """로그 항목을 JSONL 형식으로 저장"""
    with open(get_log_file(), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 콘솔에도 간략히 출력
    ip = entry.get("ip", "?")
    model = entry.get("request", {}).get("model", "default")
    prompt = entry.get("prompt_preview", "")[:60]
    tokens = entry.get("response", {}).get("usage", {}).get("total_tokens", "?")
    duration = entry.get("duration_ms", "?")
    print(f"  [{entry['timestamp']}] {ip} | {tokens} tokens | {duration}ms | {prompt}...")


class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        start = time.time()

        # 요청 읽기
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # 프롬프트 파싱
        try:
            req_data = json.loads(body)
            messages = req_data.get("messages", [])
            last_msg = messages[-1]["content"] if messages else ""
        except Exception:
            req_data = {}
            last_msg = ""

        # 백엔드로 전달
        backend_url = f"http://localhost:{BACKEND_PORT}{self.path}"
        req = Request(
            backend_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(req) as resp:
                resp_body = resp.read()
                status = resp.status
        except URLError as e:
            error_msg = str(e)
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": error_msg}).encode())

            log_entry({
                "timestamp": datetime.now().isoformat(),
                "ip": self.client_address[0],
                "path": self.path,
                "prompt_preview": last_msg[:200],
                "error": error_msg,
            })
            return

        duration_ms = int((time.time() - start) * 1000)

        # 응답 파싱
        try:
            resp_data = json.loads(resp_body)
            content = ""
            choices = resp_data.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                reasoning = msg.get("reasoning", "")
        except Exception:
            resp_data = {}
            choices = []
            content = ""
            reasoning = ""

        # 로그 기록
        log_entry({
            "timestamp": datetime.now().isoformat(),
            "ip": self.client_address[0],
            "path": self.path,
            "duration_ms": duration_ms,
            "request": {
                "model": req_data.get("model", "default"),
                "messages": messages,
                "max_tokens": req_data.get("max_tokens"),
                "temperature": req_data.get("temperature"),
                "enable_thinking": req_data.get("enable_thinking"),
            },
            "prompt_preview": last_msg[:200],
            "response": {
                "content_preview": content[:200] if content else "",
                "reasoning_length": len(reasoning) if reasoning else 0,
                "usage": resp_data.get("usage", {}),
                "finish_reason": choices[0].get("finish_reason") if choices else None,
            },
        })

        # 클라이언트에 응답 전달
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp_body)

    def do_GET(self):
        # GET 요청은 그대로 전달 (models 엔드포인트 등)
        backend_url = f"http://localhost:{BACKEND_PORT}{self.path}"
        try:
            req = Request(backend_url, method="GET")
            with urlopen(req) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(resp_body)
        except URLError:
            self.send_response(502)
            self.end_headers()

    def log_message(self, format, *args):
        # 기본 HTTP 로그 억제 (커스텀 로그 사용)
        pass


if __name__ == "__main__":
    print(f"📝 로깅 프록시: http://0.0.0.0:{PROXY_PORT} → http://localhost:{BACKEND_PORT}")
    print(f"📂 로그 저장: {LOG_DIR}/")
    print()
    server = HTTPServer(("0.0.0.0", PROXY_PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n프록시 종료")
