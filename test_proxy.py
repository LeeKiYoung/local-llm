"""
llm-proxy.py 테스트
실행: .venv/bin/python test_proxy.py
"""

import json
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError

PASS = 0
FAIL = 0


def test(name, condition):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}")


# ============================================================
# 1. 모킹용 백엔드 서버 (mlx_lm.server 대체)
# ============================================================

class MockBackendHandler(BaseHTTPRequestHandler):
    """다양한 응답 시나리오를 시뮬레이션"""

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))

        # 요청 내용에 따라 다른 응답
        messages = body.get("messages", [])
        last_msg = messages[-1]["content"] if messages else ""

        if "error" in last_msg.lower():
            # 에러 응답 시뮬레이션
            resp = {"error": "simulated error"}
            self.send_response(500)
        elif "empty" in last_msg.lower():
            # choices가 빈 응답
            resp = {
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
            }
            self.send_response(200)
        elif "no_choices" in last_msg.lower():
            # choices 키 자체가 없는 응답
            resp = {"usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5}}
            self.send_response(200)
        elif "malformed" in last_msg.lower():
            # JSON이 아닌 응답
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"not json")
            return
        else:
            # 정상 응답
            resp = {
                "id": "test-123",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": f"테스트 응답: {last_msg}",
                        "reasoning": "thinking about it...",
                    },
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            }
            self.send_response(200)

        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp).encode())

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"models": ["test-model"]}).encode())

    def log_message(self, format, *args):
        pass  # 로그 억제


def start_server(handler, port):
    server = HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def send_request(port, content="안녕!", extra_fields=None):
    """프록시에 요청 전송"""
    data = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 200,
    }
    if extra_fields:
        data.update(extra_fields)

    req = Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req) as resp:
            body = resp.read()
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, {"raw": body.decode(errors="replace")}
    except URLError as e:
        if hasattr(e, "code"):
            return e.code, {}
        return 0, {"error": str(e)}


# ============================================================
# 2. 테스트 실행
# ============================================================

if __name__ == "__main__":
    BACKEND_PORT = 19081
    PROXY_PORT = 19080
    LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "test")

    # 환경변수 설정
    os.environ["BACKEND_PORT"] = str(BACKEND_PORT)
    os.environ["PROXY_PORT"] = str(PROXY_PORT)

    # 로그 디렉토리 정리
    os.makedirs(LOG_DIR, exist_ok=True)

    # llm-proxy의 LOG_DIR을 테스트용으로 변경
    import importlib.util
    spec = importlib.util.spec_from_file_location("llm_proxy", os.path.join(os.path.dirname(__file__), "llm-proxy.py"))
    proxy_module = importlib.util.module_from_spec(spec)

    # LOG_DIR 패치
    import types
    original_log_dir = None

    # 서버 시작
    print("\n🧪 프록시 테스트 시작\n")

    print("  서버 시작 중...")
    mock_backend = start_server(MockBackendHandler, BACKEND_PORT)
    time.sleep(0.5)

    # 프록시 시작 (llm-proxy.py를 모듈로 import하지 않고 서브프로세스로)
    import subprocess
    env = os.environ.copy()
    env["BACKEND_PORT"] = str(BACKEND_PORT)
    env["PROXY_PORT"] = str(PROXY_PORT)
    proxy_proc = subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "llm-proxy.py")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1)
    print("  서버 준비 완료\n")

    # --- 테스트 케이스 ---

    print("📋 1. 정상 요청/응답")
    status, resp = send_request(PROXY_PORT, "안녕하세요!")
    test("상태 코드 200", status == 200)
    test("choices 존재", "choices" in resp)
    test("content 포함", "테스트 응답" in resp.get("choices", [{}])[0].get("message", {}).get("content", ""))
    test("usage 포함", "usage" in resp)

    print("\n📋 2. choices가 빈 응답")
    status, resp = send_request(PROXY_PORT, "empty response")
    test("상태 코드 200", status == 200)
    test("빈 choices", resp.get("choices") == [])

    print("\n📋 3. choices 키 없는 응답")
    status, resp = send_request(PROXY_PORT, "no_choices response")
    test("상태 코드 200", status == 200)
    test("choices 키 없음", "choices" not in resp)

    print("\n📋 4. malformed 응답 (JSON 아닌 응답)")
    status, resp = send_request(PROXY_PORT, "malformed response")
    test("상태 코드 200", status == 200)

    print("\n📋 5. enable_thinking 파라미터 전달")
    status, resp = send_request(PROXY_PORT, "thinking test", {"enable_thinking": False})
    test("상태 코드 200", status == 200)
    test("정상 응답", "choices" in resp)

    print("\n📋 6. 긴 프롬프트 (200자 초과)")
    long_prompt = "가" * 500
    status, resp = send_request(PROXY_PORT, long_prompt)
    test("상태 코드 200", status == 200)
    test("정상 응답", "choices" in resp)

    print("\n📋 7. GET 요청 (models 엔드포인트)")
    try:
        req = Request(f"http://127.0.0.1:{PROXY_PORT}/v1/models", method="GET")
        with urlopen(req) as r:
            get_status = r.status
            get_resp = json.loads(r.read())
        test("GET 상태 코드 200", get_status == 200)
        test("models 응답", "models" in get_resp)
    except Exception as e:
        test(f"GET 요청 실패: {e}", False)

    print("\n📋 8. 로그 파일 생성 확인")
    from datetime import datetime
    log_file = os.path.join("logs", f"{datetime.now().strftime('%Y-%m-%d')}.jsonl")
    test("로그 파일 존재", os.path.exists(log_file))

    if os.path.exists(log_file):
        with open(log_file) as f:
            lines = f.readlines()
        test("로그 항목 존재", len(lines) > 0)

        if lines:
            entry = json.loads(lines[-1])
            test("timestamp 포함", "timestamp" in entry)
            test("ip 포함", "ip" in entry)
            test("prompt_preview 포함", "prompt_preview" in entry)
            test("duration_ms 포함", "duration_ms" in entry)

    # --- 정리 ---
    print("\n" + "=" * 40)
    print(f"  결과: ✅ {PASS} passed, ❌ {FAIL} failed")
    print("=" * 40)

    proxy_proc.terminate()
    mock_backend.shutdown()

    sys.exit(0 if FAIL == 0 else 1)
