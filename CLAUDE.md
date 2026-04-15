<!-- GSD:project-start source:PROJECT.md -->
## Project

**local-llm**

Apple Silicon Mac에서 MLX 기반 로컬 LLM을 OpenAI 호환 API 서버로 실행하는 프로젝트. Qwen3.5-35B(텍스트 특화)와 SuperGemma4-26B(멀티모달)를 서버 시작 시 선택해 실행하며, openclaw 및 OpenAI SDK 호환 클라이언트에 로컬 추론 API를 제공한다.

**Core Value:** openclaw 등 OpenAI SDK 호환 클라이언트가 이미지를 포함한 메시지를 보내면 로컬에서 완전히 처리되어 응답이 돌아온다.

### Constraints

- **Tech stack**: Python 3.11 + MLX 0.31+ + FastAPI — 기존 스택 유지, mlx-vlm 추가
- **Memory**: M5 Pro 64GB, 모델 1개만 로드 (15~20GB 점유)
- **API 호환성**: OpenAI SDK 호환 포맷 유지 필수 (클라이언트 변경 없이 동작해야 함)
- **플랫폼**: Apple Silicon 전용 (mlx, mlx-vlm 모두 Apple Silicon 필수)
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.11 - All application logic, API server, inference integration
- Shell (Bash) - Setup automation and server launch scripts
## Runtime
- Python 3.11.14 (Apple Silicon specific - requires 3.10+)
- Virtual environment: `.venv/` (isolated dependency management)
- pip - Python package management
- Lockfile: No explicit lock file (setup.sh pins mlx-lm and fastapi versions)
## Frameworks
- **MLX** 0.31.1 - Apple Silicon native ML framework (GPU acceleration via Metal)
- **FastAPI** 0.135.2 - Async HTTP API server for OpenAI-compatible endpoints
- **Uvicorn** 0.42.0 - ASGI server runner for FastAPI
- **pytest** 9.0.2 - Unit testing framework
- **NumPy** 2.4.3 - Numerical computing (MLX dependency)
- **transformers** 5.0.0+ - Model loading and tokenization (mlx-lm dependency)
- **sentencepiece** - Tokenization library (mlx-lm dependency)
## Key Dependencies
- mlx-lm 0.31.1 - Provides load(), generate(), and stream_generate() functions for model inference
- mlx 0.31.1 - Core computation engine, mx.clear_cache() for KV cache management
- fastapi 0.135.2 - HTTP API framework for /v1/chat/completions and /v1/models endpoints
- transformers 5.0.0+ - Tokenizer.apply_chat_template() for prompt formatting with enable_thinking support
- pyyaml - Config parsing for model definitions
- jinja2 - Chat template rendering engine
- protobuf - Message serialization (transformers dependency)
- numpy 2.4.3 - Array operations (MLX dependency)
## Configuration
- `HF_HOME` - Hugging Face cache directory (default: `~/.cache/huggingface`)
- `profiles/config-262k.json` - Default Qwen3.5 config with 262K token context window
- `profiles/config-1m.json` - Extended context (1M tokens) with YaRN rope scaling
- `setup.sh` - Automated environment setup (Python check, venv creation, pip install)
- `llm-server.sh` - API server launcher with profile switching and thinking mode control
- `llm-chat.sh` - Interactive CLI chat launcher with context profile support
## Platform Requirements
- **Architecture:** Apple Silicon only (arm64 - M1/M2/M3/M4/M5)
- **Memory:** Minimum 24GB RAM (16GB for lighter models like Qwen3.5-9B)
- **Disk:** 20GB+ free (model caching at ~/.cache/huggingface/hub/)
- **Python:** 3.10+ (tested with 3.11.14)
- **macOS:** Native support for sleep prevention via caffeinate command
- Single Apple Silicon Mac (M-series)
- Network accessible via localhost or local network IP
- Tailscale integration supported for remote access (no public internet exposure)
## Logs & Artifacts
- Location: `logs/` directory (auto-created)
- Format: JSONL (one JSON object per line, date-partitioned)
- Contents: Request IP, thinking mode, token counts, duration, content preview
- Filename: `YYYY-MM-DD.jsonl` (new file per calendar day)
- Location: `~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-35B-A3B-4bit/`
- Size: ~19-20GB (quantized to 4-bit)
- Auto-downloaded on first run via mlx_lm
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- Kebab-case for executable Python scripts: `llm-api-server.py`, `llm-proxy.py`, `llm-server.sh`, `llm-chat.sh`
- Snake_case for test files: `test_api_server.py`, `test_proxy.py`
- Avoid underscores in filenames for production code; reserve for tests
- Snake_case for all functions: `parse_request()`, `normalize_messages()`, `strip_thinking()`, `get_log_file()`, `log_entry()`
- Descriptive verbs: `make_completion_response()`, `make_chunk()`, `run_inference()`, `run_inference_streaming()`
- Private/internal functions prefixed with underscore: `_stream_response()`, `_produce()`
- Snake_case for all variables: `model`, `tokenizer`, `model_id`, `gpu_semaphore`, `pending`, `req_id`, `req_data`, `resp_data`
- All-caps for module-level constants: `MAX_QUEUE`, `LOG_DIR`, `DEFAULT_THINKING`, `BACKEND_PORT`, `PROXY_PORT`, `_SENTINEL`
- English names (no Korean) for variables, except in docstrings and comments
- No explicit type hints in most function signatures; only on critical functions: `parse_request(data: dict) -> dict`
- Dataclasses for structured mock data: `@dataclass class MockResponse`
- Dictionary literals for configuration/response objects
## Code Style
- 4-space indentation (Python standard)
- No explicit formatter configured (no .black, .flake8, or .isort files)
- Max line length: flexible, up to ~100 characters observed
- String quotes: double quotes for standard strings, f-strings for interpolation
- No ESLint/Flake8/Pylint configuration detected
- Code appears hand-formatted without automated checks
- Imports organized manually (no sort automation)
## Import Organization
- No path aliases detected; all imports are absolute or from standard library
- Projects use direct module imports without '@' aliases or path configuration
- Mix of `import module` and `from module import specific` - both patterns used
- No `import *` wildcard imports
- Conditional imports for mocking: `sys.modules["mlx"] = MagicMock()` pattern in tests
## Error Handling
- Silent exception handling common: `except Exception: pass` in data parsing (llm-proxy.py:154)
- JSON decode errors caught specifically: `except (json.JSONDecodeError, TypeError):` (llm-api-server.py:101)
- URLError caught for network errors: `except URLError as e:` (llm-proxy.py:117)
- Finally blocks for cleanup: `finally: pending -= 1` for queue management (llm-api-server.py:262)
- No custom exception classes; relies on built-in exceptions
## Logging
- Custom `log_entry()` function writes structured JSON to JSONL files: `logs/{YYYY-MM-DD}.jsonl`
- Console output mixed with file logging - both happen simultaneously
- Log entries contain: timestamp (ISO format), ip, duration_ms, prompt_preview, usage stats, finish_reason
- Emoji indicators in console output: 🧠ON/OFF for thinking, 📡 for streaming, ✅ for success, ❌ for failure
- Per-request logging on every API call; detailed metrics tracked
## Comments
- Section dividers using ASCII comment blocks (Korean labels):
- Inline comments explain complex logic: `# mlx.core를 mock으로 대체`
- Comments in Korean (혼합 사용) within code, but docstrings can mix Korean/English
- Python docstrings used sparingly, only on key functions
- Single-line docstring pattern: `"""Function description"""`
- Example: `"""OpenAI 포맷을 Qwen3.5 채팅 템플릿 호환으로 정규화"""`
- Module-level docstrings at file start describe overall purpose:
## Function Design
- `parse_request()`: 14 lines (configuration extraction)
- `normalize_messages()`: 30 lines (message transformation)
- `run_inference()`: 34 lines (core inference logic)
- No particularly large functions; modular approach preferred
- Functions accept dictionaries as main parameter: `def parse_request(data: dict)`
- Callback/handler functions accept standard HTTP request objects: `async def chat_completions(request: Request)`
- Prefer config dicts over multiple arguments for flexibility
- Explicit return types when important: `-> dict`
- Dictionary returns for structured data: `make_completion_response()`, `parse_request()`
- Generator returns for streaming: `run_inference_streaming()` yields responses
- JSON responses wrapped in FastAPI response classes: `JSONResponse()`, `StreamingResponse()`
## Module Design
- No explicit `__all__` declarations; top-level functions directly callable
- Global variables shared across module: `app`, `model`, `tokenizer`, `model_id`, `gpu_semaphore`, `pending`
- Module-as-singleton pattern for FastAPI `app` instance
- Constants and globals at module top (lines 28-37 in llm-api-server.py)
- Helper functions grouped by purpose: logging, parsing, response building, inference
- Async endpoint handlers after helpers
- Main entry point and argument parsing at bottom (`if __name__ == "__main__"`)
## Configuration & Constants
- Command-line arguments via `argparse`: `--model`, `--host`, `--port`, `--max-queue`, `--think`
- Environment variables: `BACKEND_PORT`, `PROXY_PORT` (with defaults)
- Hard-coded defaults: `MAX_QUEUE = 5`, `DEFAULT_THINKING = False`
## String Handling
## Dependency Injection & Globals
- Global `model`, `tokenizer`, `gpu_semaphore` initialized at startup
- Modified at runtime by main() function
- Accessed directly by endpoint handlers
- Trade-off: Simple single-model setup but limits testing/concurrency
- Globals modified via test fixtures: `server_module.model = MagicMock()`
- No dependency injection framework; direct assignment in setup/teardown
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Single-model inference server with semaphore-based request queuing
- Streaming support via asyncio Queue bridge from sync MLX generators
- OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints
- Per-request control of model features (thinking, sampling parameters)
- Automatic GPU memory management (KV cache clearing after each request)
## Layers
- Purpose: HTTP request handling and OpenAI API compatibility
- Location: `llm-api-server.py` lines 221-354 (endpoints)
- Contains: FastAPI route handlers, request/response formatting
- Depends on: Inference layer, logging layer
- Used by: External clients via HTTP
- Purpose: Model loading, prompt formatting, token generation
- Location: `llm-api-server.py` lines 156-217 (`run_inference`, `run_inference_streaming`)
- Contains: MLX model/tokenizer integration, sampler configuration, streaming wrapper
- Depends on: MLX core library (`mlx.core`, `mlx_lm`)
- Used by: API layer for generating responses
- Purpose: Normalize OpenAI format to model template format
- Location: `llm-api-server.py` lines 60-119 (`parse_request`, `normalize_messages`, `get_prompt_preview`)
- Contains: Parameter parsing, message normalization (multimodal content, tool_calls), content conversion
- Depends on: Nothing (pure data transformation)
- Used by: Inference layer
- Purpose: JSONL event logging to daily-rotated files
- Location: `llm-api-server.py` lines 41-56 (`get_log_file`, `log_entry`)
- Contains: Daily log file management, structured event recording
- Depends on: Nothing (filesystem only)
- Used by: All request handlers
- Purpose: Transparent logging proxy with thinking response filtering
- Location: `llm-proxy.py` lines 79-183 (`ProxyHandler`)
- Contains: HTTP proxying, thinking tag stripping, request/response logging
- Depends on: Backend API server
- Used by: External clients (alternative to direct API access)
## Data Flow
- Global model/tokenizer loaded once at startup via `mlx_lm.load()`
- Per-request state (accumulated text, token counts) maintained in handler scope
- Streaming: state accumulated in `event_generator` closure across queue iterations
- GPU concurrency: `gpu_semaphore` (Semaphore(1)) ensures single active inference
- Request queue tracking: `pending` counter, `MAX_QUEUE` threshold for rate limiting
## Key Abstractions
- Purpose: Standardize response structure for OpenAI client compatibility
- Examples: `make_completion_response()` (line 123), `make_chunk()` (line 142)
- Pattern: Dictionary factories that build compliant response objects with id, created timestamp, choices array, usage stats
- Purpose: Convert OpenAI format (with multimodal arrays and JSON string tool_calls) to MLX chat template format
- Examples: `normalize_messages()` (line 77)
- Pattern: Message-by-message conversion with type checking and JSON parsing
- Purpose: Decouple sampling parameters from inference
- Examples: `make_sampler()` from mlx_lm.sample_utils (line 166, 203)
- Pattern: Factory function accepting temperature, top_p; returns sampler object passed to stream_generate
## Entry Points
- Location: `llm-api-server.py` line 29 (global `app`)
- Triggers: uvicorn.run() at startup (line 391)
- Responsibilities: Request routing, async context management
- Location: `llm-api-server.py` lines 358-395 (`main()`)
- Triggers: Direct execution (`python llm-api-server.py`)
- Responsibilities:
- Location: `llm-server.sh` (execution layer)
- Triggers: `./llm-server.sh [options]`
- Responsibilities:
## Error Handling
- Rate limiting: `if pending >= MAX_QUEUE` → return 429 JSONResponse
- Backend unavailable (proxy): URLError catch → return 502 with error message
- JSON parsing: try/except with fallback empty dict
- Missing fields: use `.get()` with defaults (e.g., `messages[]`, `max_tokens`)
- Inference interruption: `finally` blocks ensure `mx.clear_cache()` always runs
## Cross-Cutting Concerns
- JSONL-based, daily-rotated files in `logs/` directory
- Every request logs: timestamp, client IP, streaming flag, thinking flag, duration, token counts, prompt preview, content preview
- Proxy version adds request/response body details
- OpenAI format compliance checked via `.get()` with sensible defaults
- No strict schema validation (permissive parsing)
- Message content can be string or list (multimodal)
- None implemented (runs on localhost by default, exposed via llm-server.sh)
- Proxy can be placed in front for auth/rate limiting if needed
- `enable_thinking`: Per-request boolean (default from `--think` CLI flag)
- Sampling: `temperature`, `top_p` per request (defaults 1.0, 1.0)
- Max tokens: Per request (default 2048) or via `max_completion_tokens` alias
- Stop sequences: Per request (passed to sampler)
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
