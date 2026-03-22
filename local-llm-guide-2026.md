# 로컬 LLM 가이드 2026 — M5 Pro 64GB

> 작성일: 2026-03-22
> 환경: Apple M5 Pro / 64GB Unified Memory / GPU 20코어 / macOS

## 시스템 현황

| 항목 | 값 |
|------|-----|
| 총 메모리 | 64GB |
| 기본 사용량 | ~21GB |
| 여유 메모리 | ~42GB |
| 디스크 여유 | ~713GB |
| 메모리 대역폭 | ~273GB/s |

## 코딩 특화 모델

| 모델 | 구조 | 메모리(Q4) | 특징 |
|-----|------|--------:|------|
| **Qwen3-Coder-Next 80B-A3B** | MoE, 3B 활성 | ~15GB | 코딩 에이전트 최강, 256K 컨텍스트 |
| **Qwen3-Coder 30B-A3B** | MoE, 3.3B 활성 | ~18GB | 실무 코딩용, SWE-Bench 강자 |
| **Qwen 2.5 Coder 14B** | Dense | ~9GB | HumanEval 85%, 가볍고 빠름 |
| **DeepSeek V3.2 Speciale** | MoE | ~40GB | LiveCodeBench 90%, 빠듯하지만 가능 |

## 범용 모델 (한국어 + 코딩 + 추론)

| 모델 | 구조 | 메모리(Q4) | 특징 |
|-----|------|--------:|------|
| **Qwen3.5-35B-A3B** | MoE, 3B 활성 | ~20GB | 비전 포함, 한국어 강함 |
| **Qwen3.5-122B-A10B** | MoE, 10B 활성 | ~40GB | 대형급 성능, 도전 가능 |
| **Qwen3.5-27B** | Dense | ~17GB | 안정적 올라운더 |
| **Llama 3.3 70B** | Dense | ~35GB | GPT-3.5급, 영어 최강 |
| **GPT-OSS 120B** | Dense | ~60GB | OpenAI 첫 오픈소스, 64GB에선 빠듯 |
| **Mistral Large 3 (41B-A)** | MoE | ~25GB | Apache 2.0, 균형잡힌 성능 |

## 가볍고 빠른 모델

| 모델 | 메모리(Q4) | 특징 |
|-----|--------:|------|
| **Qwen3 7B** | ~5GB | 소형 최강, HumanEval 76 |
| **Phi-4 Mini** | ~3GB | MS, 수학/추론 특화 |
| **Llama 3.3 8B** | ~5GB | 균형잡힌 소형 |
| **Qwen3.5-9B** | ~6GB | 온디바이스 최적화 |

## 추천 TOP 3

| 순위 | 모델 | 이유 |
|:---:|------|------|
| 1 | **Qwen3-Coder-Next 80B-A3B** | 코딩용 최강. 80B인데 3B만 활성이라 빠르고 가벼움 (~15GB) |
| 2 | **Qwen3.5-35B-A3B** | 한국어 + 코딩 + 비전 올라운더. ~20GB로 여유 |
| 3 | **Llama 3.3 70B Q4** | ~35GB로 대형 모델 풀 가동. 42GB 여유면 충분 |

## 설치 방법

### Ollama (가장 쉬움)

```bash
# 설치
brew install ollama

# 모델 실행
ollama run qwen3-coder-next
ollama run qwen3.5:35b-a3b
ollama run llama3.3:70b
```

### LM Studio (GUI + MLX 백엔드, 속도 20~30% 빠름)

- https://lmstudio.ai 에서 다운로드
- MLX 백엔드 선택 시 Apple Silicon 최적화

## Flash-MoE: SSD 스트리밍으로 초대형 모델 실행

> 출처: [Simon Willison 블로그](https://simonwillison.net/2026/Mar/18/llm-in-a-flash/) / [flash-moe GitHub](https://github.com/danveloper/flash-moe)

### 개요

Dan Woods가 **Qwen3.5-397B** (209GB, 4-bit)를 M3 Max 48GB에서 **4.36 tok/s**로 실행 성공.
모델을 RAM에 전부 올리지 않고, **SSD에서 필요한 전문가만 스트리밍**하는 방식.

### 활용 기술

| 기술 | 설명 |
|------|------|
| **Apple "LLM in a Flash" 논문** | SSD(플래시)에 모델 저장 → 필요시 DRAM으로 로드. 데이터 전송량 최소화 |
| **Karpathy autoresearch 패턴** | Claude Code(Opus 4.6)에 논문 입력 → 90개 실험 자동 설계/실행 |
| **MoE 구조 활용** | 397B 중 토큰당 17B(4개 전문가)만 활성 → 전문가당 6.75MB만 SSD에서 로드 |
| **순수 C/Metal 구현** | Python 없이 Objective-C + Metal 셰이더로 직접 구현 |
| **OS 페이지 캐시** | macOS가 자주 쓰는 전문가를 자동 캐싱 (71% 히트율) |

### 성능 결과 (M3 Max 48GB)

| 설정 | 속도 | 비고 |
|------|-----:|------|
| 4-bit 전문가 (FMA 커널) | 4.36 tok/s | 최고 품질, 도구 호출 지원 |
| 4-bit 전문가 (기본) | 3.90 tok/s | FMA 최적화 이전 |
| 2-bit 전문가 | 5.74 tok/s | JSON 형식 손상, 도구 호출 불가 |
| 2-bit 단일 토큰 | 7.05 tok/s | 피크 성능 (실용성 제한) |

### M5 Pro 64GB 적용 예상

| 비교 항목 | M3 Max 48GB | M5 Pro 64GB |
|----------|------------|-------------|
| 메모리 | 48GB | 64GB (더 많음) |
| 메모리 대역폭 | ~400GB/s | ~273GB/s (낮음) |
| GPU 코어 | 40코어 | 20코어 |
| SSD 속도 | ~17.5GB/s | 비슷 |
| 예상 속도 | 4.36 tok/s | ~3-4 tok/s (GPU 코어 차이) |
| 장점 | GPU 파워 | 메모리 여유 → 캐시 히트율 향상 가능 |

### 주요 발견 (58개 실험)

- LZ4 압축, prefetch, 예측적 라우팅 등이 **오히려 성능 저하**
- Apple Silicon에서는 단순한 직렬 파이프라인(GPU → SSD → GPU)이 최적
- "운영체제를 신뢰하라" — 커스텀 캐시보다 OS 페이지 캐시가 더 효율적

### 설치 및 실행

```bash
git clone https://github.com/danveloper/flash-moe
cd flash-moe/metal_infer
make
./infer --prompt "Hello" --tokens 100
./chat  # 대화형 모드 (도구 호출 포함)
```

---

## Qwen3.5 Medium 모델 벤치마크 비교

> 출처: [Digital Applied - Qwen 3.5 Medium Models](https://www.digitalapplied.com/blog/qwen-3-5-medium-model-series-benchmarks-pricing-guide)

### 지식 및 추론

| 벤치마크 | 122B-A10B | 27B | 35B-A3B | GPT-5 mini | Claude Sonnet 4.5 |
|---------|-----------|-----|---------|-----------|------------------|
| MMLU-Pro | **86.7** | 86.1 | 85.3 | 83.7 | 80.8 |
| GPQA Diamond | **86.6** | 85.5 | 84.2 | 82.8 | 80.1 |
| HMMT Feb 2025 | 91.4 | **92.0** | 89.0 | 89.2 | 90.0 |
| MMMLU (다국어) | **86.7** | 85.9 | 85.2 | 86.2 | 78.2 |
| MMMU-Pro (비전) | **76.9** | 67.3 | 68.4 | 67.3 | 75.0 |

### 코딩 및 소프트웨어 엔지니어링

| 벤치마크 | 122B-A10B | 27B | 35B-A3B | GPT-5 mini | Claude Sonnet 4.5 |
|---------|-----------|-----|---------|-----------|------------------|
| SWE-bench Verified | 72.0 | **72.4** | 69.2 | 72.0 | 62.0 |
| Terminal-Bench 2 | **49.4** | 41.6 | 40.5 | 31.9 | 18.7 |
| LiveCodeBench v6 | 78.9 | **80.7** | 74.6 | 80.5 | 82.7 |
| CodeForces | 2100 | 1899 | 2028 | **2160** | 2157 |

### 에이전트/도구 사용

| 벤치마크 | 122B-A10B | 27B | 35B-A3B | GPT-5 mini | Claude Sonnet 4.5 |
|---------|-----------|-----|---------|-----------|------------------|
| BFCL-V4 (Tool Use) | **72.2** | 68.5 | 67.3 | 55.5 | 54.8 |
| BrowseComp (Search) | **63.8** | 61.0 | 61.0 | 48.1 | 41.1 |

---

## 35B-A3B vs 27B 실사용 비교

> 출처: [In-Depth Local Performance Comparison](https://sonusahani.com/blogs/qwen-27b-vs-qwen-35b) / [Which Model Fits Your GPU](https://insiderllm.com/guides/qwen35-local-guide-which-model-fits-your-gpu/)

### 실사용 속도 차이

| 환경 | 35B-A3B | 27B | 배수 |
|------|--------:|----:|-----:|
| RTX 3090 (Q4) | 111 tok/s | 34 tok/s | 3.3x |
| RTX 3090 (Q8) | 46 tok/s | 7.5 tok/s | 6.1x |
| Apple Silicon MLX | ~40-50 tok/s | ~25-30 tok/s | ~1.7x |

### 컨텍스트 길이에 따른 메모리 변화 (핵심!)

| 컨텍스트 | 35B-A3B (Q4) | 27B (Q4) |
|---------|:-----------:|:--------:|
| 4K | 22GB | 17GB |
| 262K (풀) | 25GB (+3GB) | 33GB (+16GB) |

35B-A3B는 MoE + DeltaNet(선형 어텐션)이라 긴 컨텍스트에서도 메모리 안정.
27B는 Dense라 컨텍스트 길어지면 메모리 16GB 폭증.

### 27B 알려진 이슈

- Ollama에서 도구 호출(tool calling) 비작동 ([GitHub #14493](https://github.com/ollama/ollama/issues/14493))
- Thinking 모드 호환성 문제
- 긴 문단 생성 시 10~20초 vs 35B-A3B 3초 미만

### 결론: 왜 대부분 35B-A3B를 선택하는가

- 벤치마크는 27B가 높지만 (짧은 프롬프트 기준)
- 실사용에서 속도 3~6배 + 컨텍스트 메모리 안정성이 결정적
- Mac 48~64GB에서 "가장 추천되는 모델" (InsiderLLM)

---

## Qwen3.5-35B-A3B 양자화 가이드

> 출처: [Unsloth GGUF Benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)

### 양자화별 품질/크기 비교

| 양자화 | 디스크 크기 | PPL (↓좋음) | KLD 99.9% (↓좋음) | 품질 손실 | M5 Pro 64GB |
|-------|--------:|------:|----------:|------:|:---:|
| Q2_K_XL | 12.0GB | 7.044 | 2.909 | 큼 | 가볍지만 품질 저하 |
| Q3_K_M | 15.5GB | 6.732 | 0.973 | 보통 | 메모리 절약용 |
| **Q4_K_M** | **18.5GB** | **6.605** | **0.548** | **미미** | **★ 추천 (최적 밸런스)** |
| Q4_K_XL | 19.2GB | 6.592 | 0.410 | 미미 | Q4_K_M보다 약간 나음 |
| Q5_K_XL | 23.2GB | 6.549 | 0.236 | 극소 | 품질 중시 |
| Q6_K_XL | 28.2GB | 6.539 | 0.144 | 거의 없음 | 여유 있으면 추천 |
| Q8_K_XL | 36.0GB | 6.535 | 0.103 | 무시 가능 | 최고 품질 (빠듯) |

### M5 Pro 64GB (42GB 여유) 양자화별 시뮬레이션

| 양자화 | 메모리 (4K) | 메모리 (262K) | 남는 메모리 | 판정 |
|-------|----------:|------------:|--------:|------|
| Q4_K_M | ~22GB | ~25GB | **17GB** | ★ 최추천 — 속도+품질+여유 밸런스 |
| Q6_K_XL | ~30GB | ~33GB | 9GB | 품질 업, 여유 줄어듦 |
| Q8_K_XL | ~38GB | ~41GB | 1GB | 최고 품질, 매우 빠듯 |

---

## 전체 비교표 (M5 Pro 64GB 기준)

### 성능 + 실용성 종합

| 모델 | 활성 파라미터 | 메모리 | 속도 | 한국어 | 코딩 | 에이전트 | 비전 | 난이도 |
|------|----------:|------:|-----:|:---:|:---:|:---:|:---:|:---:|
| **Qwen3-Coder-Next 80B-A3B** | 3B | ~15GB | ~25 tok/s | ★★ | ★★★ | ★★★ | X | 쉬움 |
| **Qwen3-Coder 30B-A3B** | 3.3B | ~18GB | ~22 tok/s | ★★ | ★★★ | ★★★ | X | 쉬움 |
| **★ Qwen3.5-35B-A3B (추천)** | 3B | ~20GB | ~20 tok/s | ★★★ | ★★ | ★★ | O | 쉬움 |
| **Qwen3.5-27B** | 27B (Dense) | ~17GB | ~15 tok/s | ★★★ | ★★★ | ★★ | O | 쉬움 |
| **Qwen3.5-122B-A10B** | 10B | ~40GB | ~10 tok/s | ★★★ | ★★ | ★★★ | **최강** | 쉬움 |
| **Mistral Large 3** | 41B | ~25GB | ~12 tok/s | ★ | ★★ | ★★ | X | 쉬움 |
| **Llama 3.3 70B** | 70B (Dense) | ~35GB | ~8 tok/s | ★ | ★★ | ★★ | X | 쉬움 |
| **DeepSeek V3.2 Speciale** | MoE | ~40GB | ~8 tok/s | ★★ | ★★★ | ★★ | X | 쉬움 |
| **GPT-OSS 120B** | 120B (Dense) | ~60GB | 빠듯 | ★ | ★★ | ★★ | X | 어려움 |
| **Qwen3.5-397B (flash-moe)** | 17B | 5.5GB+SSD | ~3-4 tok/s | ★★★ | ★★★ | ★★★ | O | 빌드 필요 |

### 메모리 여유 시뮬레이션 (42GB 기준)

| 조합 | 메모리 합계 | 남는 메모리 | 가능? |
|------|--------:|--------:|:---:|
| **35B-A3B 단독 (추천)** | 20GB | **22GB** | **여유** |
| Coder-Next + 35B-A3B | 35GB | 7GB | O |
| Coder-Next + 27B | 32GB | 10GB | O |
| 27B + 35B-A3B | 37GB | 5GB | 빠듯 |
| 122B-A10B 단독 | 40GB | 2GB | 빠듯 |
| Llama 70B 단독 | 35GB | 7GB | O |
| 397B flash-moe | 5.5GB | 36GB | 넉넉 |

### 시나리오별 최종 추천

| 용도 | 추천 | 메모리 | 이유 |
|------|------|------:|------|
| **★ 올라운더 (최추천)** | **Qwen3.5-35B-A3B** | 20GB | 한국어+코딩+비전, 20GB로 여유, ~20 tok/s 빠름 |
| 코딩 올인 | Qwen3-Coder-Next 80B-A3B | 15GB | 코딩 에이전트 최강, 가볍고 빠름 |
| 코딩 품질 최고 | Qwen3.5-27B | 17GB | SWE 72.4, LiveCode 80.7, Dense라 품질 안정 |
| 에이전트/도구 호출 | Qwen3.5-122B-A10B | 40GB | BFCL 72.2, 도구 사용 압도적 |
| 최고 지능 | Qwen3.5-397B (flash-moe) | 5.5GB+SSD | 느리지만 397B급 두뇌. 빌드 도전 |
| 듀얼 운용 | 35B-A3B + Coder-Next | 35GB | 채팅은 35B, 코딩은 Coder로 분업 |
| 속도 최우선 | Qwen3.5-9B | 6GB | 40+ tok/s, 간단한 작업용 |

## llmfit 검증 결과 (2026-03-22)

[llmfit](https://github.com/AlexsJones/llmfit)으로 M5 Pro 64GB에서 501개 호환 모델 자동 분석.

| 상태 | 모델 | Score | tok/s 예측 | 메모리% |
|:---:|------|:---:|------:|------:|
| Good | Qwen3-Coder-Next 80B-A3B | 99 | 105 | 64% |
| Perfect | Qwen3.5-122B-A10B (NVFP4) | 96 | 62 | 52% |
| Perfect | Qwen3-Coder 30B-A3B | 95 | 70 | 24% |
| **Perfect** | **Qwen3.5-35B-A3B (선택)** | **92** | **105** | **29%** |
| Perfect | GPT-OSS 20B | 91 | 64 | 17% |
| Marginal | GPT-OSS 120B | 83 | 33 | 93% |

- Perfect = 메모리 여유 충분 / Good = 가능하지만 빠듯 / Marginal = 타이트
- 실측 103 tok/s vs 예측 105 tok/s → 거의 일치
- 설치: `brew install llmfit` / 실행: `llmfit fit`

---

## 참고 사항

- **MoE 모델**: 전체 파라미터 중 일부만 활성화 → 큰 모델의 지능 + 작은 모델의 속도
- **Q4 양자화**: 원본 대비 성능 손실 5% 미만, 메모리 절약 최고
- **Apple Silicon 장점**: GPU와 RAM이 통합(Unified Memory)이라 VRAM 따로 불필요
- **MLX**: Apple 네이티브 ML 프레임워크, llama.cpp보다 20~30% 빠름
- **권장**: 총 메모리의 70% 이하로 모델 사용 (64GB 기준 ~45GB까지)

## 참고 링크

- [Best Local LLMs for Apple Silicon Mac 2026](https://apxml.com/posts/best-local-llms-apple-silicon-mac)
- [Qwen3-Coder Ollama](https://ollama.com/library/qwen3-coder)
- [Qwen3-Coder-Next Ollama](https://ollama.com/library/qwen3-coder-next)
- [GPT-OSS 120B Hugging Face](https://huggingface.co/openai/gpt-oss-120b)
- [Qwen3.5 Ollama](https://ollama.com/library/qwen3.5)
- [Open Source LLM Leaderboard 2026](https://onyx.app/open-llm-leaderboard)
- [flash-moe (Dan Woods GitHub)](https://github.com/danveloper/flash-moe)
- [autoresearch (Karpathy GitHub)](https://github.com/karpathy/autoresearch)
- [Simon Willison - LLM in a Flash](https://simonwillison.net/2026/Mar/18/llm-in-a-flash/)
- [Apple "LLM in a Flash" 논문](https://machinelearning.apple.com/research/efficient-large-language)
- [llmfit (모델-하드웨어 매칭 도구)](https://github.com/AlexsJones/llmfit)
