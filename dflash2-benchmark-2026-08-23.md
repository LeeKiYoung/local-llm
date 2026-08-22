# dFlash2 실측 — Qwen3.8-27B-8bit @ M5 Pro 64GB (2026-08-23)

## 셋업
- `mlx-dspark` 0.15.1 (별도 venv, 프로젝트 .venv 무손상)
- target `mlx-community/Qwen3.8-27B-8bit` (기존 캐시 재사용)
- drafter `incoai/Qwen3.8-27B-DFlash2` (3.85GB 추가 다운로드)
- cfg: block_size=8, selector_top_k=16, conv_kernel_size=2 → 블로그 dFlash2 설명과 일치
- 피크 메모리 **33.4GB** (기존 34GB와 사실상 동일)
- 조건: thinking OFF (프로덕션과 동일), temp=0, max_new_tokens=512

## 결과 (tok/s)
| 프롬프트 | baseline | dflash | dflash md4 | 배속(md4) | accept/8 |
|---|---|---|---|---|---|
| code_gen    | 9.71 | 32.20 | 33.02 | **3.40×** | 6.25 |
| code_edit   | 9.84 | 22.41 | 25.39 | **2.58×** | 4.13 |
| tool_reason | 9.90 | 18.63 | 23.92 | **2.42×** | 3.52 |
| prose_ko    | 9.72 | 13.68 | 17.15 | **1.76×** | 2.62 |
| **평균**    | 9.79 | 2.22× | | **2.54×** | |

- baseline 9.7~9.9 tok/s = 기존 기록 9.8과 일치 (측정 유효성 확인)
- 12/16은 8과 동일 결과(block_size=8에서 클램프)
- 코드일수록 acceptance 높음, 한국어 산문이 최악
- **실사용 지표: 코딩/에이전트 3개 평균 2.80× (24~33 tok/s).** prose_ko는 실트래픽 비중이 낮아 4개 평균(2.54×)은 과소평가
- **max_draft_tokens 정정 (n=3 반복 측정, 256토큰 warm)**: md4는 일관된 이득이 아니라 워크로드 의존
  | | default(block8) | md4 | md4/default |
  |---|---|---|---|
  | code_gen | [38.17, 38.22, 38.16] med 38.17 | [34.38, 34.41, 34.43] med 34.41 | **0.902×** |
  | tool_reason | [19.39, 18.67, 18.50] med 18.67 | [22.83, 22.67, 22.78] med 22.78 | **1.221×** |
  - 기제: acceptance 높으면 큰 블록이 유리(한 번에 많이 통과), 낮으면 작은 블록이 유리(드래프터 낭비 감소)
  - **결론: 기본값(block 8) 유지가 안전.** 최대 이득 구간인 코드 생성에서 md4는 10% 손해
  - 재현성 매우 높음(spread 0.05~0.89) → 앞선 sweep의 18.02 vs 18.58 및 첫 벤치의 md4 소폭 우위는
    노이즈가 아니라 콜드/워밍업 효과였음. 512토큰 표의 md4 열도 같은 이유로 과대평가 가능

## 주장 대비
- README/블로그 **3.6×는 재현 안 됨** (평균 2.54×). 단 코드 생성 단독은 3.40×로 근접
- REGISTRY 주석에 speedup은 M4 Pro 측정이라 명시돼 있음
- **"lossless" 불성립**: temp=0에서 baseline과 bit-identical 아님
  - baseline 자기 재현 True / dflash 자기 재현 True / 둘이 서로 다름
  - 토큰 divergence 지점: #1, #16, #136, #268
  - 출력 품질은 동등: code_gen(변수명/쿼트/엣지케이스), prose_ko("아키텍처를 갖추고"↔"구조를 가지고"), tool_reason(contextlib↔prometheus_client 접근 차이) 모두 유효한 대안
  - 영향: 응답 캐싱, 재현 가능한 eval에는 결정성 가정 못 함

## 스트리밍 (통합 가능성)
- `dflash_generate(on_text=...)` 콜백이 라운드당 1회 호출
- 162토큰 → 39청크, 청크 간격 median 125ms / max 176ms, 청크당 ~18자(≈4토큰)
- `"".join(chunks) == r.text` True → 유실 없음
- 토큰 단위는 아니지만 SSE 청크로 그대로 전달 가능. `make_chunk()`에 문자열 그대로 투입
- 드래프터 cfg `sliding_window=2048` → 드래프터 KV 캐시는 컨텍스트에 따라 커지지 않음. 262K/1M 프로필은 여전히 타겟 캐시가 지배적이라 현재와 동일

## 이미지 — 무음 실패 확인 (중요)
dflash 경로는 이미지를 **처리하지 못하는데 에러도 내지 않는다.**
- `target.is_vlm=False`, 모듈 `mlx_lm.models.qwen3_5`, vision tower 없음
  (Target 클래스가 qwen3_5를 mlx-lm 텍스트 전용 모듈로 라우팅. `anthropic_api.py`도 "image blocks are dropped" 명시)
- chat template은 `<|vision_start|><|image_pad|><|vision_end|>`를 넣지만 `<|image_pad|>`가 **1개**
  (448x448 이미지 = 전체 31토큰. 실제 VLM이면 수백 개로 확장돼야 함)
- 테스트: 파란 원 안 노란 사각형 → 응답 "The image shows a white square with a black border." **완전 환각**
- **통합 시 필수**: image 파트가 있는 요청은 반드시 기존 mlx_vlm 경로로 라우팅. 그냥 흘리면 조용히 거짓 응답

## ★ 대안 발견: mlx-vlm native MTP (권장)
`mlx-community/Qwen3.8-27B-MTP-8bit` — **0.48GB, 추가 패키지 없음, 프로젝트 .venv에서 그대로 동작**
- `model_type: qwen3_5_mtp` → mlx-vlm 0.6.13의 `DRAFTER_KIND_BY_MODEL_TYPE`에 등록됨, kind 자동 감지
- 실행: `python -m mlx_vlm.server --model mlx-community/Qwen3.8-27B-8bit --draft-model mlx-community/Qwen3.8-27B-MTP-8bit`
  → 로그 "Drafter ready; speculative decoding enabled." / block_size=3

### 동일 서버 A/B (서버 보고 decode tok/s, thinking off, temp=0)
| | baseline | MTP | 배속 |
|---|---|---|---|
| code_gen | 9.7 | 22.5 | **2.32×** |
| tool_reason | 9.7 | 18.1 | **1.87×** |
| prose_ko | 9.8 | 17.3 | **1.77×** |
| image | 10.3 | 21.9 | **2.13×** |

### 이미지 — 정상 동작 (dflash2와 결정적 차이)
- 같은 테스트 이미지(파란 원 안 노란 사각형)에 **"There is a yellow square inside a blue circle in this image."**
- baseline과 답 동일 → speculative가 이미지 품질을 해치지 않음
- 이미지 요청도 decode 21.9 tok/s로 가속됨 → speculative가 이미지 경로에서도 실제 작동
- mlx-vlm 서버 코드에 draft_model과 pixel_values를 배제하는 로직 없음 (제약은 logits_processors, thinking_budget뿐)

### MTP는 "작은 모델로 바꿔서 빠른 것"이 아님 (확인)
- `--model`은 여전히 `Qwen3.8-27B-8bit`. MTP-8bit(0.48GB)은 `--draft-model`, 후보만 제시하고 타겟이 검증
- 드래프터 config: `is MoE: False` (dense), `text_config`는 타겟 스펙(64층/5120) 참조용, `vision_config` 포함
- **출력 동일성 (baseline vs MTP, 같은 서버·thinking off·temp=0)**
  | | identical | 비고 |
  |---|---|---|
  | code_gen | **True** | 808자 완전 일치 |
  | tool_reason | **True** | 1025자 완전 일치 |
  | prose_ko | False | 165자 지점 1곳 |
- prose_ko 차이는 near-tie 증거: dflash2 테스트에서는 baseline "아키텍처를 갖추고"/dflash "구조를 가지고",
  MTP 테스트에서는 baseline "구조를 가지고"/MTP "아키텍처를 갖추고" — **같은 쌍이 역전됨.**
  두 후보 확률이 거의 같은 자리라 부동소수점 차이로 어느 쪽이든 뽑힘. speculative 결함 아님
- dflash2는 4개 프롬프트 전부 갈라짐(토큰 #1/#16/#136/#268) → MTP가 훨씬 충실

### dflash2(mlx-dspark) 대비
| | dflash2 | MTP native |
|---|---|---|
| code_gen | **3.32×** (32.2) | 2.32× (22.5) |
| tool_reason | 1.88× | 1.87× |
| prose_ko | 1.41× | **1.77×** |
| 이미지 | **환각** | **정답** |
| 추가 용량 | 3.85GB + 별도 패키지 | **0.48GB, 없음** |
- 코드 생성 최고 속도만 보면 dflash2, 그 외 전부 MTP native 우위
- (dflash2는 in-process gen-only, MTP는 서버 decode rate 측정. baseline이 양쪽 9.7~9.9로 동일해 배속 비교는 유효)

## 기각된 경로
- **mlx-vlm native + DFlash2**: 불가. mlx-vlm 0.6.13의 `DFlashDraftModel`은 DFlash1 세대.
  DFlash2의 `candidate_selector`(bilinear selector) + `attention_conv`/`mlp_conv`(two-tap conv) 가중치 23개를
  `ValueError`로 명시 거부 (무음 드롭 아님)
- **MTPLX** (`Youssofal/Qwen3.8-27B-MTPLX-*`): mlx-vlm이 `mlx_lm_extra_tensors`를 모름(grep 0건).
  MTP가 별도 드래프터 리포가 아니라 `mtp.safetensors`로 모델에 동봉된 mlx-lm 규약. 30GB 다운로드 중단
- **Qwen3.8-9B**: 드래프터 없음 + base_model이 Qwen3.5-9B인 서드파티 distill

## 미해결
- 대안 경로 MTPLX 측정 중. `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality` = 8-bit 30GB,
  base_model `Qwen/Qwen3.8-27B` 정품, 파일에 `model-vision.safetensors` + `mtp.safetensors` 둘 다 존재
  → 멀티모달 유지 + 스펙 디코딩이라 우리 Core Value에 더 맞을 가능성
  (`-Optimized-Speed`는 4-bit 20.7GB)
## 기각
- Qwen3.8-9B: dflash/dspark 드래프터 HF에 없음(검색 0건). `empero-ai/Qwen3.8-9B-Distill`은 base_model이
  `Qwen/Qwen3.5-9B`인 서드파티 distill이라 "3.8의 작은 버전"이 아님. 품질 저하로 기각

---

## 최종: 우리 서버(llm-api-server.py)에 MTP 적용 후 실측 (2026-08-23)

원인: 기존 코드가 `draft_kind`/`draft_block_size`만 넘기고 **`draft_model`을 넘기지 않아** speculative 무동작.
`generate_step`은 `draft_model`도 받는다 (`_MTP_SUPPORTED` 감지가 `draft_kind`만 확인해 이를 놓쳤음).

동일 서버·동일 프롬프트·warm·non-stream:
| | run1 | run2 | run3 |
|---|---|---|---|
| baseline (`--no-draft`) | 8.82 | 9.56 | 9.57 tok/s |
| MTP (기본) | 22.19 | 25.21 | 25.24 tok/s |

**2.64×** (9.57 → 25.24). `block_size=6`이 3보다 빠름(27.65 vs 22.47, 직접 generate 측정) — 기존 기본값 6 유지.

검증 완료: 이미지 정답("big blue circle and a small yellow square"), tool calling 정상(`finish_reason=tool_calls`),
기존 테스트 108개 통과, 드래프터 로드 실패 시 speculative만 끄고 서버 계속.

### 함께 수정한 기존 이슈: thinking OFF 스트리밍 무효화
thinking OFF(기본)일 때 스트리밍이 실질적으로 동작하지 않았다. `_stream_response`가 `</think>`를
기다리며 전체를 버퍼링하는데, thinking OFF면 chat template이 `<think>\n\n</think>`를 완성형으로
prefill하므로 생성 텍스트에 `</think>`가 나오지 않는다(실측 12/12건 0회). 종료 태그를 끝까지
기다리다 마지막에 전체를 한 번에 내보내고 있었다.

수정: `thinking_done` 초기화에 `not enable_thinking`을 추가해 pass-through. 도달 불가가 된
잔여 flush 블록 제거.

| | 수정 전 | 수정 후 |
|---|---|---|
| thinking OFF (기본) | 청크 3개 | **청크 33개** |
| thinking ON | 청크 33개 | 33개 (회귀 없음) |

청크 도착 간격 70~90ms로 점진 확인. think 태그 누출 0건.
스트리밍+tools 조합도 정상(`finish_reason=tool_calls`, 구조화 tool_calls, content/`<tool_call>` 누출 없음).

## 무검열 모델(abliterated)에도 동일 드래프터 적용 — 실측

타겟: `orcarouter/Qwen3.8-27B-Uncensored-MLX` 8-bit (abliterated)
드래프터: 동일한 `mlx-community/Qwen3.8-27B-MTP-8bit` (정품 Qwen3.8-27B 기반)

| 프롬프트 | no draft | MTP | 배속 | 정품 모델 배속 |
|---|---|---|---|---|
| code | 9.37 | **28.56** | 3.05× | 3.16× |
| prose_ko | 9.80 | 16.36 | 1.67× | 1.77× |

- 드래프터 로드/바인딩 정상 (`kind=mtp`)
- **abliteration으로 인한 acceptance 손실은 정품 대비 ~3~6%뿐.** 정품 드래프터가 거부 방향을
  제안해 리젝이 늘 것이라 예상했으나 실측에서는 거의 차이 없음
- 품질/무검열 특성은 구조적으로 안전 — 최종 토큰은 타겟(abliterated)이 검증해 내보내므로
  드래프터가 무엇을 제안하든 출력 분포는 타겟의 것
- **피크 36.3GB** (정품 33.4GB보다 높음 — 이 체크포인트가 28GB로 약간 더 큼). 48GB 환경은 타이트
- `llm-server.sh`의 활성화 조건(`Qwen3.8-27B` 문자열 매칭)에 이 경로가 걸리므로 자동 ON

## 컨텍스트 길이별 디코딩 속도 + MTP 기여분 (계측 도입 후 실측)

같은 프롬프트, 캐시 히트 상태(prefill≈16토큰)에서 `decode_tps`만 비교. GPU 독점(dsh 중단) 상태.

| 컨텍스트 | MTP OFF | MTP ON | 배속 |
|---|---|---|---|
| 4,714 | 9.7 | **21.7** | 2.24× |
| 18,754 | 9.1 | **19.1** | 2.10× |
| 36,434 | 8.1 | **15.0** | 1.85× |

- **컨텍스트 길이 자체의 영향은 작다.** MTP OFF 기준 4.7K→36K에서 9.7→8.1 (16% 감소).
  "긴 컨텍스트에서 디코딩이 반토막" 가설은 기각.
- **MTP는 36K에서도 1.85배 기여.** 배속이 줄긴 하지만 무용지물이 아니다.

### 합성 벤치가 실사용을 과대평가한다 (중요)
dsh 실사용 로그의 같은 구간(ctx 37~38K)은 `DECODE=7.0~8.3`으로,
위 표의 **MTP OFF 수치(8.1)와 사실상 동일**했다. 즉 실제 대화에서는 MTP가 거의 기여하지 못했다.

원인 추정: 합성 프롬프트는 같은 문장 반복이라 드래프터 적중률이 비정상적으로 높다.
실제 대화는 도구 출력·JSON·에러 로그·파일 목록이 섞여 드래프터가 다음 토큰을 맞추기 어렵다.

→ **이 문서 앞부분의 2.64배 / 3.16배는 합성 프롬프트 기준값이며, 실사용 상한이 아니다.**
   실사용 개선 레버는 "긴 컨텍스트"가 아니라 **실제 대화 내용에서의 드래프터 적중률**이다.

### 프리픽스 캐시는 매우 효과적 (단, 슬롯 1개)
| | prefill | cached | 소요 |
|---|---|---|---|
| 첫 턴 (34K) | 34,198 | 0 | 125s (프리필 97s + 디코딩 28s) |
| 이후 턴 | 156 | 34,478 | **8.9s** |

- 14배 단축. 부분 히트도 정상 동작(새 도구 출력 1,500토큰만 프리필).
- **슬롯이 하나뿐**이라 다른 대화·부수 요청(세션 타이틀 등)이 끼면 덮어써지고 다음 턴이 전체 재프리필된다.
  실측: 벤치 요청이 끼어든 직후 dsh 턴이 `cached=0`으로 38K 전체 재프리필(130s).
- `queue_wait_ms`로 직렬화 비용도 확인: ctx=133/gen=8 요청이 큐 대기만 56초(조용할 때 동일 크기는 1.7s).
