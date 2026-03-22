M5 Pro 64GB 노트북을 사고, 로컬 LLM의 최적해를 찾아 떠난 여정.

요즘 로컬 LLM 모델이 쏟아지고 있습니다.
Qwen3.5, Llama 3.3, GPT-OSS, DeepSeek, Mistral...
도대체 뭘 골라야 하지? 에서 시작했습니다.

벤치마크 점수만 보면 Qwen3.5-27B가 코딩 1등이었는데,
실제로 써본 사람들의 이야기는 달랐습니다.

결론: Qwen3.5-35B-A3B

35B 파라미터인데 실제로는 3B만 활성화되는 MoE 구조.
쉽게 말하면, 전문가 256명이 있는데 질문마다 적합한 소수만 투입되는 방식입니다.
덕분에 큰 모델의 지능은 유지하면서, 속도는 3~6배 빠릅니다.

Apple Silicon의 MLX로 돌려보니:
- 초당 103 토큰 생성 (체감상 거의 실시간)
- 메모리 19.6GB만 사용 (64GB 중 30% 수준)
- GPT-5 mini보다 높은 벤치마크 점수 (MMLU 85.3 vs 83.7)

API 서버로 띄우면 집에 있는 다른 기기에서도 접속 가능하고,
컨텍스트도 262K에서 1M까지 확장할 수 있습니다.

클라우드 API 비용 없이, 내 데이터 유출 걱정 없이, 오프라인에서도.
이 정도면 로컬 LLM, 이제 충분히 실용적인 단계에 온 것 같습니다.

따라해보고 싶으신 분들을 위해 빠른 셋업 가이드:

```
python3 -m venv .venv
source .venv/bin/activate
pip install mlx-lm
mlx_lm.chat --model mlx-community/Qwen3.5-35B-A3B-4bit
```

4줄이면 끝입니다. 모델은 첫 실행 시 자동 다운로드(~19GB).
자세한 셋업 과정은 GitHub에 정리해뒀습니다.

https://github.com/LeeKiYoung/local-llm

#LocalLLM #AppleSilicon #Qwen3.5 #OnDeviceAI
