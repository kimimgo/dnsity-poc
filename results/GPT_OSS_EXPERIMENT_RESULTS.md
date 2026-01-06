# gpt-oss-20b 기반 실험 결과

**실험 버전**: v2.0.0 (NIAH), v2.1.0 (KMMLU)
**실험일**: 2026-01-07
**모델**: OpenAI gpt-oss-20b (21B params, 3.6B active, MoE, MXFP4)
**추론 엔진**: vLLM 0.13.0
**GPU**: RTX 4090 (24GB VRAM)

---

## 1. 실험 결과 요약

### NIAH (Needle In A Haystack) 평가

| Dataset | Samples | Accuracy | Notes |
|---------|---------|----------|-------|
| **Global NIAH** | 200 | **100.0%** (200/200) | Perfect retrieval |
| **Korean NIAH** | 200 | **98.5%** (197/200) | Near-perfect retrieval |
| **Average** | 400 | **99.25%** | Excellent NIAH performance |

### Position-wise Accuracy (Korean NIAH)

| Position Range | Accuracy | Samples |
|----------------|----------|---------|
| 20-40% | 100.0% | 63 |
| 40-60% | 95.5% | 66 |
| 60-80% | 100.0% | 71 |

### KMMLU (Korean Massive Multitask Language Understanding)

| Subject | Samples | Accuracy | vs Baseline |
|---------|---------|----------|-------------|
| **Law** | 100 | **31.0%** (31/100) | +6%p vs Llama-3-8B |

---

## 2. 모델 비교

### NIAH 성능 비교

| Model | Global NIAH | Korean NIAH | Average |
|-------|-------------|-------------|---------|
| **gpt-oss-20b** | **100.0%** | **98.5%** | **99.25%** |
| Llama-3-8B + Gist (25 tokens) | 20.0% | 49.0% | 34.5% |
| Llama-3-8B + RAG | ~60% | ~50% | ~55% |
| Llama-3-8B (Full Context) | ~95% | ~90% | ~92.5% |

### KMMLU 성능 비교

| Model/Method | Law (100) | vs Baseline |
|--------------|-----------|-------------|
| Llama-3-8B (Baseline) | 25.0% | - |
| Llama-3-8B + RAG | 31.0% | +6%p |
| Llama-3-8B + LoRA | 24.0% | -1%p |
| Llama-3-8B + Gist | 23.0% | -2%p |
| **gpt-oss-20b** | **31.0%** | **+6%p** |

---

## 3. 핵심 인사이트

### gpt-oss-20b의 강점

1. **NIAH 성능**: 99.25% 평균 정확도로 거의 완벽한 정보 추출 능력
2. **추론 능력**: Reasoning 모드를 통한 단계별 사고 과정 출력
3. **효율성**: MoE + MXFP4로 16GB VRAM에서 실행 가능 (3.6B 활성 파라미터)
4. **한국어 지원**: Korean NIAH 98.5%로 한국어 처리 우수

### 비교 분석

1. **vs Llama-3-8B**:
   - NIAH: 99.25% vs 34.5% (Gist) → **+64.75%p** 향상
   - KMMLU: 31% vs 25% → **+6%p** 향상 (RAG와 동일 수준)

2. **vs RAG**:
   - NIAH에서는 gpt-oss-20b가 월등히 우수
   - KMMLU에서는 동일 수준 (둘 다 31%)
   - **결론**: 모델 크기 증가 = RAG 추가 효과와 유사

3. **vs Gist Token**:
   - Gist Token (34.5%) << gpt-oss-20b (99.25%) on NIAH
   - Gist Token은 압축 목적이므로 직접 비교는 부적절
   - **Gist Token + 더 큰 모델**이 유망한 방향

---

## 4. 실험 환경

### vLLM 서버 설정

```bash
vllm serve openai/gpt-oss-20b \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code
```

### 모델 구성

- Architecture: GptOssForCausalLM
- Quantization: MXFP4 (Marlin backend)
- Attention: TRITON_ATTN
- Max sequence length: 4096 tokens
- Memory usage: ~14GB (including KV cache)

### 평가 스크립트

- KMMLU: `scripts/eval_kmmlu_vllm.py`
- NIAH: `scripts/eval_niah_vllm.py`

---

## 5. 결과 파일

| Evaluation | File |
|------------|------|
| KMMLU Law | `results/kmmlu/gptoss20b_law_100.json` |
| Global NIAH | `results/gptoss20b_niah_global_niah_200.json` |
| Korean NIAH | `results/gptoss20b_niah_korean_niah_200.json` |

---

## 6. 결론 및 향후 계획

### 주요 결론

1. **gpt-oss-20b는 NIAH에서 거의 완벽한 성능** (99.25%)
2. **KMMLU에서 Llama-3-8B + RAG와 동일 수준** (31%)
3. **MoE 아키텍처로 효율적인 추론** (3.6B 활성 파라미터)
4. **Reasoning 모드**가 복잡한 질문에 도움

### 향후 실험 계획

1. **Gist Token + gpt-oss-20b**: 더 큰 모델에서 Gist Token 효과 검증
2. **RAG + gpt-oss-20b**: 지식 증강 효과 측정
3. **KMMLU 전체 평가**: 1,000 샘플 평가
4. **LongBench/RULER**: 표준 장문 이해 벤치마크

---

## 7. 부록: 샘플 응답

### KMMLU 응답 예시 (Reasoning)

```
Question: 경비업법령상 시설주가 무기를 지급할 수 있는 특수경비원은?

Reasoning: We need to know Korean law: The Security Industry Act (경비업법)
regulates security guards. Under the act, special security guards (특수경비원)
can be armed if they meet certain conditions...

Answer: C
```

### NIAH 응답 예시

```
Question: What is the secret passkey mentioned in the text?

Answer: HJFN5P

(100% accuracy on Global NIAH)
```
