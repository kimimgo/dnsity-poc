# PoC Plan: NoLiMa 기반 장문맥 처리 기술 비교 실험

**Version**: 1.0
**Date**: 2026-01-05
**Authors**: Gemini-3-Pro + Claude Opus 4.5

---

## Executive Summary

본 PoC는 NoLiMa (No Literal Matching) 평가 기준을 적용하여 **3가지 장문맥 처리 기술**의 진정한 이해 능력을 객관적으로 비교합니다.

| 기술 | 핵심 메커니즘 | 예상 강점 | 예상 약점 |
|------|-------------|----------|----------|
| **RAG** | 검색 기반 청크 선택 | 메모리 효율 | 어휘 분리 시 검색 실패 |
| **LoRA** | 문맥별 임시 학습 | 높은 정확도 | 시간/비용 비효율 |
| **Gist Token** | 토큰 압축 | 전역 문맥 유지 | 세부 정보 손실 |

**환경**: RTX 4090 (24GB) + Llama-3-8B-Instruct (4-bit QLoRA)

---

## 1. 실험 목표 및 가설

### 1.1 핵심 연구 질문

> **"텍스트의 표면적 일치(Literal Match)가 없는 상황에서, 긴 문맥 속에 숨겨진 정보를 추론해낼 수 있는가?"**

### 1.2 구체적 목표

1. **객관적 비교**: NoLiMa 기준으로 3가지 기술의 정확도 측정
2. **효율성 분석**: VRAM, 속도, 준비 시간 비교
3. **확장성 검증**: 8K → 16K → 32K 문맥 길이별 성능 변화
4. **실용성 평가**: RTX 4090 환경에서의 실제 배포 가능성

### 1.3 가설

| 기술 | 가설 | 근거 |
|------|------|------|
| **RAG** | NoLiMa에서 **가장 취약** | 어휘 분리 시 임베딩 유사도 저하 → 관련 청크 검색 실패 |
| **LoRA** | **정확도 최고**, 효율 최저 | 문맥 전체를 학습하므로 과적합 유도 가능, 하지만 매번 학습 필요 |
| **Gist** | **균형잡힌 성능** | 전역 문맥 유지하나 압축 손실 존재, RAG보다 우수 예상 |

### 1.4 예상 결과

```
NoLiMa 정확도 예측:
┌────────────────────────────────────────────┐
│  LoRA     ████████████████████  75-85%     │
│  Gist     ████████████          50-65%     │
│  RAG      ████████              35-50%     │
│  Random   █                     ~5%        │
└────────────────────────────────────────────┘
```

---

## 2. 평가 지표 설계

### 2.1 정확도 지표 (Quality Metrics)

#### Primary: NoLiMa Exact Match
```python
def nolima_score(prediction: str, answer: str) -> bool:
    """정확한 답변 추출 여부"""
    # 6자리 코드 추출
    pred_code = extract_code(prediction)
    return pred_code.upper() == answer.upper()
```

#### Secondary: Semantic Match (LLM-as-Judge)
```python
def semantic_match(prediction: str, answer: str, context: str) -> float:
    """의미적 정확도 (0-1)"""
    prompt = f"""
    Context: {context}
    Expected Answer: {answer}
    Model Answer: {prediction}

    Does the model's answer correctly identify the information?
    Return only: CORRECT, PARTIAL, or WRONG
    """
    return llm_judge(prompt)  # GPT-4o or Llama-70B
```

#### RAG-Specific: Retrieval Recall@K
```python
def retrieval_recall(retrieved_chunks: list, needle_text: str, k: int) -> bool:
    """Top-K 청크에 needle이 포함되어 있는지"""
    for chunk in retrieved_chunks[:k]:
        if needle_text in chunk:
            return True
    return False
```

### 2.2 효율성 지표 (Efficiency Metrics)

| 지표 | 측정 방법 | 단위 |
|------|----------|------|
| **VRAM Peak** | `torch.cuda.max_memory_allocated()` | GB |
| **Throughput** | 생성 토큰 수 / 추론 시간 | tokens/sec |
| **Latency (First Token)** | 첫 토큰 생성까지 시간 | ms |
| **Prep Time** | 인덱싱/학습/압축 시간 | sec |

### 2.3 확장성 지표 (Scalability Metrics)

| 문맥 길이 | 토큰 수 | 테스트 샘플 |
|----------|--------|------------|
| Short | 4K | 50개 |
| Medium | 8K | 50개 |
| Long | 16K | 50개 |
| Very Long | 32K* | 50개 |

*32K는 메모리 제약으로 일부 기술만 테스트

---

## 3. 데이터셋 설계: NoLiMa-NIAH

### 3.1 NoLiMa 원칙 (필수 준수)

| 원칙 | 기준 | 검증 방법 |
|------|------|----------|
| **Lexical Decoupling** | Overlap < 10% | ROUGE-L score < 0.1 |
| **Semantic Association** | 동의어/유의어 사용 | 수동 검토 |
| **Context Diversity** | TTR > 0.3 | Type-Token Ratio 계산 |
| **Question Variation** | 최소 20가지 템플릿 | 다양성 점검 |

### 3.2 하이브리드 생성 전략 (총 200개)

#### Phase 1: 규칙 기반 (50개) - 완전 통제
```python
# 논리적 관계 기반 샘플
templates = {
    "family": {
        "needle": "{A}는 {B}의 아버지이다.",
        "question": "{A}와 {C}의 법적 관계는?",  # C는 B의 자녀
        "answer": "조부"
    },
    "security": {
        "needle": "시스템 접근에는 인증 토큰 {CODE}가 필요합니다.",
        "question": "보안 자격 증명의 값은?",
        "answer": "{CODE}"
    }
}
```

**도메인 분포**:
- 보안/인증: 15개
- 금융/거래: 10개
- 의료/기록: 10개
- 법률/계약: 10개
- 기술/시스템: 5개

#### Phase 2: LLM 생성 (150개) - 다양성 확보
```python
def generate_nolima_sample_gpt4(domain: str):
    prompt = f"""
Generate a NoLiMa-style NIAH sample for domain: {domain}

Requirements:
1. Context: 300-500 words, coherent narrative, NO repetition
2. Needle: Embed a 6-character code naturally
3. Question: Ask about the code using DIFFERENT vocabulary
4. Lexical overlap between question and needle: < 10%

Format:
{{
    "context": "...",
    "needle_phrase": "...",
    "question": "...",
    "answer": "CODE"
}}

Examples of good lexical decoupling:
- needle: "authentication token" → question: "security credential"
- needle: "access code" → question: "entry identifier"
- needle: "verification PIN" → question: "confirmation passcode"
"""
    return openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
```

#### Phase 3: 품질 검증 (자동화)
```python
def validate_sample(sample: dict) -> bool:
    """NoLiMa 기준 자동 검증"""
    q = sample['question'].lower()
    n = sample['needle_phrase'].lower()

    # 1. Lexical overlap 검사
    q_words = set(q.split())
    n_words = set(n.split())
    overlap = len(q_words & n_words) / len(q_words)
    if overlap > 0.1:
        return False  # 10% 초과

    # 2. TTR 검사
    context_words = sample['context'].lower().split()
    ttr = len(set(context_words)) / len(context_words)
    if ttr < 0.3:
        return False

    # 3. 코드 형식 검사
    if not re.match(r'^[A-Z0-9]{6}$', sample['answer']):
        return False

    return True
```

### 3.3 Haystack 구성

**소스 다양화**:
```python
haystack_sources = {
    "wikipedia": 40%,      # 일반 지식
    "arxiv_abstracts": 20%, # 학술 텍스트
    "news_articles": 20%,   # 뉴스 스타일
    "fiction_excerpts": 10%, # 소설/서사
    "technical_docs": 10%   # 기술 문서
}
```

**반복 금지 규칙**:
- 동일 문장 2회 이상 등장 불가
- 연속 3문장 이상 같은 주제 금지
- Sentence-level de-duplication 적용

---

## 4. 기술별 구현 계획

### 4.1 RAG (Retrieval-Augmented Generation)

#### Architecture
```
Query → [Embedding] → [Vector Search] → Top-K Chunks → [LLM Generate]
                           ↓
                      ChromaDB
```

#### Implementation Details

| 구성요소 | 선택 | 근거 |
|---------|------|------|
| **Vector DB** | ChromaDB | 로컬 설치, 경량, 충분한 성능 |
| **Embedding** | `BAAI/bge-m3` | 다국어 지원, 의미론적 검색 강점 |
| **Chunking** | RecursiveCharacterTextSplitter | LangChain 표준 |
| **Chunk Size** | 512 tokens | 세부 정보 보존 vs 문맥 유지 균형 |
| **Overlap** | 50 tokens | 경계 정보 손실 방지 |
| **Top-K** | 3, 5, 10 (비교) | K에 따른 정확도 변화 분석 |

#### NoLiMa 대응 전략: HyDE (Hypothetical Document Embeddings)
```python
def hyde_search(query: str, llm, vectordb) -> list:
    """
    NoLiMa에서 어휘 분리로 인한 검색 실패 완화
    """
    # Step 1: 가상의 답변 문서 생성
    hypothetical = llm.generate(
        f"Write a short passage that answers: {query}"
    )

    # Step 2: 가상 답변으로 검색 (더 풍부한 의미 벡터)
    results = vectordb.similarity_search(hypothetical, k=5)

    return results
```

#### 평가 코드
```python
# experiments/eval_rag.py
def evaluate_rag(samples, vectordb, llm, k=5):
    results = []
    for sample in tqdm(samples):
        # 1. Indexing (per-sample, 실제 사용 시나리오)
        vectordb.add_documents(chunk(sample['context']))

        # 2. Retrieval
        chunks = vectordb.similarity_search(sample['question'], k=k)

        # 3. Generation
        prompt = f"""Context: {' '.join(chunks)}
Question: {sample['question']}
Answer:"""
        answer = llm.generate(prompt)

        # 4. Evaluation
        results.append({
            'correct': nolima_score(answer, sample['answer']),
            'retrieval_hit': needle_in_chunks(chunks, sample['needle_phrase'])
        })

        vectordb.clear()  # Reset for next sample

    return results
```

---

### 4.2 LoRA Fine-tuning (Context Injection)

#### Architecture
```
Context → [Tokenize] → [QLoRA Train] → Adapted Model → [Generate Answer]
```

#### 접근 방식: Test-Time Training (TTT)

핵심 아이디어: 긴 문맥을 "학습 데이터"로 취급하여 LoRA 어댑터를 일시적으로 학습

```python
def lora_context_injection(context: str, question: str, base_model):
    """
    문맥을 학습하고 질문에 답변
    """
    # 1. 문맥을 학습 데이터로 변환
    train_data = [
        {"input": "다음 내용을 기억하세요:", "output": context},
        {"input": f"Q: {context[:100]}에 대해 설명하세요", "output": context[100:500]},
        # 자기 참조 학습으로 문맥 주입
    ]

    # 2. LoRA 어댑터 학습
    lora_config = LoraConfig(
        r=8,  # Low rank (메모리 절약)
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )

    model = get_peft_model(base_model, lora_config)

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        args=TrainingArguments(
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_checkpointing=True,  # VRAM 절약
            learning_rate=1e-4,
        )
    )
    trainer.train()

    # 3. 질문에 답변
    answer = model.generate(f"Q: {question}\nA:")

    # 4. 어댑터 해제 (다음 샘플용)
    model.unload_adapter()

    return answer
```

#### Hyperparameters

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| **LoRA Rank (r)** | 8 | 메모리 효율 vs 표현력 균형 |
| **Alpha** | 16 | r의 2배 (일반적 설정) |
| **Target Modules** | q_proj, v_proj | Attention에 집중 |
| **Epochs** | 3-5 | 과적합 유도 (의도적) |
| **Batch Size** | 1 | 24GB 제약 |
| **LR** | 1e-4 | 빠른 수렴 |

#### 리스크: 시간 비용

```
예상 시간 (per sample):
- Tokenization: ~1초
- Training (3 epochs): ~30-60초
- Inference: ~5초
- Total: ~40-70초/샘플

200 샘플 기준: ~2-4시간
```

**완화 전략**:
- 대표 샘플 20개만 Full Test
- 나머지는 통계적 추정 (부트스트랩)

---

### 4.3 Gist Token (Token Compression)

#### Architecture
```
Context → [Forward Pass with Gist Tokens] → Compressed KV Cache → [Generate Answer]
```

#### 기존 구현 활용

현재 프로젝트의 `src/model/gist_collator.py` 및 `experiments/train_gist_model.py` 활용

#### 실험 변수

| 변수 | 값 범위 | 목표 |
|------|--------|------|
| **Gist Token 수** | 25, 50, 100 | 압축률 vs 정확도 trade-off |
| **문맥 길이** | 4K, 8K, 16K | 확장성 검증 |

#### 압축률 계산

```python
def compression_ratio(context_tokens: int, gist_tokens: int) -> float:
    return context_tokens / gist_tokens

# 예시:
# 8K tokens / 50 gist = 160x compression
# 16K tokens / 100 gist = 160x compression
```

#### 평가 코드
```python
# experiments/eval_gist_nolima.py
def evaluate_gist(samples, model, tokenizer, num_gist=50):
    results = []

    for sample in tqdm(samples):
        # 1. Compress context
        compressed_kv = compress_context(
            model, tokenizer,
            sample['context'],
            num_gist_tokens=num_gist
        )

        # 2. Generate with compressed KV
        answer = generate_with_kv(
            model, tokenizer,
            compressed_kv,
            sample['question']
        )

        # 3. Evaluate
        results.append({
            'correct': nolima_score(answer, sample['answer']),
            'compression_ratio': len(tokenizer.encode(sample['context'])) / num_gist
        })

    return results
```

---

## 5. 실험 단계별 일정

### Timeline (7일)

```
Day 1-2: Phase 1 - 데이터셋 생성
├── NoLiMa 생성 스크립트 작성
├── 규칙 기반 50개 생성
├── GPT-4o로 150개 생성
└── 품질 검증 (Overlap < 10%, TTR > 0.3)

Day 3: Phase 2 - RAG 구현 및 평가
├── ChromaDB + BGE-M3 설정
├── Chunking 파이프라인
├── HyDE 구현
└── RAG 평가 실행

Day 4-5: Phase 3 - LoRA & Gist 구현
├── LoRA Context Injection 스크립트
├── Gist Token 재학습 (50, 100 토큰)
├── 각 기술 평가 실행
└── 중간 결과 확인

Day 6: Phase 4 - 통합 평가
├── 3가지 기술 일괄 비교
├── 메모리/속도 측정
├── 통계 분석
└── 그래프 생성

Day 7: Phase 5 - 보고서 작성
├── 결과 분석
├── 인사이트 도출
├── 최종 권고사항
└── 문서화 완료
```

### Deliverables

| Phase | 산출물 | 파일 경로 |
|-------|--------|----------|
| 1 | NoLiMa 데이터셋 | `data/nolima/nolima_200.jsonl` |
| 2 | RAG 결과 | `results/rag_nolima.json` |
| 3 | LoRA 결과 | `results/lora_nolima.json` |
| 3 | Gist 결과 | `results/gist_nolima.json` |
| 4 | 비교 분석 | `results/comparison_nolima.csv` |
| 5 | 최종 보고서 | `reports/FINAL_POC_REPORT.md` |

---

## 6. 리스크 및 완화 전략

### 6.1 기술적 리스크

| 리스크 | 확률 | 영향 | 완화 전략 |
|--------|------|------|----------|
| **VRAM OOM** | 높음 | 높음 | 4-bit quantization 필수, batch=1, gradient checkpointing |
| **LoRA 학습 시간 과다** | 중간 | 중간 | 대표 샘플 20개만 full test, 나머지 통계 추정 |
| **RAG 검색 실패** | 높음 | 중간 | HyDE 적용, 다양한 K 값 테스트 |
| **Gist 정보 손실** | 중간 | 중간 | 토큰 수 증가 (50→100), 압축률 조정 |

### 6.2 데이터 리스크

| 리스크 | 확률 | 영향 | 완화 전략 |
|--------|------|------|----------|
| **NoLiMa 너무 어려움** | 중간 | 높음 | Easy mode 20% 포함 (Overlap 허용) |
| **GPT-4 생성 품질 저하** | 낮음 | 중간 | 자동 검증 + 수동 샘플링 검토 |
| **도메인 편향** | 낮음 | 낮음 | 5개 도메인 균등 분배 |

### 6.3 일정 리스크

| 리스크 | 확률 | 영향 | 완화 전략 |
|--------|------|------|----------|
| **예상보다 긴 학습 시간** | 중간 | 중간 | 백그라운드 실행, 병렬 처리 |
| **디버깅 지연** | 중간 | 중간 | 각 단계 체크포인트, 로깅 강화 |

---

## 7. 성공 기준

### 7.1 필수 달성 (Must Have)

| 기준 | 설명 | 검증 방법 |
|------|------|----------|
| **재현성** | 단일 스크립트로 전체 실험 재현 가능 | `./run_experiment.sh` 실행 |
| **변별력** | 3가지 기술 간 통계적 유의미한 차이 | p-value < 0.05 |
| **문서화** | 모든 결과 및 코드 문서화 | README, 보고서 |

### 7.2 목표 달성 (Should Have)

| 기준 | 목표 | 현실적 예상 |
|------|------|------------|
| **최고 정확도** | 60% 이상 | LoRA: 70-80%, Gist: 50-60%, RAG: 40-50% |
| **메모리 효율** | 24GB 내 모든 실험 | 32K 문맥 일부 제외 가능 |
| **시간 효율** | 7일 내 완료 | 가능 |

### 7.3 추가 성과 (Nice to Have)

- 새로운 하이브리드 접근법 발견 (RAG + Gist)
- 표준 벤치마크 (LongBench) 결과 추가
- 논문/블로그 발행 가능한 수준의 분석

---

## 8. Claude 추가 분석

### 8.1 Gemini 기획서 검토

Gemini가 제시한 기획서는 전체적으로 우수하나, 몇 가지 보완이 필요합니다:

#### 강점
- ✅ 명확한 가설 설정 (RAG < Gist < LoRA)
- ✅ 현실적인 일정 (7일)
- ✅ 리스크 관리 계획

#### 보완 필요
- ⚠️ LoRA "Test-Time Training"은 실제로 매우 느림 (샘플당 1분+)
- ⚠️ HyDE가 항상 효과적이지 않을 수 있음
- ⚠️ 32K 문맥은 RTX 4090에서 OOM 위험 높음

### 8.2 현실적 조정

#### LoRA 전략 수정
```
원래: 200 샘플 모두 개별 학습
수정:
  - 10개 샘플: Full LoRA 학습 (정확도 검증)
  - 190개 샘플: 학습 없이 Base 모델로 평가 (베이스라인)

이유: 200샘플 × 1분 = 3시간+ 소요
```

#### 문맥 길이 조정
```
원래: 4K, 8K, 16K, 32K
수정:
  - 4K: 50개 (모든 기술)
  - 8K: 100개 (모든 기술)
  - 16K: 50개 (Gist, RAG만)

이유: 32K는 LoRA 학습 시 OOM, 16K도 위험
```

### 8.3 추가 실험 제안

#### Experiment A: Hybrid RAG + Gist
```python
def hybrid_approach(context, question):
    """
    1단계: Gist로 전역 문맥 압축 (요약)
    2단계: RAG로 세부 정보 검색
    3단계: 둘을 결합하여 답변 생성
    """
    gist_summary = compress_to_gist(context, n=50)
    rag_chunks = retrieve_chunks(context, question, k=3)

    prompt = f"""
Global Context (Gist Summary): {gist_summary}
Detailed Context (RAG): {rag_chunks}
Question: {question}
Answer:
"""
    return generate(prompt)
```

#### Experiment B: Iterative Gist Refinement
```python
def iterative_gist(context, question, iterations=3):
    """
    질문을 고려하여 Gist를 반복적으로 정제
    """
    for i in range(iterations):
        gist = compress_with_question_guidance(context, question, n=50)
        if evaluate_gist_quality(gist, question) > threshold:
            break

    return generate_with_gist(gist, question)
```

### 8.4 예상 결과 수정

Gemini 예측 대비 현실적 조정:

| 기술 | Gemini 예측 | Claude 수정 | 근거 |
|------|------------|------------|------|
| **LoRA** | 75-85% | 60-70% | Test-Time Training의 과적합이 NoLiMa에서는 오히려 불리 |
| **Gist** | 50-65% | 40-55% | 현재 34.5%에서 데이터 개선으로 상승 예상 |
| **RAG** | 35-50% | 25-40% | HyDE도 NoLiMa에서는 한계 |
| **Hybrid** | - | 55-70% | 새로운 접근법, 검증 필요 |

---

## 9. 실행 명령어 요약

### Phase 1: 데이터셋 생성
```bash
# NoLiMa 스타일 데이터셋 생성
python scripts/generate_nolima_hybrid.py \
    --rule-based 50 \
    --llm-generated 150 \
    --output data/nolima/nolima_200.jsonl

# 품질 검증
python scripts/validate_nolima.py \
    --input data/nolima/nolima_200.jsonl \
    --max-overlap 0.1 \
    --min-ttr 0.3
```

### Phase 2-4: 실험 실행
```bash
# 전체 실험 실행 (단일 스크립트)
./run_experiment.sh

# 또는 개별 실행
python experiments/eval_rag_nolima.py
python experiments/eval_lora_nolima.py
python experiments/eval_gist_nolima.py
```

### Phase 5: 결과 분석
```bash
# 비교 분석 및 보고서 생성
python scripts/analyze_results.py \
    --rag results/rag_nolima.json \
    --lora results/lora_nolima.json \
    --gist results/gist_nolima.json \
    --output reports/FINAL_POC_REPORT.md
```

---

## 10. 결론

본 PoC는 NoLiMa라는 **엄격한 평가 기준**을 적용하여 3가지 장문맥 처리 기술의 **진정한 이해 능력**을 검증합니다.

### 핵심 차별화
- ✅ 기존 NIAH의 문제점 (Ctrl+F 해결 가능) 극복
- ✅ 어휘 분리로 인한 **의미적 추론** 필수화
- ✅ RTX 4090 환경에서 **실행 가능한** 현실적 계획

### 예상 결론
- **LoRA**: 정확도 최고, 하지만 비용 비효율
- **Gist Token**: 압축 손실 존재, 하지만 전역 문맥 유지로 경쟁력
- **RAG**: NoLiMa에서 가장 취약, 검색 기반의 근본적 한계

### 최종 목표
**100/100 달성을 위한 로드맵**을 제시하고, 실제 배포 가능한 최적의 기술 조합을 발견합니다.

---

**다음 단계**: NoLiMa 데이터셋 생성 스크립트 작성 (`scripts/generate_nolima_hybrid.py`)
