# Gist Token PoC 실험 설계 요약

## 1. 실험 목표

**핵심 가설**: Gist Token 기반 문맥 압축이 가정용 GPU(24GB VRAM) 환경에서 RAG 및 Full Context 방식 대비 메모리 효율성과 추론 속도를 획기적으로 개선하면서도, 글로벌 문맥 이해 성능을 유지할 수 있다.

**검증 항목**:
1. **메모리 절감**: KV Cache 메모리 사용량 100배 이상 감소 (4000 토큰 → 40 Gist 토큰)
2. **속도 향상**: RAG 대비 TTFT(Time To First Token) 50% 이상 단축
3. **품질 유지**: 글로벌 문맥 이해 질문에서 Full Context 대비 80% 이상 성능
4. **압축 한계**: Passkey Retrieval 정확도로 정보 손실 정도 측정

## 2. 실험 그룹 설정

### Baseline 1: Full Context (Upper Bound)
- **방식**: Llama-3-8B에 전체 문맥(2k-8k 토큰)을 직접 입력
- **장점**: 정보 손실 없음, 최고 품질
- **단점**: 느린 추론 속도, 높은 메모리 사용, 긴 문맥에서 OOM 발생 가능

### Baseline 2: RAG (ChromaDB)
- **방식**: 문서를 500 토큰 청크로 분할, 벡터 검색 후 Top-3 청크 사용
- **장점**: 메모리 효율적, 실시간 DB 업데이트 가능
- **단점**: 검색 지연, 문맥 파편화, 글로벌 이해 취약

### Experimental: Gist Token
- **방식**: 문서를 N개(10, 25, 50) Gist Token으로 압축 후 KV Cache 저장
- **변수**: Gist 토큰 개수 (10, 25, 50) → 압축률 vs 정보 보존 trade-off
- **목표**: 메모리/속도는 RAG 수준, 품질은 Full Context에 근접

## 3. 데이터셋 구성

### 3.1 소스 데이터
- **개인화 데이터 시뮬레이션**: Wikipedia 긴 문서, arXiv 논문 초록, 기술 문서
- **크기**: 문서당 2,000 ~ 8,000 토큰 (Llama-3 토크나이저 기준)
- **분량**: 학습 200개, 검증 50개, 테스트 50개

### 3.2 합성 Q&A 생성 (GPT-4 / Llama-3-70B)
각 문서당 3가지 유형의 질문 생성:

1. **세부 정보 조회 (Factual Lookup)**: "이 문서에서 언급된 X의 정의는?"
2. **글로벌 주제 (Global Theme)**: "이 문서의 전체적인 논지는 무엇인가?"
3. **추론 (Reasoning)**: "A와 B의 차이점은 무엇이며, 왜 중요한가?"

### 3.3 Passkey 삽입 (정보 손실 측정)
- 문서 중간에 랜덤 5자리 숫자 삽입
- "문서에 숨겨진 Passkey는?"이라는 질문으로 정확한 회수 여부 확인
- Gist의 손실 압축(Lossy Compression) 정도를 정량화

### 3.4 데이터 포맷 (JSONL)
```json
{
  "instruction": "다음 텍스트를 Gist Token으로 압축한 뒤, 질문에 답하시오.",
  "context": "...(2000~8000 토큰)...",
  "gist_tokens": "<GIST_0><GIST_1>...<GIST_9>",
  "question": "이 논문의 핵심 기여는 무엇인가?",
  "answer": "...",
  "gist_start_idx": 1523,
  "gist_end_idx": 1533
}
```

## 4. 학습 설정

### 모델 구성
- **Base Model**: meta-llama/Meta-Llama-3-8B-Instruct
- **Quantization**: 4-bit NF4 (bitsandbytes) → 모델 메모리 ~5GB
- **LoRA Config**:
  - r=16, alpha=32, dropout=0.05
  - target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  - **modules_to_save**: ["embed_tokens", "lm_head"] ← 필수!

### Gist Token 설정
- **변수 실험**: 10개, 25개, 50개 (3가지 실험)
- **토큰 ID**: `<GIST_0>` ~ `<GIST_N>`

### 학습 하이퍼파라미터
- **Learning Rate**: 1e-4 (LoRA), 5e-5 (embed_tokens)
- **Warmup Steps**: 100 (임베딩 발산 방지)
- **Batch Size**: 1-2 (24GB VRAM 제약)
- **Gradient Accumulation**: 8 (effective batch size = 8-16)
- **Epochs**: 3-5
- **Mixed Precision**: BF16

### 핵심 커스텀 로직
- **GistDataCollator**: Attention mask 조작
  - `mask[i, :, gist_end:, :gist_start] = False`
  - Question이 Context를 볼 수 없도록 차단

## 5. 평가 지표

### 5.1 정량적 지표

| 지표 | 측정 방법 | 목표 |
|------|----------|------|
| **압축률** | 원본 토큰 수 / Gist 토큰 수 | 100x ~ 400x |
| **VRAM 사용량** | `torch.cuda.max_memory_allocated()` | RAG 대비 동등 또는 낮음 |
| **TTFT** | 첫 토큰 생성까지 시간 (ms) | RAG 대비 50% 단축 |
| **처리량** | tokens/sec | Full Context 대비 2배 이상 |
| **Passkey Accuracy** | 정확히 회수한 비율 (%) | 70% 이상 (손실 압축 감안) |

### 5.2 정성적 지표

| 질문 유형 | 평가 방법 | 성공 기준 |
|----------|-----------|----------|
| **Factual Lookup** | Exact Match / F1 Score | Full Context 대비 85%+ |
| **Global Theme** | LLM-as-Judge (GPT-4 평가) | "Coherent & Complete" 판정 80%+ |
| **Reasoning** | Human Evaluation | 논리적 일관성 3점 척도에서 평균 2.5+ |

### 5.3 비교 매트릭스 (예상)

| 방식 | VRAM (GB) | TTFT (ms) | Global 정확도 | Passkey 정확도 |
|------|-----------|-----------|---------------|----------------|
| Full Context | 18-22 | 800-1200 | 95% | 100% |
| RAG (Top-3) | 8-10 | 400-600 | 60% | 75% |
| **Gist-10** | **8-9** | **200-300** | **85%** | **70%** |
| **Gist-25** | **9-10** | **250-350** | **90%** | **80%** |
| **Gist-50** | **10-12** | **300-400** | **92%** | **85%** |

## 6. 실험 단계별 계획

### Phase 1: 환경 및 데이터 준비 (1-2일)
- [ ] 가상환경 구축, 의존성 설치
- [ ] Llama-3-8B 4-bit 양자화 로딩 검증
- [ ] Wikipedia/arXiv 데이터 수집 및 청킹 (2k-8k 토큰)
- [ ] GPT-4 API로 Q&A 쌍 생성 (문서당 3개)
- [ ] Passkey 삽입 및 JSONL 포맷 변환

### Phase 2: Gist Token 구현 (2-3일)
- [ ] Gist Token 추가 및 임베딩 레이어 확장
- [ ] `GistDataCollator` 구현 (attention mask 로직)
- [ ] Attention mask 시각화 검증 (matplotlib)
- [ ] LoRA config 설정 (`modules_to_save` 포함)

### Phase 3: 학습 실행 (각 2-4시간 × 3회)
- [ ] **Experiment 1**: 10 Gist Tokens
- [ ] **Experiment 2**: 25 Gist Tokens
- [ ] **Experiment 3**: 50 Gist Tokens
- [ ] 각 실험마다 체크포인트 저장 및 로그 기록

### Phase 4: Baseline 구축 (1-2일)
- [ ] Full Context 추론 파이프라인
- [ ] RAG 시스템 구축 (ChromaDB + sentence-transformers)
- [ ] 동일 테스트셋으로 성능 측정

### Phase 5: 평가 및 분석 (1-2일)
- [ ] Passkey Retrieval 테스트 실행
- [ ] 정량 지표 수집 (VRAM, TTFT, Throughput)
- [ ] GPT-4 LLM-as-Judge 평가 (Global Theme 질문)
- [ ] 결과 시각화 (비교 그래프, attention heatmap)

### Phase 6: KV Cache 압축 구현 (1일)
- [ ] `compress_context()` 함수 구현
- [ ] KV Cache 슬라이싱 검증
- [ ] 압축 전후 메모리 사용량 비교

## 7. 예상 실패 모드 및 대응

### 실패 1: 모델이 Gist Token을 무시
**증상**: 학습 loss는 낮지만, 검증 시 답변이 랜덤하거나 "모른다"고 응답
**원인**: Attention mask가 제대로 적용되지 않음
**대응**:
- Mask 시각화로 `mask[gist_end:, :gist_start]`가 False인지 확인
- Forward pass 중간에 attention_weights 출력해서 Context 참조 여부 체크

### 실패 2: Gist 임베딩 발산 (Gradient Explosion)
**증상**: Loss가 NaN, 임베딩 값이 ±1000 이상
**원인**: Learning rate가 너무 높거나 warmup 부족
**대응**:
- embed_tokens learning rate를 1e-5로 낮춤
- Warmup steps를 200으로 증가
- Gradient clipping 추가 (`max_grad_norm=1.0`)

### 실패 3: Passkey Retrieval 20% 미만 (심각한 정보 손실)
**증상**: Gist Token이 정보를 전혀 압축하지 못함
**원인**: Gist 개수가 너무 적거나, 학습 데이터 부족
**대응**:
- Gist Token 개수를 50개로 증가
- 학습 데이터를 300개로 확대
- Multi-task learning: Passkey 질문을 명시적으로 학습 데이터에 추가

### 실패 4: OOM (Out of Memory)
**증상**: Batch size 1에서도 메모리 부족
**원인**: 긴 시퀀스(8k 토큰) + Gradient 저장
**대응**:
- Gradient checkpointing 활성화: `model.gradient_checkpointing_enable()`
- 최대 시퀀스 길이를 4k로 제한
- Optimizer를 8-bit Adam으로 변경 (`bitsandbytes.optim.Adam8bit`)

## 8. 성공 기준

본 PoC가 성공했다고 판단하는 기준:

1. ✅ **메모리**: Gist-25 방식이 RAG와 동등한 VRAM 사용 (10GB 이하)
2. ✅ **속도**: TTFT가 RAG 대비 30% 이상 빠름 (400ms → 280ms)
3. ✅ **품질**: Global Theme 질문에서 RAG(60%) 대비 20%p 이상 향상 (80%+)
4. ✅ **압축 검증**: Passkey Accuracy 70% 이상 (정보가 실제로 압축됨을 증명)

**추가 인사이트 목표**:
- Gist Token 개수와 성능의 관계 그래프 도출
- 어떤 유형의 질문에서 Gist가 특히 강점/약점을 보이는지 분석
- RAG + Gist 하이브리드 가능성 탐색

## 9. 타임라인 요약

| 단계 | 소요 시간 | 누적 |
|------|-----------|------|
| Phase 1: 데이터 준비 | 1-2일 | 2일 |
| Phase 2: 구현 | 2-3일 | 5일 |
| Phase 3: 학습 (3회) | 0.5일 | 5.5일 |
| Phase 4: Baseline | 1-2일 | 7.5일 |
| Phase 5: 평가 | 1-2일 | 9.5일 |
| Phase 6: KV 압축 | 1일 | 10.5일 |

**총 예상 기간**: 약 2주 (실제 GPU 학습 시간 제외)

## 10. 다음 단계 (PoC 이후)

성공 시:
- Hierarchical Gist: 책 전체를 다단계 압축
- Multimodal Gist: 이미지+텍스트 통합 압축
- Production 배포: vLLM 통합, 멀티 유저 캐싱 시스템

실패 시:
- ICAE, AutoCompressor 등 대안 기술 탐색
- Gist + RAG 하이브리드 우선 구현
- LLMLingua와의 정량 비교 논문 작성
