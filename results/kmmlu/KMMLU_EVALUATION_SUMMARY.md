# KMMLU 평가 결과 요약

## 실험 개요

**목표**: Foundation 모델이 학습하지 못한 한국어 전문 지식(법률)에 대해 RAG, LoRA, Gist Token의 실제 학습 효과 비교

**데이터셋**: KMMLU (Korean MMLU)
- Law: 1,000 test / 1,297 train
- Criminal-Law: 200 test / 85 train

**기준 모델**: Llama-3-8B-Instruct (4-bit/8-bit 양자화)

## 최종 실험 결과 (100 샘플 기준 - Law only)

| 방법 | Law (100) | vs Baseline | 비고 |
|------|-----------|-------------|------|
| **Baseline (Zero-shot)** | 25.00% | - | Foundation 모델 한계 |
| **RAG (BGE-M3, Top-3)** | 31.00% | **+6.00%p** | 지식 검색 효과 확인 ✅ |
| **LoRA (2 epochs)** | 24.00% | -1.00%p | 과적합, 개선 없음 |
| **Gist Token (25 tokens)** | 23.00% | -2.00%p | KMMLU에서는 효과 없음 |
| GPT-4 (Reference) | - | - | 59.95% |

### 핵심 결론

**RAG만이 유일하게 Baseline 대비 성능 향상** (+6.00%p)
- LoRA와 Gist Token은 KMMLU 법률 평가에서 효과가 없거나 오히려 하락

## 초기 실험 결과 (30 샘플 기준 - 참고용)

| 방법 | Law | Criminal-Law | Overall | vs Baseline |
|------|-----|--------------|---------|-------------|
| **Baseline (Zero-shot)** | 23.33% | 30.00% | 26.67% | - |
| **RAG (BGE-M3, Top-3)** | 46.67% | 16.67% | 31.67% | +5.00%p |
| **LoRA (2 epochs)** | 13.33% | 36.67% | 25.00% | -1.67%p |

⚠️ 30 샘플 결과는 통계적 변동이 큼 (1문제 = 3.3%p)

## 상세 분석

### 1. Baseline (Zero-shot)
- **Law**: 23.33% (7/30)
- **Criminal-Law**: 30.00% (9/30)
- **Overall**: 26.67% (16/60)
- **분석**: Foundation 모델의 한국어 법률 지식이 매우 부족함

### 2. RAG (Retrieval Augmented Generation)
- **Law**: 46.67% (14/30) - **+23.34%p 향상**
- **Criminal-Law**: 16.67% (5/30) - **-13.33%p 하락**
- **Overall**: 31.67% (19/60)
- **분석**:
  - Law 분야에서 큰 향상 (지식 베이스에 Law train 데이터 1,297개 존재)
  - Criminal-Law 하락은 지식 베이스 불균형 (85개만 존재)으로 인한 검색 노이즈

### 3. LoRA Fine-tuning
- **Law**: 13.33% (4/30) - **-10.00%p 하락**
- **Criminal-Law**: 36.67% (11/30) - **+6.67%p 향상**
- **Overall**: 25.00% (15/60)
- **분석**:
  - 학습 데이터 패턴에 과적합
  - Loss 2.63 → 1.31로 감소했으나 일반화 실패
  - 학습 데이터와 테스트 데이터의 분포 차이

### 4. Gist Token (100 샘플 평가 완료)
- **Law (100)**: 23.00% (23/100) - **-2.00%p 하락**
- **분석**:
  - NIAH 평가(34.5%)에서는 효과가 있었으나 KMMLU에서는 성능 하락
  - Gist Token은 "needle 추출"에는 적합하나 "지식 이해/추론"에는 부적합
  - 25개 토큰으로 복잡한 법률 지식을 압축하기에는 한계

## 핵심 인사이트

### 1. RAG의 강점과 한계
- **강점**: 관련 지식을 직접 제공하여 즉시 성능 향상 (+6.00%p)
- **한계**: 검색 품질에 종속, 지식 베이스 불균형 시 노이즈 발생

### 2. LoRA의 과적합 문제
- 1,382개 학습 데이터로는 일반화 불충분
- Multiple-choice 형식에 특화된 패턴 학습만 발생
- **결론**: 소량 데이터 Fine-tuning은 전문 지식 습득에 비효과적

### 3. Gist Token의 한계
- **NIAH vs KMMLU 차이**: 34.5% (NIAH) vs 23.00% (KMMLU)
- **분석**: Gist는 "특정 정보 추출"에는 효과적이나 "복합적 추론"에는 한계
- **결론**: Gist Token은 코드/패스키 추출 등 단순 압축에 적합, 전문 지식 Q&A에는 부적합

### 4. 방법론별 적합 용도
| 방법 | 적합한 태스크 | 부적합한 태스크 |
|------|--------------|----------------|
| RAG | 지식 Q&A, 문서 검색 | 실시간 처리, 오프라인 |
| LoRA | 대규모 데이터 학습 | 소량 데이터 학습 |
| Gist | 정보 추출, 압축 | 복합 추론, 전문 지식 |

## 한계점

1. **샘플 수**: 100 샘플 (통계적 유의성 개선, 그러나 전체 1,000샘플 중 10%)
2. **단일 도메인**: Law만 평가 (Criminal-Law 제외)
3. **LoRA 하이퍼파라미터**: 최적화되지 않은 설정 (2 epochs, lr=2e-4)
4. **Gist 평가 방식**: 단순화된 방식 (KV Cache 압축 대신 토큰 직접 주입)

## 결론

**전문 지식 습득에 RAG가 가장 효과적**
- RAG: +6.00%p (유일한 개선)
- LoRA: -1.00%p (과적합)
- Gist: -2.00%p (압축 한계)

Gist Token은 원래 목표인 "긴 문맥 압축"에는 효과적이나,
"전문 지식 Q&A"에는 RAG가 더 적합함을 확인.

## 파일 위치

### 100 샘플 결과 (최종)
- Baseline: `results/kmmlu/baseline_law_100.json`
- RAG: `results/kmmlu/rag_law_100.json`
- LoRA: `results/kmmlu/lora_law_100.json`
- Gist: `results/kmmlu/gist_law_100.json`

### 30 샘플 결과 (참고용)
- Baseline: `results/kmmlu/baseline_law_30.json`
- RAG: `results/kmmlu/rag_combined_30.json`
- LoRA: `results/kmmlu/lora_law_30.json`
- Gist: `results/kmmlu/gist_law_30.json`

### 데이터
- 지식 베이스: `data/kmmlu/law_criminal_knowledge.jsonl` (1,382 문서)
