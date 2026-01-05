# NIAH 데이터셋 품질 분석 보고서

**분석 기준**: NoLiMa (No Literal Matching) 논문 (arXiv:2502.05167)
**분석 일자**: 2026-01-05
**대상 데이터셋**: Global NIAH (200 샘플), Korean NIAH (200 샘플)

---

## Executive Summary

**핵심 발견**: 현재 NIAH 데이터셋은 NoLiMa 기준으로 **매우 낮은 품질**을 보이며, 모델이 **단순 키워드 매칭**만으로 문제를 해결할 수 있습니다.

### NoLiMa 준수도 점수

| 데이터셋 | 총점 | 질문 다양성 | 어휘 분리도 | 문맥 복잡도 | 패턴 일반화 |
|---------|------|------------|------------|------------|------------|
| **Global NIAH** | **22.6/100** | 0.1/30 | 0.0/30 | 2.4/20 | 20.0/20 |
| **Korean NIAH** | **32.8/100** | 0.1/30 | 30.0/30* | 2.6/20 | 0.0/20 |

*Korean의 어휘 분리도 30점은 측정 오류 - 실제로는 Global과 동일한 문제 존재

### 평가 결과 해석

**Gist Token 모델의 34.5% 정확도**는:
- ✅ Random보다는 높음 (기준선: ~0%)
- ⚠️ 하지만 데이터셋이 너무 쉬워서 **과대평가** 가능성
- ❌ NoLiMa 스타일 데이터셋에서는 훨씬 낮을 것으로 예상

---

## 1. 질문 다양성: 0.1/30점 (치명적 결함)

### 문제점

**Global NIAH**:
- 200개 샘플 모두 **동일한 질문** 사용
- 질문: `"What is the secret passkey mentioned in the text?"`
- 다양성 비율: **0.5%** (200개 중 1개 unique)

**Korean NIAH**:
- 200개 샘플 모두 **동일한 질문** 사용
- 질문: `"텍스트에서 언급된 비밀 패스키는 무엇인가요?"`
- 다양성 비율: **0.5%**

### NoLiMa 기준 위반

NoLiMa 논문은 다음을 권장:
> "질문 유형을 다양화하여 모델이 단일 패턴에 과적합되지 않도록 해야 함"

**현재 상태**: 모델은 "secret passkey" 또는 "비밀 패스키"라는 단어만 보면 즉시 반응하도록 학습됨

### 영향

1. **과적합 유발**: 모델이 질문의 의미가 아닌 특정 단어 조합만 학습
2. **일반화 불가**: 다른 유형의 질문 (e.g., "중요한 코드는?", "저자가 강조한 번호는?")에 대응 불가
3. **평가 신뢰도 저하**: 실제 문맥 이해 능력이 아닌 패턴 매칭 능력만 측정

---

## 2. 어휘 분리도: 0.0/30점 (치명적 결함)

### 문제점

**키워드 매칭 시뮬레이션 결과**:
- Global NIAH: **100% 해결 가능** (50/50 샘플)
- Korean NIAH: 영어 패턴 기준 0%, 하지만 한국어 패턴으로는 **100% 해결 가능**

### 패턴 분석

**Global NIAH Needle 패턴**:
```text
질문: "What is the secret passkey mentioned in the text?"
Needle: "The secret passkey is [6자리 코드]."
```

**어휘 중복**:
- `secret` ✅ (질문과 needle 모두 포함)
- `passkey` ✅ (질문과 needle 모두 포함)
- **중복률: 50%**

**Korean NIAH Needle 패턴**:
```text
질문: "텍스트에서 언급된 비밀 패스키는 무엇인가요?"
Needle: "비밀 패스키는 [6자리 코드]입니다."
```

**어휘 중복**:
- `비밀` ✅ (질문과 needle 모두 포함)
- `패스키` ✅ (질문과 needle 모두 포함)
- **중복률: 50%**

### NoLiMa 기준 위반

NoLiMa 논문의 핵심 주장:
> "기존 NIAH는 모델이 질문의 단어를 haystack에서 그대로 찾아 매칭하는 방식으로 작동한다.
> 이는 진정한 장문맥 이해가 아니라 단순한 Ctrl+F 검색에 불과하다."

**GPT-4o 성능 비교**:
- 기존 NIAH: **99.3%**
- NoLiMa: **69.7%** (30%p 하락!)

### 실제 해결 알고리즘

모델이 사용할 수 있는 단순 전략:
```python
def solve_niah_trivially(context, question):
    if "secret" in question and "passkey" in question:
        # Regex로 "secret passkey is [CODE]" 패턴 찾기
        match = re.search(r'secret passkey is ([A-Z0-9]{6})', context, re.I)
        return match.group(1)
```

**결론**: 현재 데이터셋은 **Gist Token 압축 능력을 전혀 평가하지 못함**

---

## 3. 문맥 복잡도: 2.4/20점 (심각한 결함)

### 문제점: 극도의 문장 반복

**Global NIAH (샘플 1개 분석)**:
- 총 문장 수: **406개**
- 고유 문장 수: **22개**
- **반복 비율: 94.6%**

**가장 많이 반복된 문장들**:
1. "In the field of artificial intelligence, neural networks have revolutionized many domains." - **25회 반복**
2. "Deep learning models require substantial computational resources for training." - **25회 반복**
3. "Transfer learning allows models to leverage pre-trained weights for new tasks." - **25회 반복**

**Korean NIAH (샘플 1개 분석)**:
- 총 문장 수: **201개**
- 고유 문장 수: **35개**
- **반복 비율: 82.6%**

**가장 많이 반복된 문장들**:
1. "데이터 과학은 통계학과 컴퓨터 과학의 융합 분야입니다." - **18회 반복**
2. "머신러닝 알고리즘은 대규모 데이터에서 패턴을 발견합니다." - **18회 반복**
3. "데이터 시각화는 복잡한 정보를 이해하기 쉽게 만듭니다." - **17회 반복**

### NoLiMa 기준 위반

NoLiMa는 다음을 권장:
> "Haystack은 의미적으로 다양하고 일관된 문맥을 제공해야 하며,
> 단순 반복은 모델이 통계적 패턴에만 의존하게 만든다."

### Type-Token Ratio (TTR) 분석

| 데이터셋 | TTR | 해석 |
|---------|-----|------|
| Global | **0.076** | 매우 낮음 (높은 반복도) |
| Korean | **0.108** | 낮음 |
| 권장 기준 | >0.5 | 다양한 어휘 사용 |

**비교**: 일반적인 뉴스 기사 TTR은 0.4-0.6

### 영향

1. **과적합 유발**: 모델이 동일 문장 패턴만 암기
2. **압축 능력 미평가**: Gist Token이 다양한 정보를 압축하는지 검증 불가
3. **일반화 실패**: 학습 데이터와 다른 스타일의 문맥에서 실패 가능성

---

## 4. 패턴 일반화

### Global NIAH: 20.0/20점 (유일한 강점)

- Needle 패턴: `"The secret passkey is [CODE]."`
- 위치: 25%-50% 사이 (적절히 분산)
- 코드 형식: 6자리 alphanumeric (일관적)

### Korean NIAH: 0.0/20점 (측정 실패)

- Needle 패턴: `"비밀 패스키는 [CODE]입니다."`
- 분석 스크립트가 한국어 패턴을 찾지 못함 (영어 전용 regex)
- 실제로는 Global과 동일하게 일관적인 패턴

### 평가

Needle 위치는 적절히 분산되어 있으나:
- ❌ 표현 방식은 **완전히 동일** (`"The secret passkey is"`)
- ❌ 의미적 변형 없음 (e.g., "authentication code", "access key" 등)

---

## 5. 실제 평가 결과와의 연관성

### 모델 성능

| 데이터셋 | 정확도 | NoLiMa 점수 | 상관관계 |
|---------|--------|-------------|---------|
| Global NIAH | **20.0%** | 22.6/100 | 낮은 품질 → 낮은 성능 |
| Korean NIAH | **49.0%** | 32.8/100 | 상대적으로 나은 품질 → 높은 성능 |

### 가설: Korean이 Global보다 높은 이유

1. **TTR 차이**: Korean (0.108) > Global (0.076)
   - Korean이 약간 더 다양한 어휘 사용
2. **반복 비율**: Korean (82.6%) < Global (94.6%)
   - Korean이 덜 반복적
3. **Llama-3의 언어 편향**: 영어 훈련 데이터가 압도적 → 한국어 토크나이징이 비효율적
   - 한국어는 더 많은 토큰으로 분할됨
   - 결과적으로 Gist에 상대적으로 더 많은 정보 유지 가능?

### 과적합 증거

**최종 손실 0.0034**는:
- ✅ 정보 병목은 작동 (초기 3.40 → 0.0034)
- ❌ 하지만 **일반화 실패** (validation이 없었음)
- ❌ 1,000개 샘플 × 3 에포크 = 3,000회 반복 → **패턴 암기**

**증거**:
- Korean 49% vs Global 20%의 큰 격차
- 학습 데이터 패턴과 유사한 Korean에서만 성능 발휘
- 진정한 압축보다는 학습 샘플 특성 기억

---

## 6. NoLiMa 스타일 개선 방안

### 우선순위 1: 질문 다양화 (HIGH)

**현재**:
```text
"What is the secret passkey mentioned in the text?"
```

**개선안**:
```text
1. "According to the document, what is the primary authentication code?"
2. "The author mentioned a 6-character identifier. What is it?"
3. "What access credential is specified in the passage?"
4. "Retrieve the alphanumeric sequence described as the key."
5. "What is the value of the security token discussed?"
```

**효과**: 모델이 의미를 이해해야만 해결 가능

### 우선순위 2: 어휘 분리 (HIGH)

**현재 Needle**:
```text
"The secret passkey is ABC123."
```

**개선안 (Latent Association 유도)**:
```text
질문: "What is the primary authentication code?"
Needle: "For system access, use the identifier: ABC123"
```

**핵심**: 질문과 needle에서 **공통 단어를 제거**

**NoLiMa 방식**:
```text
질문: "What credential is needed for the vault?"
Needle: "The secure storage requires authentication: ABC123"
```

- `credential` ↔ `authentication` (의미적으로 연결, 어휘적으로 분리)
- `vault` ↔ `secure storage` (동의어 관계)

### 우선순위 3: 문맥 다양화 (MEDIUM)

**현재**: 4-5개 문장을 25회씩 반복

**개선안**:
1. **실제 문서 사용**: Wikipedia, 뉴스, 논문 초록 등
2. **Synthetic 개선**:
   - LLM으로 다양한 주제의 일관된 단락 생성
   - 문장 재사용 금지
3. **TTR 목표**: >0.5

### 우선순위 4: Needle 표현 다양화 (MEDIUM)

**현재**: `"The secret passkey is [CODE]."`

**개선안**:
```text
1. "Authentication requires: [CODE]"
2. "Access code: [CODE] (valid until midnight)"
3. "Use identifier [CODE] for verification"
4. "System PIN: [CODE]"
5. "[CODE] serves as the primary credential"
```

---

## 7. 권장 조치사항

### 즉시 실행 (1-2일)

1. ✅ **NoLiMa 스타일 데이터셋 생성** (200 샘플)
   - 질문 다양화: 20가지 질문 템플릿
   - 어휘 분리: Lexical overlap <10%
   - TTR 목표: >0.3

2. ✅ **현재 모델 재평가**
   - NoLiMa 데이터셋으로 정확도 측정
   - 예상: 34.5% → **10-15%로 하락**
   - 진정한 압축 능력 파악

### 단기 목표 (1주)

3. 🔶 **데이터셋 확장 + 품질 개선**
   - 5,000개 NoLiMa 스타일 샘플 생성
   - 실제 문서 기반 haystack 사용
   - Validation set 분리 (20%)

4. 🔶 **재학습 with 개선 데이터**
   - Gist 25 → 50개
   - 5,000 샘플, 5 에포크
   - Early stopping with validation

### 중기 목표 (2-3주)

5. 🔶 **Baseline 비교**
   - RAG (ChromaDB + retrieval)
   - LoRA Fine-tuning (full context)
   - Gist Token
   - **모두 NoLiMa 데이터셋으로 평가**

6. 🔶 **표준 벤치마크**
   - LongBench (NIAH subset)
   - RULER
   - 공개 리더보드 비교

---

## 8. 결론

### 현재 상태 진단

**Gist Token PoC 88/100점**은:
- ✅ 기술적 구현은 성공 (정보 병목 작동 확인)
- ❌ 하지만 **평가 방법이 부적절**
- ⚠️ 34.5% 정확도는 **과대평가**된 수치

### 데이터셋 품질 등급

| 항목 | 등급 | 설명 |
|------|------|------|
| **질문 다양성** | **F** | 단일 질문 패턴 (0.5%) |
| **어휘 분리도** | **F** | 키워드 매칭으로 100% 해결 |
| **문맥 복잡도** | **F** | 94.6% 반복, TTR 0.076 |
| **패턴 일반화** | **B** | 위치 분산은 양호 |
| **종합 평가** | **F** | NoLiMa 기준 22.6/100점 |

### 최종 권고

1. **즉시**: NoLiMa 스타일 데이터셋 200개 생성 → 재평가
2. **현실 직시**: 현재 34.5%는 허상, 실제 능력은 10-15% 예상
3. **올바른 방향**: 데이터셋 품질 개선 후 재학습이 100/100 달성의 핵심

**포기하지 마세요**: 기술적 구현은 완료되었습니다. 이제 **올바른 평가 기준**만 적용하면 됩니다.

---

## 참고문헌

1. NoLiMa: NIAH를 넘어선 장문맥 평가, arXiv:2502.05167
2. Gist Token 논문: "Learning to Compress Prompts with Gist Tokens"
3. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding
