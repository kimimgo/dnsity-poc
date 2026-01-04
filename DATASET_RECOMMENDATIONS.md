# Gist Token 평가를 위한 벤치마크 데이터셋 추천

## 🤖 Gemini 분석 (gemini-3-pro-preview)

Gist Token과 같은 Context Compression 기술은 '정보의 손실(Lossy Compression)'을 전제로 하므로, 단순한 성능 저하뿐만 아니라 **"어떤 정보가 보존되고(Global), 어떤 정보가 소실되는지(Passkey/Detail)"**를 입체적으로 분석하는 것이 핵심입니다.

### 평가 프레임워크 3축

1. **Passkey Retrieval (정보 보존율)**: 압축 과정에서 Critical한 단일 토큰이 소실되는지 측정
2. **Global Understanding (맥락 압축력)**: 전체 내용을 아울러야만 답할 수 있는 질문(요약, 주제 파악)
3. **Factual QA (세부 정보 조회)**: RAG와 직접 비교되는 영역

---

## 📊 Phase 1: 글로벌 표준 벤치마크 (1차 실험)

### 1. LongBench (★ 최우선 추천)

**출처**: [ACL 2024](https://aclanthology.org/2024.acl-long.172/), [arXiv:2308.14508](https://arxiv.org/abs/2308.14508)

**개요**:
- 첫 번째 이중언어(영어/중국어) 다중 과제 벤치마크
- 21개 데이터셋, 6개 카테고리
- 평균 길이: 6,711 단어 (영어), 13,386 자 (중국어)

**평가 태스크**:
| 태스크 | 데이터셋 | Gist 평가 적합성 |
|--------|----------|------------------|
| **Single-doc QA** | NarrativeQA, Qasper | ✅ Factual QA |
| **Multi-doc QA** | MultiFieldQA, HotpotQA | ✅ Global Understanding |
| **Summarization** | GovReport, QMSum | ✅✅ **Global (핵심)** |
| **Few-shot Learning** | TREC, TriviaQA | Factual QA |
| **Synthetic Tasks** | Passkey Retrieval | ✅✅ **Passkey (핵심)** |
| **Code Completion** | LCC, RepoBench | Factual QA |

**장점**:
- ✅ 학계 표준 (200+ citations)
- ✅ Hugging Face 즉시 사용 가능
- ✅ 3축 평가 모두 커버 (Passkey 포함!)
- ✅ 2k-8k 토큰 범위 완벽 매칭

**단점**:
- ❌ 일부 태스크는 Gist보다 RAG에 유리할 수 있음 (Multi-doc)

**접근성**:
```python
from datasets import load_dataset
dataset = load_dataset("THUDM/LongBench", "narrativeqa")
```
- [Hugging Face](https://huggingface.co/datasets/THUDM/LongBench)
- [GitHub](https://github.com/THUDM/LongBench)

**LongBench v2 (2025)**:
- 503개 난이도 높은 객관식 문제
- 8k~2M 단어 컨텍스트
- 인간 전문가 정확도: 53.7% (15분 제한)
- [웹사이트](https://longbench2.github.io/)

---

### 2. NIAH (Needle In A Haystack) (★ 필수)

**출처**: [GitHub - gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)

**개요**:
- Passkey Retrieval의 표준 구현
- 긴 텍스트(Haystack) 속에 무작위 정보(Needle) 삽입
- 압축 알고리즘의 "해상도" 측정

**평가 방식**:
```
Context: [5000 tokens of random text]
         ...
         The secret password is: X7G9K2
         ...
         [5000 more tokens]

Question: What is the secret password mentioned in the text?
```

**장점**:
- ✅ **Passkey Retrieval 직접 측정**
- ✅ Gist Token의 정보 손실 정량화
- ✅ 자동 생성 가능 (다양한 길이/위치 실험)
- ✅ RAG의 chunking 전략 약점 노출

**단점**:
- ❌ 합성 데이터 (현실 시나리오와 괴리 가능)

**사용 사례**:
- Gist Token 개수(10/25/50)별 Passkey 정확도 측정
- 압축률에 따른 정보 손실 임계점 파악

---

### 3. SCROLLS (보조 벤치마크)

**출처**: [EMNLP 2022](https://aclanthology.org/2022.emnlp-main.623/), [SCROLLS Benchmark](https://www.scrolls-benchmark.com/)

**개요**:
- 7개 자연어 태스크
- 문학, 과학, 비즈니스, 엔터테인먼트 도메인

**평가 태스크**:
- GovReport (요약)
- QMSum (회의록 요약)
- QASPER (과학 논문 QA)
- NarrativeQA (소설 이해)
- QuALITY (객관식 독해)
- ContractNLI (계약서 추론)

**장점**:
- ✅ 도메인 다양성 (계약서, 논문 등 전문 문서)
- ✅ Global Understanding 강조

**단점**:
- ❌ Passkey 직접 측정 없음
- ❌ LongBench와 일부 중복

**접근성**: [Hugging Face](https://huggingface.co/datasets/tau/scrolls)

---

### 4. InfBench (∞Bench) (선택적)

**출처**: [arXiv:2402.13718](https://arxiv.org/abs/2402.13718)

**개요**:
- 100k+ 토큰 초장문 평가
- 12개 태스크 (합성 + 현실)
- 영어/중국어

**장점**:
- ✅ 극한 압축률 테스트 (100k → 50 Gist)

**단점**:
- ❌ 요구사항(2k-8k)을 초과하는 길이
- ❌ 24GB VRAM으로 Full Context 실행 불가능 (비교 기준 없음)

**권장**: PoC 성공 후 확장 실험으로 활용

---

## 🇰🇷 Phase 2: 한국어 벤치마크 (2차 실험)

### 1. KorQuAD 2.0 (HTML/Long Context)

**출처**: [Hugging Face](https://huggingface.co/datasets/squad_kor_v2)

**개요**:
- 한국어 위키백과 기반 기계독해
- HTML 구조 포함 (표, 리스트)
- 평균 문맥 길이: ~1.5k 토큰

**장점**:
- ✅ 한국어 표준 벤치마크
- ✅ 구조적 정보 압축 테스트
- ✅ Factual QA 평가

**단점**:
- ❌ 길이가 다소 짧음 (2k 미만)
- ❌ Global Understanding 약함

**보완 방법**:
여러 문서를 연결하여 4k-8k 토큰 구성

---

### 2. 행정고시 문제 데이터셋 (자체 구축 권장)

**배경**:
- 2025년 1월 6일 뉴스: AI 국가대표 선발 과제로 행정고시 문제 활용
- 긴 지문(법률, 정책 문서) + 복잡한 추론

**구축 방법**:
1. 공개된 행정고시 기출문제 수집 (5급 공채)
2. 긴 지문 선별 (2k-6k 토큰)
3. 영역별 분류:
   - 헌법: 판례 전문 + 법리 이해
   - 행정학: 긴 정책 사례 분석
   - 경제학: 복잡한 시나리오 추론

**평가 적합성**:
- ✅✅ **Global Understanding** (정책 전반 이해 필수)
- ✅ Factual QA (법조문 정확한 인용)
- ✅ Reasoning (다단계 논리)

**예상 샘플**:
```json
{
  "context": "[헌법재판소 2015헌마123 판례 전문 4000토큰]",
  "question": "이 판례가 기존 판례와 달리 판단한 핵심 쟁점은?",
  "answer": "...",
  "type": "global_understanding"
}
```

**장점**:
- ✅ 한국어 전문 도메인
- ✅ 실제 인간 평가 기준 존재 (합격 커트라인)
- ✅ RAG 취약 영역 (판례 전체 흐름 이해)

**단점**:
- ❌ 수동 수집 및 라벨링 필요
- ❌ 저작권 문제 (공공 데이터 확인 필요)

---

### 3. Ko-NIAH (자체 생성 필수)

**Gemini 추천 방법**:
1. 한국어 소설/뉴스 20개 연결 (4k-8k 토큰)
2. 중간에 무관한 문장 삽입: "이민규의 비밀번호는 1234이다"
3. 질문: "문서에서 언급된 이민규의 비밀번호는?"

**이유**:
- ✅ 한국어 조사/어미 처리 압축 검증
- ✅ Passkey Retrieval 직접 측정

**생성 스크립트**:
```python
import random
from transformers import AutoTokenizer

def create_ko_niah(base_texts, needle, position="middle"):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    combined = "".join(base_texts)
    tokens = tokenizer.encode(combined)

    if position == "middle":
        insert_idx = len(tokens) // 2

    needle_tokens = tokenizer.encode(needle)
    result_tokens = tokens[:insert_idx] + needle_tokens + tokens[insert_idx:]

    return tokenizer.decode(result_tokens)
```

---

## 🧠 Claude 분석 및 추가 인사이트

### 1. 최근 Gist Token 연구와의 정합성

**ACL 2025 논문 발견** ([arXiv:2412.17483](https://arxiv.org/abs/2412.17483)):
- "A Silver Bullet or a Compromise for Full Attention?"
- **사용 데이터셋**: RULER, MMLU-Pro, GSM8K, HellaSwag, BBH
- **컨텍스트 길이**: 16k 토큰
- **평가 모델**: Llama3.1-8B, Qwen2-7B

**우리 실험과의 차이점**:
| 항목 | ACL 2025 논문 | 우리 PoC |
|------|---------------|----------|
| 컨텍스트 | 16k | 2k-8k (GPU 제약) |
| 데이터셋 | RULER (합성) | LongBench (현실) |
| 목표 | 학술 분석 | RAG 대체 가능성 검증 |

**권장 전략**:
- LongBench로 메인 실험 → ACL 논문과 직접 비교 가능
- RULER 일부 채택 → 논문 재현성 확보

---

### 2. 데이터셋 선정 우선순위

#### 1차 실험 (글로벌)

**필수 (Tier 1)**:
1. ✅ **LongBench** - Summarization (GovReport, QMSum)
   - Global Understanding 핵심 지표
   - 학계 표준 비교 가능

2. ✅ **NIAH (자체 생성)**
   - Passkey Retrieval 직접 측정
   - 압축률별 정보 손실 정량화

**선택 (Tier 2)**:
3. ⭕ **LongBench** - QA (Qasper, NarrativeQA)
   - Factual QA 보완
   - RAG 비교 기준

#### 2차 실험 (한국어)

**필수 (Tier 1)**:
1. ✅ **Ko-NIAH (자체 생성)**
   - 한국어 토큰화 압축 검증
   - 조사/어미 처리 확인

**선택 (Tier 2)**:
2. ⭕ **행정고시 기출 (자체 구축)**
   - 실용성 극대화
   - AI 국가대표 과제와 비교

3. ⭕ **KorQuAD 2.0 (확장)**
   - 여러 문서 연결하여 4k-8k 구성

---

### 3. 실험 설계 비교표 (최종)

| 비교군 | 데이터셋 | 평가 지표 | 예상 결과 |
|--------|----------|-----------|-----------|
| **Full Context** | LongBench | Global: 95%<br>Passkey: 100%<br>Factual: 90% | Upper Bound |
| **RAG (Top-3)** | LongBench | Global: 60%<br>Passkey: 75%<br>Factual: 85% | 현재 표준 |
| **Gist-10** | LongBench | Global: 85%<br>Passkey: 70%<br>Factual: 80% | 고압축 |
| **Gist-25** | LongBench | Global: 90%<br>Passkey: 80%<br>Factual: 85% | 균형 ✅ |
| **Gist-50** | LongBench | Global: 92%<br>Passkey: 85%<br>Factual: 88% | 저압축 |

**성공 기준**:
- Gist-25가 RAG 대비 Global에서 +20%p 이상
- Passkey에서 70% 이상 (정보 보존 증명)
- VRAM 사용량 RAG와 동등 (10GB 이하)

---

### 4. 타임라인 및 단계별 작업

| Phase | 작업 | 소요 | 산출물 |
|-------|------|------|--------|
| **1A** | LongBench 다운로드 및 탐색 | 0.5일 | 데이터셋 샘플 100개 |
| **1B** | NIAH 생성 스크립트 작성 | 0.5일 | `create_niah.py` |
| **1C** | 전처리 파이프라인 구현 | 1일 | JSONL 포맷 변환 |
| **2A** | Ko-NIAH 생성 (나무위키/뉴스) | 0.5일 | 한국어 50개 샘플 |
| **2B** | 행정고시 기출 수집 | 1일 | PDF → Text 변환 |
| **2C** | 한국어 Q&A 라벨링 (GPT-4) | 0.5일 | 한국어 JSONL |

**총 소요**: 4일 (데이터 준비 완료)

---

## 📦 즉시 실행 가능한 액션 아이템

### 1. 환경 변수 설정
```bash
# .env 파일 생성
echo "SERPAPI_KEY=95e37232983304675ebc7f5403ec4a8643fdac7795f799f0fe648e6373d3948b" >> .env
echo "SEMANTIC_SCHOLAR_API_KEY=q2bslDrjtT1hNXhGLbWW26F1UaoYC9HWayWXoIrs" >> .env
```

### 2. 데이터셋 다운로드 스크립트
```python
# scripts/download_datasets.py
from datasets import load_dataset

# LongBench
datasets = {
    "narrativeqa": load_dataset("THUDM/LongBench", "narrativeqa", split="test"),
    "qasper": load_dataset("THUDM/LongBench", "qasper", split="test"),
    "gov_report": load_dataset("THUDM/LongBench", "gov_report", split="test"),
}

for name, ds in datasets.items():
    ds.save_to_disk(f"data/longbench/{name}")
    print(f"✅ {name}: {len(ds)} samples")
```

### 3. NIAH 생성 스크립트
```python
# scripts/create_niah.py
import random
from datasets import load_dataset

def generate_niah_sample(base_text, needle, position=0.5):
    """
    Args:
        base_text: 긴 텍스트 (4k-8k 토큰)
        needle: 숨길 정보 (예: "The password is X7G9K2")
        position: 삽입 위치 (0.0~1.0)
    """
    words = base_text.split()
    insert_idx = int(len(words) * position)

    result = words[:insert_idx] + [needle] + words[insert_idx:]
    return " ".join(result)

# PG19 소설 데이터셋 활용
pg19 = load_dataset("pg19", split="test")
samples = []

for i in range(100):
    book = random.choice(pg19)
    text = book["text"][:50000]  # 약 10k 토큰

    password = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=6))
    needle = f"The secret passkey is {password}."

    context = generate_niah_sample(text, needle, position=random.uniform(0.3, 0.7))

    samples.append({
        "context": context,
        "question": "What is the secret passkey mentioned in the text?",
        "answer": password,
        "needle_position": context.index(needle) / len(context)
    })

# 저장
import json
with open("data/niah/english_niah_100.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")
```

---

## 🔍 참고 문헌

### 주요 논문
- [LongBench (ACL 2024)](https://aclanthology.org/2024.acl-long.172/)
- [Gist Token Comprehensive Study (ACL 2025)](https://arxiv.org/abs/2412.17483)
- [Learning to Compress Prompts (arXiv 2023)](https://arxiv.org/abs/2304.08467)
- [SCROLLS (EMNLP 2022)](https://aclanthology.org/2022.emnlp-main.623/)
- [InfBench (arXiv 2024)](https://arxiv.org/abs/2402.13718)

### 데이터셋 링크
- [LongBench Hugging Face](https://huggingface.co/datasets/THUDM/LongBench)
- [SCROLLS Hugging Face](https://huggingface.co/datasets/tau/scrolls)
- [InfBench GitHub](https://github.com/OpenBMB/InfiniteBench)
- [KorQuAD 2.0](https://huggingface.co/datasets/squad_kor_v2)
- [NIAH GitHub](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)

---

## ✅ 다음 단계

1. **즉시 실행**: LongBench 3개 태스크 다운로드 (GovReport, Qasper, NarrativeQA)
2. **NIAH 생성**: 영어 100개 샘플 자동 생성
3. **한국어 준비**: Ko-NIAH 생성 스크립트 작성
4. **전처리 파이프라인**: JSONL 포맷 통일

**예상 완료**: 2일 이내 (데이터 준비 완성)
