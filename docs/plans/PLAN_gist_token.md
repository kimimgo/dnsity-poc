# Gist Token PoC - TDD Implementation Plan

## 프로젝트 개요

**목표**: Gist Token 기반 문맥 압축을 TDD 방식으로 구현하여 가정용 GPU(RTX 4090 24GB)에서 효율적인 장문 처리

**핵심 가설**: Gist Token이 RAG 대비 메모리/속도 효율성을 유지하면서 Full Context 수준의 글로벌 문맥 이해 성능을 달성할 수 있다.

**개발 원칙**:
- 모델 성능(Loss)이 아닌 **로직의 정합성**과 **파이프라인 안정성**을 테스트
- 확률적 학습 요소는 Sanity Check로 검증 (`test_overfit_one_batch`)
- 테스트 커버리지 ≥ 80% (핵심 로직 ≥ 90%)

---

## Phase별 요약

| Phase | 상태 | 의존성 | 병렬 가능 | 예상 소요 |
|-------|------|---------|----------|-----------|
| Phase 1 | ✅ 완료 | - | - | 완료 |
| Phase 2 | 🔴 대기 | #1 | No | 2-3일 |
| Phase 3 | 🔴 대기 | #2 | No | 0.5일 + GPU 6-12h |
| Phase 4 | 🟡 병렬 가능 | - | **Yes** | 1-2일 |
| Phase 5 | 🔴 대기 | #3, #4 | Partial | 1-2일 |
| Phase 6 | 🔴 대기 | #3 | No | 1일 |

**Critical Path**: Phase 2 → Phase 3 → Phase 6 → Phase 5

---

## Phase 1: 환경 및 데이터 준비 ✅ 완료

**GitHub Issue**: [#1](https://github.com/kimimgo/dnsity-poc/issues/1) (Closed)

### 완료 항목
- ✅ LongBench 다운로드 스크립트 (7/7 tests)
- ✅ NIAH 생성 스크립트 (9/9 tests)
- ✅ GPU 환경 검증 (RTX 4090 25.3GB)
- ✅ 데이터셋 샘플 생성 (LongBench 9.4MB + NIAH 2.0MB)

### Quality Gate
- [x] 전체 테스트 29/32 통과 (Phase 1 관련 100%)
- [x] 데이터셋 생성 완료
- [x] GPU 환경 검증 완료

---

## Phase 2: Gist Token 구현 (Critical Path)

**GitHub Issue**: [#2](https://github.com/kimimgo/dnsity-poc/issues/2)

**중요도**: ⭐⭐⭐⭐⭐ **프로젝트의 성패를 결정하는 Phase**

### TDD 작업 순서

#### Component 1: Token Embedder & Tokenizer

**[RED] - 실패하는 테스트 작성**
```bash
# tests/unit/test_tokenizer_expansion.py
pytest tests/unit/test_tokenizer_expansion.py::test_tokenizer_expansion -v
pytest tests/unit/test_tokenizer_expansion.py::test_vocab_size_change -v
pytest tests/unit/test_tokenizer_expansion.py::test_embedding_layer_resize -v
```

**테스트 내용**:
- Gist 토큰 `<GIST_0>` ~ `<GIST_N>`이 tokenizer에 추가되었는지
- vocab_size가 정확히 N+1 증가했는지
- `model.resize_token_embeddings()` 후 shape 검증

**[GREEN] - 최소 구현**
```python
# src/model/gist_tokenizer.py
def add_gist_tokens(tokenizer, model, num_gist_tokens):
    gist_tokens = [f"<GIST_{i}>" for i in range(num_gist_tokens)]
    tokenizer.add_special_tokens({"additional_special_tokens": gist_tokens})
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model
```

**[REFACTOR] - 개선**
- Config 파일에서 `num_gist_tokens` 로드
- 토큰 추가 로직을 재사용 가능한 함수로 분리

**검증 명령어**:
```bash
pytest tests/unit/test_tokenizer_expansion.py -v --cov=src/model/gist_tokenizer --cov-report=term-missing
```

---

#### Component 2: GistDataCollator (Most Critical)

**[RED] - 실패하는 테스트 작성**
```bash
# tests/unit/test_gist_collator.py
pytest tests/unit/test_gist_collator.py::test_attention_mask_generation -v
pytest tests/unit/test_gist_collator.py::test_query_cannot_see_context -v
pytest tests/unit/test_gist_collator.py::test_query_can_see_gist -v
pytest tests/unit/test_gist_collator.py::test_gist_can_see_context -v
pytest tests/unit/test_gist_collator.py::test_batch_processing -v
```

**테스트 내용**:
```python
def test_query_cannot_see_context():
    """Query 구간이 Context를 볼 수 없어야 함"""
    input_ids = torch.tensor([[1, 2, 3, 32000, 4, 5]])  # 32000 = Gist Token
    gist_token_id = 32000

    mask = create_gist_mask(input_ids, gist_token_id)

    # Query(index 4, 5)가 Context(index 2, 3)를 볼 수 없어야 함
    assert mask[0, 4, 2] == 0  # or -inf depending on implementation
    assert mask[0, 4, 3] == 0
```

**[GREEN] - 최소 구현**
```python
# src/model/gist_collator.py
class GistDataCollator:
    def __call__(self, features):
        # 1. 기본 causal mask 생성
        # 2. Gist token positions 찾기
        # 3. Query → Context 차단 (mask[query_idx, context_idx] = False)
        pass
```

**[REFACTOR] - 개선**
- Tensor 연산 벡터화 (for loop 제거)
- Attention mask 시각화 유틸리티 추가
- Edge case 처리 (Gist가 없는 경우)

**검증 명령어**:
```bash
pytest tests/unit/test_gist_collator.py -v --cov=src/model/gist_collator --cov-report=html
# Coverage 목표: 100%
```

**시각화 검증** (수동):
```python
# notebooks/verify_attention_mask.ipynb
visualize_attention_mask(sample_batch)
# Expected: Query 구간이 Context를 보지 못하고 Gist만 볼 수 있음
```

---

#### Component 3: LoRA Configuration

**[RED] - 실패하는 테스트 작성**
```bash
pytest tests/unit/test_lora_config.py::test_trainable_parameters -v
pytest tests/unit/test_lora_config.py::test_gist_embedding_gradient -v
pytest tests/unit/test_lora_config.py::test_lora_target_modules -v
```

**테스트 내용**:
```python
def test_gist_embedding_gradient():
    """Gist 토큰 임베딩에 gradient가 흐르는지 검증"""
    model = setup_gist_model_with_lora()
    input_data = ...
    loss = model(input_data).loss
    loss.backward()

    gist_token_indices = get_gist_token_indices(tokenizer)
    gist_embed_grad = model.model.embed_tokens.weight.grad[gist_token_indices]

    assert torch.sum(torch.abs(gist_embed_grad)) > 0, "Gist Token이 학습되고 있지 않음!"
```

**[GREEN] - 최소 구현**
```python
# src/model/gist_lora.py
from peft import LoraConfig, get_peft_model

def setup_lora(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        modules_to_save=["embed_tokens", "lm_head"],  # CRITICAL!
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    return get_peft_model(model, lora_config)
```

**[REFACTOR] - 개선**
- LoRA config를 YAML 파일로 분리
- Gradient flow 검증을 학습 callback으로 자동화

**검증 명령어**:
```bash
pytest tests/unit/test_lora_config.py -v --cov=src/model/gist_lora
```

---

#### Component 4: Attention Mask Visualization

**[RED] - 실패하는 테스트**
```bash
pytest tests/unit/test_mask_visualization.py::test_mask_visualization_output -v
```

**[GREEN] - 구현**
```python
# src/utils/visualization.py
import matplotlib.pyplot as plt

def visualize_attention_mask(mask, positions):
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap="viridis")
    # Context/Gist/Query 구간 라벨링
    plt.savefig("attention_mask.png")
```

**검증 명령어**:
```bash
pytest tests/unit/test_mask_visualization.py -v
```

---

### Phase 2 Quality Gate

**테스트 통과 기준**:
```bash
# 모든 단위 테스트 통과
pytest tests/unit/test_tokenizer_expansion.py -v
pytest tests/unit/test_gist_collator.py -v
pytest tests/unit/test_lora_config.py -v
pytest tests/unit/test_mask_visualization.py -v

# Coverage 확인
pytest tests/unit/ --cov=src/model --cov-report=term-missing
# 목표: ≥ 90%
```

**수동 검증**:
- [ ] Attention mask 시각화 결과가 설계 의도와 일치
- [ ] Gist 임베딩 gradient가 0이 아님 (`test_gradient_flow` 통과)

**완료 조건**:
- ✅ 모든 단위 테스트 100% 통과
- ✅ Coverage ≥ 90%
- ✅ Attention mask 시각적 검증 완료
- ✅ Phase 3 학습 시작 준비 완료

---

## Phase 3: 학습 실행

**GitHub Issue**: [#3](https://github.com/kimimgo/dnsity-poc/issues/3)

**의존성**: Phase 2 완료 필수

### TDD 작업 순서

#### Component 1: Trainer Sanity Check

**[RED] - 실패하는 테스트**
```bash
pytest tests/unit/test_trainer.py::test_overfit_one_batch -v
pytest tests/unit/test_trainer.py::test_gradient_accumulation -v
pytest tests/unit/test_trainer.py::test_checkpoint_save_load -v
pytest tests/unit/test_trainer.py::test_vram_limit -v
```

**테스트 내용**:
```python
def test_overfit_one_batch():
    """단일 배치 10 step 학습 시 Loss가 0에 수렴하는지"""
    model, tokenizer = setup_gist_model()
    batch = get_single_batch()

    trainer = Trainer(model=model, args=training_args)

    initial_loss = trainer.evaluate(batch)
    trainer.train(max_steps=10)
    final_loss = trainer.evaluate(batch)

    assert final_loss < initial_loss * 0.1, "모델이 단일 배치를 overfit하지 못함!"
```

**[GREEN] - 구현**
```python
# src/training/train_gist.py
from transformers import Trainer, TrainingArguments

def setup_trainer(model, tokenizer, train_dataset):
    training_args = TrainingArguments(
        output_dir="checkpoints/gist-10",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_steps=100,
        max_steps=1000,
        bf16=True,
        logging_steps=10,
        save_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=GistDataCollator(tokenizer)
    )

    return trainer
```

**[REFACTOR]**
- Custom Trainer 클래스로 확장
- VRAM 모니터링 decorator 추가

**검증 명령어**:
```bash
pytest tests/unit/test_trainer.py -v
```

---

#### Component 2: Experiment Configuration

**[RED] - 실패하는 테스트**
```bash
pytest tests/unit/test_config.py::test_load_experiment_config -v
pytest tests/unit/test_config.py::test_gist_count_variation -v
```

**[GREEN] - 구현**
```yaml
# experiments/configs/gist_10.yaml
model:
  name: meta-llama/Meta-Llama-3-8B-Instruct
  num_gist_tokens: 10
  quantization: 4bit

training:
  learning_rate: 1e-4
  batch_size: 1
  gradient_accumulation_steps: 8
  warmup_steps: 100
  max_steps: 1000
```

**검증 명령어**:
```bash
pytest tests/unit/test_config.py -v
```

---

#### Component 3: Training Execution (실제 학습)

**학습 실행** (테스트 아님):
```bash
# Experiment 1: Gist-10
python src/training/train_gist.py --config experiments/configs/gist_10.yaml

# Experiment 2: Gist-25
python src/training/train_gist.py --config experiments/configs/gist_25.yaml

# Experiment 3: Gist-50
python src/training/train_gist.py --config experiments/configs/gist_50.yaml
```

**모니터링**:
```bash
# VRAM 사용량
watch -n 1 nvidia-smi

# Tensorboard
tensorboard --logdir checkpoints/
```

---

### Phase 3 Quality Gate

**테스트 통과**:
```bash
pytest tests/unit/test_trainer.py -v
pytest tests/unit/test_config.py -v
```

**학습 완료 조건**:
- [ ] `test_overfit_one_batch` 통과
- [ ] 3개 실험 모두 완료 (Gist-10/25/50)
- [ ] 각 체크포인트 저장 완료
- [ ] VRAM 사용량 24GB 이내 유지
- [ ] Loss가 수렴 경향 확인
- [ ] Gradient explosion 없음 (grad_norm < 10)

---

## Phase 4: Baseline 구축 (병렬 가능)

**GitHub Issue**: [#4](https://github.com/kimimgo/dnsity-poc/issues/4)

**병렬 작업 가능**: Phase 2, 3와 독립적으로 진행 가능

### TDD 작업 순서

#### Component 1: Full Context Baseline

**[RED] - 실패하는 테스트**
```bash
pytest tests/unit/test_baseline.py::test_full_context_inference -v
pytest tests/unit/test_baseline.py::test_baseline_passkey_accuracy -v
pytest tests/unit/test_baseline.py::test_vram_measurement -v
```

**[GREEN] - 구현**
```python
# src/baseline/full_context.py
class FullContextBaseline:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def inference(self, context, question):
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        return self.model.generate(...)
```

**검증 명령어**:
```bash
pytest tests/unit/test_baseline.py -v --cov=src/baseline
```

---

#### Component 2: RAG Pipeline

**[RED] - 실패하는 테스트**
```bash
pytest tests/unit/test_rag.py::test_retrieval_correctness -v
pytest tests/unit/test_rag.py::test_top_k_retrieval -v
pytest tests/unit/test_rag.py::test_rag_end_to_end -v
```

**[GREEN] - 구현**
```python
# src/baseline/rag_pipeline.py
import chromadb
from sentence_transformers import SentenceTransformer

class RAGPipeline:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("documents")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def add_documents(self, documents):
        # Chunk and embed
        pass

    def retrieve(self, query, top_k=3):
        # Vector search
        pass
```

**검증 명령어**:
```bash
pytest tests/unit/test_rag.py -v --cov=src/baseline/rag_pipeline
```

---

### Phase 4 Quality Gate

```bash
pytest tests/unit/test_baseline.py -v
pytest tests/unit/test_rag.py -v
```

**완료 조건**:
- [ ] Full Context baseline이 NIAH에서 95% 이상 정확도
- [ ] RAG 파이프라인 단위 테스트 100% 통과
- [ ] 성능 측정 로직 검증 완료
- [ ] Coverage ≥ 80%

---

## Phase 5: 평가 및 분석

**GitHub Issue**: [#5](https://github.com/kimimgo/dnsity-poc/issues/5)

**의존성**: Phase 3, 4 완료 필요

### TDD 작업 순서

#### Component 1: Passkey Retrieval Metric

**[RED] - 실패하는 테스트**
```bash
pytest tests/unit/test_metrics.py::test_passkey_extraction -v
pytest tests/unit/test_metrics.py::test_passkey_exact_match -v
pytest tests/unit/test_metrics.py::test_passkey_edge_cases -v
```

**[GREEN] - 구현**
```python
# src/evaluation/metrics.py
import re

def extract_passkey(model_output):
    """모델 출력에서 Passkey 추출"""
    pattern = r"[A-Z0-9]{6}"
    match = re.search(pattern, model_output)
    return match.group(0) if match else None

def passkey_accuracy(predictions, ground_truths):
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    return correct / len(predictions)
```

**검증 명령어**:
```bash
pytest tests/unit/test_metrics.py -v --cov=src/evaluation/metrics
```

---

#### Component 2: Quantitative Metrics

**[RED] - 실패하는 테스트**
```bash
pytest tests/unit/test_profiler.py::test_vram_measurement -v
pytest tests/unit/test_profiler.py::test_ttft_measurement -v
pytest tests/unit/test_profiler.py::test_throughput_calculation -v
```

**[GREEN] - 구현**
```python
# src/evaluation/profiler.py
import torch
import time

class MemoryProfiler:
    @staticmethod
    def measure_vram():
        return torch.cuda.max_memory_allocated() / 1e9  # GB

class LatencyProfiler:
    @staticmethod
    def measure_ttft(model, inputs):
        start = time.time()
        _ = model.generate(**inputs, max_new_tokens=1)
        return (time.time() - start) * 1000  # ms
```

**검증 명령어**:
```bash
pytest tests/unit/test_profiler.py -v --cov=src/evaluation/profiler
```

---

### Phase 5 Quality Gate

```bash
pytest tests/unit/test_metrics.py -v
pytest tests/unit/test_evaluator.py -v
pytest tests/unit/test_profiler.py -v
```

**완료 조건**:
- [ ] 평가 스크립트가 Mock 데이터에서 정확한 점수 계산
- [ ] 전체 테스트셋 평가 완료 (중단 없음)
- [ ] 결과 시각화 생성 완료
- [ ] Coverage ≥ 80%

---

## Phase 6: KV Cache 압축 구현

**GitHub Issue**: [#6](https://github.com/kimimgo/dnsity-poc/issues/6)

**의존성**: Phase 3 완료 필요

### TDD 작업 순서

#### Component 1: KV Cache Manager

**[RED] - 실패하는 테스트**
```bash
pytest tests/unit/test_kv_cache.py::test_kv_cache_extraction -v
pytest tests/unit/test_kv_cache.py::test_kv_cache_shape_gqa -v
pytest tests/unit/test_kv_cache.py::test_kv_cache_slicing -v
pytest tests/unit/test_kv_cache.py::test_kv_value_preservation -v
```

**테스트 내용**:
```python
def test_kv_cache_slicing():
    """Gist 구간만 슬라이싱했을 때 shape 검증"""
    # Llama-3 8B: 32 layers, 8 kv_heads, seq_len=1000
    past_kv = create_dummy_cache(num_layers=32, num_kv_heads=8, seq_len=1000, dim=128)

    gist_indices = [999]  # Gist token at position 999
    compressed = compress_context(past_kv, gist_indices)

    # Shape should be reduced: seq_len 1000 -> ~10 (Gist only)
    assert compressed[0][0].shape[2] < 50
```

**[GREEN] - 구현**
```python
# src/inference/kv_cache_manager.py
def compress_context(past_key_values, gist_indices):
    """KV Cache에서 Gist 구간만 추출"""
    compressed_kv = []

    for layer_kv in past_key_values:
        key, value = layer_kv
        # Slice to keep only Gist positions
        compressed_key = key[:, :, gist_indices, :]
        compressed_value = value[:, :, gist_indices, :]
        compressed_kv.append((compressed_key, compressed_value))

    return tuple(compressed_kv)
```

**[REFACTOR]**
- KV Cache를 파일로 저장/로드 (torch.save/load)
- CacheManager 클래스로 여러 문서 관리

**검증 명령어**:
```bash
pytest tests/unit/test_kv_cache.py -v --cov=src/inference/kv_cache_manager
# Coverage 목표: 90%
```

---

#### Component 2: Compressed Inference

**[RED] - 실패하는 테스트**
```bash
pytest tests/unit/test_compressed_inference.py::test_compressed_generation -v
pytest tests/unit/test_compressed_inference.py::test_generation_consistency -v
pytest tests/unit/test_compressed_inference.py::test_memory_saving -v
```

**[GREEN] - 구현**
```python
# src/inference/compressed_inference.py
def generate_with_compressed_kv(model, compressed_kv, question):
    """압축된 KV Cache를 사용한 생성"""
    inputs = tokenizer(question, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        past_key_values=compressed_kv,
        max_new_tokens=100
    )

    return tokenizer.decode(outputs[0])
```

**검증 명령어**:
```bash
pytest tests/unit/test_compressed_inference.py -v
```

---

### Phase 6 Quality Gate

```bash
pytest tests/unit/test_kv_cache.py -v
pytest tests/unit/test_compressed_inference.py -v
```

**완료 조건**:
- [ ] 압축 전후 생성 결과 일관성 확인
- [ ] 메모리 절감 효과 입증 (100x+ 압축)
- [ ] GQA 구조 지원 확인
- [ ] Coverage ≥ 90%

---

## 전체 워크플로우 요약

### Critical Path
```
Phase 1 (완료) → Phase 2 (2-3일) → Phase 3 (0.5일 + GPU 6-12h)
                                   → Phase 6 (1일) → Phase 5 (1-2일)
```

### 병렬 작업
```
Phase 2/3 진행 중 → Phase 4 (1-2일, 독립적)
Phase 3 진행 중 → Phase 5 평가 로직 (Mock 데이터로 미리 개발)
```

### 예상 총 소요 시간
- **Sequential**: 약 7-10일
- **Parallel 최적화**: 약 5-7일 (Phase 4를 학습 중에 개발)

---

## Quality Gates 통합 체크리스트

### Phase 2
- [ ] `pytest tests/unit/test_tokenizer_expansion.py -v` 100% 통과
- [ ] `pytest tests/unit/test_gist_collator.py -v` 100% 통과
- [ ] `pytest tests/unit/test_lora_config.py -v` 100% 통과
- [ ] Attention mask 시각화 검증 완료
- [ ] Coverage ≥ 90%

### Phase 3
- [ ] `pytest tests/unit/test_trainer.py -v` 통과
- [ ] `test_overfit_one_batch` 통과
- [ ] Gist-10/25/50 학습 완료
- [ ] VRAM < 24GB 유지

### Phase 4
- [ ] `pytest tests/unit/test_baseline.py -v` 통과
- [ ] `pytest tests/unit/test_rag.py -v` 통과
- [ ] Full Context NIAH 정확도 ≥ 95%

### Phase 5
- [ ] `pytest tests/unit/test_metrics.py -v` 통과
- [ ] `pytest tests/unit/test_evaluator.py -v` 통과
- [ ] 전체 평가 완료

### Phase 6
- [ ] `pytest tests/unit/test_kv_cache.py -v` 통과
- [ ] 압축 전후 생성 일관성 확인
- [ ] 메모리 절감 100x+ 입증

---

## 개발 환경 설정

### 테스트 실행
```bash
# 전체 테스트
pytest tests/ -v

# Coverage 리포트
pytest tests/ --cov=src --cov-report=html

# 특정 Phase만
pytest tests/unit/test_gist_collator.py -v
```

### 코드 품질
```bash
# Linting
ruff check src/ tests/

# Type checking
mypy src/
```

### Git Workflow
```bash
# Phase 2 작업 시작
git checkout -b phase-2-gist-token

# 작업 완료 후
git add .
git commit -m "feat: Implement GistDataCollator with attention masking"
git push origin phase-2-gist-token

# PR 생성
gh pr create --title "Phase 2: Gist Token 구현" --body "Closes #2"
```

---

## 성공 기준 (PoC 완료)

본 프로젝트가 성공했다고 판단하는 최종 기준:

1. ✅ **메모리**: Gist-25가 RAG와 동등한 VRAM (≤ 10GB)
2. ✅ **속도**: TTFT가 RAG 대비 30% 이상 단축
3. ✅ **품질**: Global Theme에서 RAG 대비 20%p 이상 향상 (≥ 80%)
4. ✅ **압축 검증**: Passkey Accuracy ≥ 70%

---

## 다음 단계 (PoC 성공 후)

- Hierarchical Gist (다단계 압축)
- Multimodal Gist (이미지+텍스트)
- Production 배포 (vLLM 통합)
- 논문 작성 및 벤치마크 공개

---

## Gemini 분석 요약

Gemini Pro의 주요 인사이트:

> **"ML 프로젝트의 TDD는 모델 성능이 아닌 로직의 정합성과 파이프라인 안정성을 테스트한다"**

**Critical 발견**:
- `GistDataCollator`가 프로젝트의 성패를 결정하는 Critical Path
- Attention mask 로직의 단위 테스트를 매우 촘촘하게 작성 필수
- Phase 4 (Baseline)를 학습 시간 활용해 병렬 진행 권장
- GQA 구조 (num_kv_heads ≠ num_heads) 지원 필수

**실용적 TDD 접근**:
- 확률적 학습은 `test_overfit_one_batch`로 Sanity Check
- Gradient flow는 backward 후 grad 값으로 검증
- KV Cache는 Mock 텐서로 shape 검증
- Memory profiling은 decorator로 자동화

---

## 참고 문서

- [EXPERIMENT_DESIGN.md](../../EXPERIMENT_DESIGN.md): 실험 설계 상세
- [CLAUDE.md](../../CLAUDE.md): 구현 가이드라인
- [GitHub Issues #1-#6](https://github.com/kimimgo/dnsity-poc/issues): Phase별 상세 작업

---

**작성일**: 2026-01-04
**작성자**: Claude Code + Gemini Pro Analysis
**버전**: 1.0
