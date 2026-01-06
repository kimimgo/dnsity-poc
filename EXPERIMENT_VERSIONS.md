# DNSity PoC - Experiment Versioning

**Created**: 2026-01-07
**Last Updated**: 2026-01-07

---

## Version Scheme

```
vX.Y.Z
│ │ └── Patch: Bug fixes, minor adjustments
│ └──── Minor: New experiments within same scope
└────── Major: Paradigm shift or new model/approach
```

---

## Version History

### v0.x - Infrastructure & Data Pipeline

| Version | Date | Description | Status |
|---------|------|-------------|--------|
| v0.1.0 | 2026-01-03 | Project initialization, CLAUDE.md, CONCEPT.md | ✅ Complete |
| v0.2.0 | 2026-01-04 | Data pipeline: NIAH generator (EN/KO) | ✅ Complete |
| v0.3.0 | 2026-01-05 | Attention masking implementation | ✅ Complete |
| v0.4.0 | 2026-01-05 | KV cache compression module | ✅ Complete |

**Key Artifacts**:
- `data/processed/niah/global_niah.jsonl` (200 samples)
- `data/processed/niah/korean_niah.jsonl` (200 samples)
- `src/model/gist_collator.py` (attention masking)
- `src/inference/compress.py` (KV cache compression)

---

### v1.x - Gist Token PoC (Llama-3-8B-Instruct)

**Goal**: Validate Gist Token compression on consumer GPU (24GB VRAM)

| Version | Date | Description | Results | Status |
|---------|------|-------------|---------|--------|
| v1.0.0 | 2026-01-05 | Initial training (H1 bug present) | Loss: 0.0024 | ⚠️ Buggy |
| v1.1.0 | 2026-01-06 | H1 bug fix (GistDataCollator missing) | Sanity check passed | ✅ Fixed |
| v1.2.0 | 2026-01-06 | Full-scale training (25 Gist, 1000 samples) | 34.5% NIAH | ✅ Complete |

#### v1.2.0 Details

**Training Configuration**:
- Model: Llama-3-8B-Instruct (4-bit NF4)
- Gist Tokens: 25
- Training Samples: 1,000
- Training Steps: 750 (3 epochs)
- Training Time: 77 min 36 sec
- Final Loss: 0.0034

**Evaluation Results**:
| Dataset | Samples | Accuracy |
|---------|---------|----------|
| Global NIAH | 200 | 20.0% (40/200) |
| Korean NIAH | 200 | 49.0% (98/200) |
| **Average** | 400 | **34.5%** |

**Compression Performance**:
- Input Context: ~3,800 tokens
- Output: 25 Gist tokens
- Compression Ratio: **152x**

**Key Files**:
- Checkpoint: `checkpoints/gist-llama3-8b-instruct-25-full/`
- Report: `results/kmmlu/GEMINI_EVALUATION_93.md`

---

### v2.x - gpt-oss-20b Evaluation (NEW PARADIGM)

**Goal**: Establish strong baseline with state-of-the-art open model

| Version | Date | Description | Results | Status |
|---------|------|-------------|---------|--------|
| v2.0.0 | 2026-01-07 | NIAH evaluation (Global + Korean) | 99.25% avg | ✅ Complete |
| v2.1.0 | 2026-01-07 | KMMLU Law evaluation | 31.0% | ✅ Complete |

#### v2.0.0 - NIAH Evaluation

**Model Configuration**:
- Model: openai/gpt-oss-20b (21B params, 3.6B active MoE)
- Quantization: MXFP4 (Marlin backend)
- Inference: vLLM 0.13.0
- GPU: RTX 4090 (24GB)
- Memory Usage: ~14GB

**Results**:
| Dataset | Samples | Accuracy |
|---------|---------|----------|
| Global NIAH | 200 | **100.0%** (200/200) |
| Korean NIAH | 200 | **98.5%** (197/200) |
| **Average** | 400 | **99.25%** |

**Key Files**:
- `results/gptoss20b_niah_global_niah_200.json`
- `results/gptoss20b_niah_korean_niah_200.json`

#### v2.1.0 - KMMLU Evaluation

**Results**:
| Subject | Samples | Accuracy | vs Baseline |
|---------|---------|----------|-------------|
| Law | 100 | **31.0%** | +6%p vs Llama-3-8B |

**Key Files**:
- `results/kmmlu/gptoss20b_law_100.json`
- `results/GPT_OSS_EXPERIMENT_RESULTS.md`

---

## Cross-Version Comparison

### NIAH Performance

| Version | Model | Method | Global | Korean | Average |
|---------|-------|--------|--------|--------|---------|
| v2.0.0 | gpt-oss-20b | Direct | **100.0%** | **98.5%** | **99.25%** |
| v1.2.0 | Llama-3-8B | Gist (25) | 20.0% | 49.0% | 34.5% |
| - | Llama-3-8B | RAG | ~60% | ~50% | ~55% |
| - | Llama-3-8B | Full Context | ~95% | ~90% | ~92.5% |

### KMMLU Law Performance

| Version | Model | Method | Accuracy |
|---------|-------|--------|----------|
| v2.1.0 | gpt-oss-20b | Direct | **31.0%** |
| - | Llama-3-8B | RAG | 31.0% |
| - | Llama-3-8B | Baseline | 25.0% |
| - | Llama-3-8B | LoRA | 24.0% |
| v1.2.0 | Llama-3-8B | Gist | 23.0% |

---

## Planned Versions

### v3.x - Hybrid Approaches (Planned)

| Version | Description | Status |
|---------|-------------|--------|
| v3.0.0 | RAG + gpt-oss-20b | 🔜 Planned |
| v3.1.0 | Gist Token + gpt-oss-20b | 🔜 Planned |

### v4.x - Standard Benchmarks (Planned)

| Version | Description | Status |
|---------|-------------|--------|
| v4.0.0 | LongBench evaluation | 🔜 Planned |
| v4.1.0 | RULER evaluation | 🔜 Planned |

---

## Git Tags

Each version should be tagged in git:

```bash
# Infrastructure
git tag -a v0.4.0 -m "KV cache compression module"

# Gist Token PoC
git tag -a v1.2.0 -m "Full-scale Gist training: 25 tokens, 34.5% NIAH accuracy"

# gpt-oss-20b Evaluation (NEW)
git tag -a v2.0.0 -m "gpt-oss-20b NIAH: 99.25% accuracy"
git tag -a v2.1.0 -m "gpt-oss-20b KMMLU Law: 31.0% accuracy"
```

---

## Version Selection Guide

| Scenario | Recommended Version |
|----------|---------------------|
| Maximum accuracy needed | v2.x (gpt-oss-20b) |
| Memory-constrained deployment | v1.x (Gist Token) |
| Research on compression | v1.x (Gist Token) |
| Korean language tasks | v2.0.0 (98.5% Korean NIAH) |
| KMMLU evaluation | v2.1.0 |

---

## Changelog

### 2026-01-07
- Created EXPERIMENT_VERSIONS.md
- Defined v0.x, v1.x, v2.x versioning scheme
- v2.x designated as NEW PARADIGM for gpt-oss-20b experiments
- KMMLU experiments assigned to v2.1.0

### 2026-01-06
- v1.2.0: Full-scale Gist Token training completed
- v1.1.0: H1 bug (missing GistDataCollator) fixed

### 2026-01-05
- v1.0.0: Initial Gist Token training (buggy)
- v0.3.0, v0.4.0: Core infrastructure completed
