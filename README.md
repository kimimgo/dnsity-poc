# DNSity PoC - Gist Token Context Compression

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Proof-of-Concept implementation of Gist Token technology for efficient context compression on consumer GPUs (24GB VRAM).

## 🎯 Project Goal

Compress long contexts (2k-8k tokens) into learnable virtual tokens (10-50 Gist tokens) to enable:
- 📉 **100-400x memory reduction** in KV cache
- ⚡ **50%+ faster inference** compared to RAG
- 🎓 **Global context understanding** while maintaining factual accuracy

## 🏗️ Architecture

Unlike traditional approaches:
- **RAG**: Searches external DB → slow retrieval, fragmented context
- **Fine-tuning**: Bakes knowledge into weights → inflexible updates
- **Gist Token**: Compresses context into input tokens → instant updates, global understanding

### Core Mechanism

```
[Instruction] + [Context] + [<GIST_0>...<GIST_N>] + [Question] + [Answer]
                                     ↑
                          Attention masking forces
                          information bottleneck
```

**Key Innovation**: Custom attention masking blocks Question/Answer from seeing Context directly, forcing compression through Gist tokens.

## 📊 Evaluation Strategy

### Datasets

**Phase 1 (Global)**:
- ✅ [LongBench](https://huggingface.co/datasets/THUDM/LongBench) - Summarization, QA tasks
- ✅ NIAH (Needle in Haystack) - Passkey retrieval

**Phase 2 (Korean)**:
- ✅ Ko-NIAH - Korean tokenization validation
- ✅ Administrative exam questions - Real-world evaluation

### Metrics

| Metric | Target |
|--------|--------|
| Compression Ratio | 100-400x |
| VRAM Usage | ≤ 10GB (vs RAG) |
| TTFT | 50% faster than RAG |
| Passkey Accuracy | ≥ 70% |
| Global Understanding | ≥ 80% (vs Full Context) |

## 🚀 Quick Start

### Prerequisites

- NVIDIA RTX 3090/4090 (24GB VRAM)
- Python 3.10+
- CUDA 11.8+

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/dnsity-poc.git
cd dnsity-poc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Download Datasets

```bash
# LongBench
python scripts/download_longbench.py

# Generate NIAH samples
python scripts/create_niah.py --samples 100 --output data/processed/niah_en.jsonl
```

### Training

```bash
# Train with 25 Gist tokens
python src/training/train_gist.py \
  --config experiments/configs/gist_25.yaml \
  --output_dir checkpoints/gist-25
```

### Evaluation

```bash
# Evaluate on LongBench
python experiments/eval_longbench.py \
  --checkpoint checkpoints/gist-25 \
  --dataset narrativeqa
```

## 📁 Project Structure

```
dnsity-poc/
├── CONCEPT.md              # Research background (Korean)
├── CLAUDE.md               # Development guidelines
├── EXPERIMENT_DESIGN.md    # Detailed experiment plan
├── DATASET_RECOMMENDATIONS.md  # Dataset analysis
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── model/
│   │   ├── gist_model.py      # Model setup, token addition
│   │   └── gist_collator.py   # Custom attention masking
│   ├── data/
│   │   ├── chunking.py        # Context chunking
│   │   └── synthetic_qa.py    # Q&A generation
│   ├── training/
│   │   └── train_gist.py      # Training script
│   └── inference/
│       ├── compress.py        # KV cache compression
│       └── generate.py        # Inference with Gist
│
├── tests/
│   ├── unit/                  # Unit tests
│   └── integration/           # Integration tests
│
├── experiments/
│   ├── configs/               # Training configs
│   ├── eval_longbench.py      # LongBench evaluation
│   └── eval_baseline.py       # RAG baseline
│
├── notebooks/
│   └── gist_demo.ipynb        # Interactive demo
│
└── data/
    ├── raw/                   # Original documents
    ├── processed/             # JSONL formatted
    └── synthetic/             # Generated Q&A
```

## 🧪 Experimental Phases

Based on [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md):

1. **Phase 1**: Environment & Data Preparation (2 days)
2. **Phase 2**: Gist Token Implementation (3 days)
3. **Phase 3**: Training Experiments (3 runs × 3 hours)
4. **Phase 4**: Baseline Comparison (2 days)
5. **Phase 5**: Evaluation & Analysis (2 days)
6. **Phase 6**: KV Cache Compression (1 day)

**Total**: ~2 weeks

## 📚 Key References

- [Gist Token Paper (ACL 2023)](https://arxiv.org/abs/2304.08467)
- [Comprehensive Study (ACL 2025)](https://arxiv.org/abs/2412.17483)
- [LongBench Paper](https://arxiv.org/abs/2308.14508)

## 🔧 Development

### Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest --cov=src --cov-report=html

# Type checking
mypy src/

# Linting
ruff check .
```

### Experiment Tracking

Tag successful experiments:
```bash
git tag -a exp-gist25-v1 -m "25 tokens, 4k context, 82% accuracy"
git push origin exp-gist25-v1
```

## 📈 Expected Results (Gist-25)

| Method | VRAM | TTFT | Global Acc | Passkey Acc |
|--------|------|------|------------|-------------|
| Full Context | 18-22GB | 800-1200ms | 95% | 100% |
| RAG (Top-3) | 8-10GB | 400-600ms | 60% | 75% |
| **Gist-25** | **9-10GB** | **250-350ms** | **90%** | **80%** |

## 🤝 Contributing

This is a research PoC. For questions or collaboration:
- Open an issue
- Review [CLAUDE.md](CLAUDE.md) for development guidelines

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- Based on research by [Mu et al. (2023)](https://arxiv.org/abs/2304.08467)
- Evaluation inspired by [ACL 2025 comprehensive study](https://arxiv.org/abs/2412.17483)
- Uses [LongBench](https://github.com/THUDM/LongBench) benchmark

---

**Status**: 🚧 Documentation phase - Implementation in progress

For detailed implementation guidance, see [CLAUDE.md](CLAUDE.md).
