# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DNSity PoC** - Gist Token-based Prompt Compression Research Project

This is a research implementation of Gist Token technology for consumer GPU environments (RTX 3090/4090, 24GB VRAM). The goal is to compress long contexts (thousands of tokens) into a small number of learnable virtual tokens (10-50 tokens), enabling efficient personalized data learning on limited hardware.

### Core Technology

- **Gist Token**: Compresses long text into few special tokens via attention masking during training
- **Target Model**: Llama-3-8B-Instruct with 4-bit quantization (QLoRA)
- **Key Innovation**: Forces model to compress information through attention mask manipulation, storing context in learned token embeddings rather than full KV cache

### Architecture Philosophy

Unlike RAG (retrieval-based) or fine-tuning approaches:
- **RAG**: Searches external DB, slow due to retrieval overhead, loses global context
- **Fine-tuning**: Bakes knowledge into weights, inflexible for data updates
- **Gist Token**: Compresses context into input tokens, maintains global understanding, instant updates

## Current Status

**Repository State**: Documentation phase - CONCEPT.md contains research background, no code implemented yet.

**Next Steps**: Implement the 4-phase workflow described below.

## Development Environment

### Hardware Requirements
- NVIDIA RTX 3090/4090 (24GB VRAM minimum)
- 4-bit quantization (NF4) is mandatory to fit Llama-3-8B within memory constraints

### Setup Commands
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (create requirements.txt first)
pip install torch transformers peft bitsandbytes accelerate datasets scipy

# For development with Jupyter notebooks
pip install jupyter ipywidgets

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Python Environment
- Python 3.10+ recommended
- PyTorch with CUDA support (verify CUDA 11.8+ or 12.1+)
- Transformers library (Hugging Face) >= 4.36.0
- QLoRA via bitsandbytes for 4-bit quantization

## Key Implementation Components

### 1. Gist Token Architecture

**Token Addition Strategy**:
- Add special tokens `<GIST_0>` through `<GIST_N>` to tokenizer vocabulary
- Resize model embedding layer: `model.resize_token_embeddings(len(tokenizer))`
- Newly added embeddings are randomly initialized and trained

**Critical Configuration**:
```python
# When using LoRA, MUST include:
modules_to_save=["embed_tokens", "lm_head"]
```
This ensures Gist token embeddings are trainable (not frozen), allowing them to learn compressed representations.

### 2. Attention Mask Manipulation (Core Mechanism)

The training forces compression through attention masking:

**Input Structure**: `[Instruction] + [Context] + [<GIST_0>...<GIST_N>] + [Question] + [Answer]`

**Masking Rules**:
- Gist tokens CAN attend to Context (absorb information)
- Question/Answer tokens CANNOT attend to Context (blocked via mask)
- Question/Answer tokens CAN ONLY attend to Gist tokens

**Implementation**: Custom data collator modifies causal mask:
```python
# Block Question region from seeing Context region
mask[i, :, gist_end:, :gist_start] = False
```

This bottleneck forces the model to compress all necessary information into Gist token embeddings during training, as they are the only pathway from Context to Answer.

### 3. KV Cache Compression (Inference)

**Memory Saving Strategy**:
1. Forward pass with `[Context] + [Gist Tokens]`
2. Extract `past_key_values` (KV cache)
3. **Slice away Context portion**: Keep only Gist token KV states
4. Use compressed KV for subsequent generation

**Result**: 4000 tokens → 10 Gist tokens = 400x memory reduction

### 4. Data Pipeline Requirements

**Training Data Structure** (JSONL format):
- **Instruction**: Explicit compression directive ("Compress this text using Gist tokens...")
- **Context**: 2k-8k token chunks of user data (documents, code, etc.)
- **Gist Tokens**: Placeholder special tokens
- **Question**: Questions requiring context understanding
- **Answer**: Ground truth requiring compressed information

**Synthetic Data Generation**:
- Use teacher model (GPT-4, Llama-3-70B) to generate Q&A pairs from chunks
- Question types must span: factual lookup, global themes, logical reasoning
- Filter out questions answerable without context (common knowledge)

## Implementation Workflow

### Phase 1: Model Preparation
1. Load Llama-3-8B with 4-bit quantization (`BitsAndBytesConfig`)
2. Add N Gist special tokens to tokenizer
3. Resize embedding layer to accommodate new tokens
4. Apply LoRA with `embed_tokens` in `modules_to_save`

### Phase 2: Data Preparation
1. Chunk personal data into 2-8k token segments
2. Generate Q&A pairs using teacher model
3. Format as Instruction-Context-Gist-QA structure
4. Tokenize and store Gist token position indices

### Phase 3: Training
1. Implement `GistDataCollator` with custom attention mask logic
2. Train using standard Hugging Face Trainer
3. Monitor: Gist embedding gradients (should be non-zero), loss convergence

### Phase 4: Inference & Caching
1. Compress contexts: Forward pass → extract KV cache → slice to Gist portion
2. Store compressed KV per document/user
3. Generate answers by prepending compressed KV to new questions

## Critical Implementation Details

### Attention Implementation Compatibility
- **Flash Attention 2**: May have limited custom mask support - verify or fallback to SDPA
- Use `attn_implementation="sdpa"` (Scaled Dot Product Attention) for maximum control

### Memory Management
- Model weights (4-bit): ~5-6GB
- Reserve remaining VRAM for gradients, activations, and KV cache
- Batch size will be heavily constrained (likely 1-2 on 24GB)

### Training Stability
- Use learning rate warmup to prevent Gist embedding divergence
- Monitor if model "ignores" Gist tokens (symptom: poor validation despite low training loss)
- Gist token count is a hyperparameter: balance compression vs information retention

## Evaluation Strategy

### Quantitative Metrics
- **Compression Ratio**: Original tokens / Gist tokens (target: 100-400x)
- **Passkey Retrieval**: Hide random numbers in context, measure recall post-compression
- **VRAM Usage**: `torch.cuda.max_memory_allocated()` during inference
- **Throughput**: Tokens/sec generation speed

### Qualitative Metrics
- **Global Context Understanding**: Questions about overall themes/tone
- **Hallucination Rate**: Fabricated facts from lossy compression

### Baseline Comparisons
- Full Context (upper bound quality, high memory)
- RAG with ChromaDB (fragmented context, search overhead)
- Gist Token (compressed context, memory efficient)

## Known Challenges & Mitigations

**Challenge**: Information Loss (Lossy Compression)
- **Mitigation**: Tune Gist token count; hybrid with RAG for precise lookups

**Challenge**: Training Instability
- **Mitigation**: Lower LoRA learning rate, increase warmup steps, verify gradient flow

**Challenge**: Model Ignoring Gist Tokens
- **Mitigation**: Ensure attention mask is correct, verify Context is truly blocked during Q&A

## Future Directions

- **Hierarchical Gist**: Multi-level compression for book-length documents
- **Multimodal Gist**: Compress image+text into unified Gist representations
- **Hybrid RAG+Gist**: Use RAG to retrieve, Gist to remember retrieved chunks

## Technical Context for Debugging

### When Gist Isn't Working
1. Check attention mask shape: `[batch, 1, seq_len, seq_len]`
2. Verify Gist token positions are correctly identified in collator
3. Confirm `embed_tokens` has gradients: `model.base_model.model.model.embed_tokens.weight.requires_grad`
4. Inspect KV cache slicing: ensure context portion is actually removed

### Memory Debugging
```python
import torch
print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
torch.cuda.reset_peak_memory_stats()  # Reset before inference test
```

### Recommended File Structure
```
dnsity-poc/
├── CONCEPT.md              # Research background (existing)
├── CLAUDE.md               # This file
├── README.md               # Project overview and setup instructions
├── requirements.txt        # Python dependencies
├── .gitignore              # Exclude venv/, *.pyc, __pycache__, checkpoints/, etc.
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── gist_model.py      # Model setup, token addition, LoRA config
│   │   └── gist_collator.py   # Custom DataCollator with attention masking
│   ├── data/
│   │   ├── __init__.py
│   │   ├── chunking.py        # Context chunking logic
│   │   └── synthetic_qa.py    # Q&A generation with teacher model
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_gist.py      # Training script with Trainer
│   └── inference/
│       ├── __init__.py
│       ├── compress.py        # Context → Gist KV compression
│       └── generate.py        # Inference with compressed KV
├── experiments/
│   ├── configs/               # Training configs (YAML/JSON)
│   ├── eval_metrics.py        # Evaluation scripts
│   └── run_baseline.py        # RAG baseline comparison
├── notebooks/
│   └── gist_demo.ipynb        # Interactive demo and exploration
├── data/                      # Data directory (gitignored except .gitkeep)
│   ├── raw/                   # Original documents
│   ├── processed/             # Chunked and formatted JSONL
│   └── synthetic/             # Generated Q&A pairs
└── checkpoints/               # Model checkpoints (gitignored)
```

### Common Commands (Once Implemented)

```bash
# Training
python src/training/train_gist.py \
  --config experiments/configs/gist_10tokens.yaml \
  --output_dir checkpoints/gist-10

# Single module test (example)
python -m pytest tests/test_gist_collator.py -v

# Evaluation
python experiments/eval_metrics.py \
  --checkpoint checkpoints/gist-10 \
  --test_data data/processed/test.jsonl

# Interactive inference (Jupyter)
jupyter notebook notebooks/gist_demo.ipynb
```

## Development Best Practices

### Version Control
- This is a research project - commit frequently at each experimental milestone
- Tag successful experiments: `git tag -a exp-gist10-v1 -m "10 tokens, 2k context"`
- Use `.gitignore` for: `venv/`, `*.pyc`, `__pycache__/`, `checkpoints/`, `data/raw/`, `.env`

### Experiment Tracking
- Log hyperparameters, metrics, and observations for each training run
- Consider using Weights & Biases or simple CSV logs
- Save attention mask visualizations to debug compression behavior

### Code Organization Principles
- Keep model code separate from training logic
- Make Gist token count configurable (command-line arg or config file)
- Abstract attention masking logic for easy debugging/modification

### Memory Safety
- Always wrap training in try-catch with `torch.cuda.empty_cache()` in finally block
- Use gradient checkpointing if batch size > 1 causes OOM
- Monitor VRAM with `nvidia-smi` during development

## Notes on CONCEPT.md

The research document (CONCEPT.md) is written in Korean and provides comprehensive theoretical background. Key terms:
- "가정용 GPU" = Consumer GPU
- "문맥 압축" = Context Compression
- "검색 증강 생성" = Retrieval-Augmented Generation (RAG)
- "주의 마스킹" = Attention Masking
- Sections 4.2-4.5 contain detailed implementation code examples in Korean comments
