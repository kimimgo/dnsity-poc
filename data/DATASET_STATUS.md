# Dataset Status

Generated datasets for DNSity PoC Phase 1 evaluation.

## NIAH (Needle in Haystack)

**Location**: `data/processed/niah/test_samples.jsonl`

- **Samples**: 100
- **Average context length**: ~5080 tokens (20,320 chars)
- **Context range**: 2,000-8,000 tokens
- **Needle positions**: Randomized (20%-80% through context)
- **Format**: JSONL with fields: `context`, `question`, `answer`, `needle`, `needle_position`, `context_length_chars`

**Purpose**: Evaluate compression information loss via passkey retrieval accuracy.

## LongBench

**Location**: `data/processed/longbench/`

### Tasks

1. **narrativeqa** (50 samples, 5.1 MB)
   - Long narrative comprehension
   - Questions about book-length stories

2. **qasper** (50 samples, 1.3 MB)
   - Scientific paper QA
   - Multi-hop reasoning over research papers

3. **gov_report** (50 samples, 3.0 MB)
   - Government report summarization
   - Global context understanding

**Total**: 150 samples, 9.4 MB

**Purpose**: Evaluate global context understanding and multi-hop reasoning with compressed representations.

## Generation Commands

```bash
# NIAH
python3 src/data/create_niah.py --samples 100 --output test_samples.jsonl

# LongBench
python3 src/data/download_longbench.py \
  --tasks narrativeqa qasper gov_report \
  --max-samples 50 \
  --output-dir data/processed/longbench
```

## Next Steps (Phase 2)

These datasets will be used to evaluate:
1. **Passkey Retrieval**: NIAH exact match accuracy
2. **Global Understanding**: LongBench narrative and summarization tasks
3. **Multi-hop Reasoning**: LongBench qasper scientific QA

Expected compression ratio: 100-400x (e.g., 5000 tokens → 10-50 Gist tokens)
