"""
RAG 기반 NoLiMa 평가
ChromaDB + BGE-M3 임베딩 사용
"""
import json
import torch
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm

# ChromaDB 및 임베딩
try:
    import chromadb
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: chromadb not installed. Run: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Run: pip install sentence-transformers")


class RAGEvaluator:
    def __init__(self, embedding_model="BAAI/bge-m3", chunk_size=512, chunk_overlap=50, top_k=5):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        # 임베딩 모델 로드
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model, device="cuda" if torch.cuda.is_available() else "cpu")

        # ChromaDB 초기화
        self.client = chromadb.Client()
        self.collection = None

    def chunk_text(self, text: str) -> list:
        """텍스트를 청크로 분할"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def index_document(self, context: str, doc_id: str):
        """문서 인덱싱"""
        # 기존 컬렉션 삭제 후 재생성
        try:
            self.client.delete_collection("temp_doc")
        except:
            pass

        self.collection = self.client.create_collection(
            name="temp_doc",
            metadata={"hnsw:space": "cosine"}
        )

        # 청킹
        chunks = self.chunk_text(context)

        # 임베딩 생성
        embeddings = self.embedder.encode(chunks, show_progress_bar=False).tolist()

        # 인덱싱
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=[f"{doc_id}_{i}" for i in range(len(chunks))]
        )

        return len(chunks)

    def retrieve(self, query: str) -> list:
        """관련 청크 검색"""
        query_embedding = self.embedder.encode([query], show_progress_bar=False).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=self.top_k
        )

        return results["documents"][0] if results["documents"] else []


def load_llm(model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    """LLM 로드"""
    print(f"Loading LLM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True
    )

    return model, tokenizer


def generate_answer(model, tokenizer, context_chunks: list, question: str):
    """RAG 기반 답변 생성"""
    context = "\n\n".join(context_chunks)

    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer (provide only the code/identifier mentioned):"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # 6자리 코드 추출
    matches = re.findall(r'\b[A-Z0-9]{6}\b', generated.upper())
    predicted = matches[0] if matches else ""

    return predicted, generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/nolima/nolima_200.jsonl")
    parser.add_argument("--output", default="results/rag_nolima.json")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    if not CHROMADB_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("Required packages not installed. Installing...")
        import subprocess
        subprocess.run(["pip", "install", "chromadb", "sentence-transformers", "-q"])
        print("Please restart the script after installation.")
        return

    # RAG 초기화
    rag = RAGEvaluator(
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        top_k=args.top_k
    )

    # LLM 로드
    model, tokenizer = load_llm()

    # 데이터 로드
    print(f"Loading {args.dataset}")
    with open(args.dataset) as f:
        samples = [json.loads(line) for line in f]

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Evaluating {len(samples)} samples...")

    # 평가
    results = []
    retrieval_hits = 0

    for i, sample in enumerate(tqdm(samples)):
        try:
            # 1. 인덱싱
            num_chunks = rag.index_document(sample["context"], f"doc_{i}")

            # 2. 검색
            retrieved_chunks = rag.retrieve(sample["question"])

            # 3. Needle이 검색되었는지 확인
            needle = sample.get("answer", "")
            needle_in_retrieved = any(needle in chunk for chunk in retrieved_chunks)
            if needle_in_retrieved:
                retrieval_hits += 1

            # 4. 답변 생성
            predicted, generated = generate_answer(model, tokenizer, retrieved_chunks, sample["question"])

            # 5. 평가
            ground_truth = sample["answer"].upper()
            correct = predicted == ground_truth

            results.append({
                "predicted": predicted,
                "ground_truth": ground_truth,
                "correct": correct,
                "retrieval_hit": needle_in_retrieved,
                "num_chunks": num_chunks,
                "generated": generated[:100],
                "question": sample["question"]
            })

        except Exception as e:
            print(f"\nError on sample {i}: {e}")
            results.append({
                "predicted": "",
                "ground_truth": sample["answer"],
                "correct": False,
                "retrieval_hit": False,
                "error": str(e)
            })

    # 결과 집계
    correct_count = sum(r["correct"] for r in results)
    accuracy = correct_count / len(results) * 100
    retrieval_recall = retrieval_hits / len(results) * 100

    print("\n" + "=" * 70)
    print("RAG - NoLiMa EVALUATION RESULTS")
    print("=" * 70)
    print(f"Accuracy:         {accuracy:.2f}% ({correct_count}/{len(results)})")
    print(f"Retrieval Recall: {retrieval_recall:.2f}% ({retrieval_hits}/{len(results)})")
    print(f"Chunk Size:       {args.chunk_size}")
    print(f"Top-K:            {args.top_k}")
    print("=" * 70)

    # 오답 분석
    print("\nIncorrect samples with retrieval hit (first 5):")
    incorrect_with_hit = [r for r in results if not r["correct"] and r.get("retrieval_hit")][:5]
    for i, r in enumerate(incorrect_with_hit):
        print(f"  [{i+1}] Q: {r.get('question', '')[:60]}...")
        print(f"      Expected: {r['ground_truth']}, Got: {r['predicted']}")

    print("\nRetrieval failures (first 5):")
    no_retrieval = [r for r in results if not r.get("retrieval_hit")][:5]
    for i, r in enumerate(no_retrieval):
        print(f"  [{i+1}] Q: {r.get('question', '')[:60]}...")

    # 결과 저장
    output_data = {
        "dataset": args.dataset,
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "top_k": args.top_k,
        "accuracy": accuracy,
        "retrieval_recall": retrieval_recall,
        "correct": correct_count,
        "retrieval_hits": retrieval_hits,
        "total": len(results),
        "results": results
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
