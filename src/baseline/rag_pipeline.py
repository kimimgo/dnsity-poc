"""
RAG (Retrieval-Augmented Generation) Pipeline implementation.

Uses ChromaDB for vector storage and sentence-transformers for embedding.
"""

from typing import Optional, List
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class RAGPipeline:
    """
    RAG baseline using vector similarity search.
    
    Retrieves relevant chunks from a vector database before generation.
    """
    
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        model_name: str = "gpt2",
        embedder_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "documents"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            persist_dir: Directory to persist ChromaDB
            model_name: LLM for generation
            embedder_name: Sentence transformer for embeddings
            collection_name: ChromaDB collection name
        """
        # Initialize ChromaDB
        if persist_dir:
            self.client = chromadb.PersistentClient(path=persist_dir)
        else:
            self.client = chromadb.Client()
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(name=collection_name)
        
        # Initialize embedder
        self.embedder = SentenceTransformer(embedder_name)
        
        # Initialize LLM (optional, for end-to-end inference)
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
    def _load_model(self):
        """Lazy load generation model."""
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def add_documents(self, documents: List[str], chunk_size: int = 512):
        """
        Add documents to RAG vector store.
        
        Args:
            documents: List of document texts
            chunk_size: Maximum chunk size (not used in simple implementation)
        """
        if not documents:
            return
        
        # Generate embeddings
        embeddings = self.embedder.encode(documents, convert_to_numpy=True)
        
        # Add to ChromaDB
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            ids=ids
        )
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve top-k relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of retrieved document texts
        """
        # Check if collection is empty
        if self.collection.count() == 0:
            return []
        
        # Embed query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        
        # Search
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(top_k, self.collection.count())
        )
        
        # Extract documents
        if results and "documents" in results and results["documents"]:
            return results["documents"][0]
        
        return []
    
    def inference(
        self,
        question: str,
        top_k: int = 3,
        max_new_tokens: int = 50,
        temperature: float = 0.7
    ) -> str:
        """
        End-to-end RAG inference.
        
        Args:
            question: Question to answer
            top_k: Number of documents to retrieve
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated answer
        """
        # Lazy load model
        self._load_model()
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question, top_k=top_k)
        
        # Build prompt with retrieved context
        if retrieved_docs:
            context = "\n".join(retrieved_docs)
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        # Generate answer
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode only new tokens
        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return answer.strip()
