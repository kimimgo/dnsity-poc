"""
Unit tests for RAG (Retrieval-Augmented Generation) pipeline.

Tests verify:
1. Document chunking and embedding
2. Vector similarity search (retrieval)
3. End-to-end RAG inference
"""

import pytest
from pathlib import Path
import tempfile


class TestRAGPipeline:
    """Test suite for RAG baseline."""

    def test_rag_initialization(self):
        """Test RAG pipeline initialization."""
        from src.baseline.rag_pipeline import RAGPipeline
        
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = RAGPipeline(persist_dir=tmpdir)
            
            assert rag.embedder is not None
            assert rag.collection is not None

    def test_add_documents(self):
        """Test adding documents to RAG."""
        from src.baseline.rag_pipeline import RAGPipeline
        
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = RAGPipeline(persist_dir=tmpdir)
            
            documents = [
                "The Eiffel Tower is in Paris.",
                "Paris is the capital of France.",
                "The Louvre Museum is located in Paris."
            ]
            
            rag.add_documents(documents)
            
            # Verify documents were added
            count = rag.collection.count()
            assert count > 0

    def test_retrieval_correctness(self):
        """Test that retrieval returns relevant documents."""
        from src.baseline.rag_pipeline import RAGPipeline
        
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = RAGPipeline(persist_dir=tmpdir)
            
            documents = [
                "The capital of France is Paris.",
                "Berlin is the capital of Germany.",
                "Tokyo is the capital of Japan."
            ]
            
            rag.add_documents(documents)
            
            # Query about France
            results = rag.retrieve("What is the capital of France?", top_k=1)
            
            assert len(results) > 0
            assert "Paris" in results[0] or "France" in results[0]

    def test_top_k_retrieval(self):
        """Test top-k retrieval returns k results."""
        from src.baseline.rag_pipeline import RAGPipeline
        
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = RAGPipeline(persist_dir=tmpdir)
            
            documents = [
                "Document 1 about AI",
                "Document 2 about machine learning",
                "Document 3 about deep learning",
                "Document 4 about neural networks",
                "Document 5 about transformers"
            ]
            
            rag.add_documents(documents)
            
            results = rag.retrieve("Tell me about AI", top_k=3)
            
            assert len(results) <= 3

    def test_rag_end_to_end(self):
        """Test end-to-end RAG inference."""
        from src.baseline.rag_pipeline import RAGPipeline
        
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = RAGPipeline(persist_dir=tmpdir, model_name="gpt2")
            
            documents = [
                "The Eiffel Tower was built in 1889.",
                "It is located in Paris, France.",
                "The tower is 330 meters tall."
            ]
            
            rag.add_documents(documents)
            
            answer = rag.inference(
                question="When was the Eiffel Tower built?",
                top_k=2,
                max_new_tokens=20
            )
            
            assert isinstance(answer, str)
            assert len(answer) > 0

    def test_empty_retrieval(self):
        """Test retrieval with no documents."""
        from src.baseline.rag_pipeline import RAGPipeline
        
        with tempfile.TemporaryDirectory() as tmpdir:
            rag = RAGPipeline(persist_dir=tmpdir)
            
            results = rag.retrieve("Some query", top_k=3)
            
            assert isinstance(results, list)
            assert len(results) == 0
