from rag_assistant.knowledge_base import KnowledgeBase
import pytest
from rag_assistant.evals import run_retrieval_eval


def test_ingest_and_query_returns_expected_chunks():
    kb = KnowledgeBase(chunk_size=4, chunk_overlap=1)
    kb.ingest("doc-1", "RAG combines retrieval with generation. Retrieval narrows context.")

    result = kb.query("What combines retrieval and generation?", top_k=3)

    assert result
    assert result[0].source_id == "doc-1"
    assert "retrieval" in result[0].text.lower()


def test_remove_source_reduces_index():
    kb = KnowledgeBase()
    kb.ingest("doc-a", "small piece of text")
    kb.ingest("doc-b", "another small piece")

    removed = kb.remove_source("doc-a")

    assert removed >= 1
    assert kb.stats()["documents"] == 1


def test_query_semantic_local_provider():
    kb = KnowledgeBase()
    kb.ingest("semantic-doc", "RAG means retrieval augmented generation.")

    result = kb.query("What is RAG?", retrieval="semantic", embedding_provider="local")

    assert result
    assert result[0].source_id == "semantic-doc"


def test_query_semantic_local_tfidf_provider():
    pytest.importorskip("numpy")

    kb = KnowledgeBase()
    kb.ingest("semantic-tfidf-doc", "Embeddings can be generated locally with Numpy.")

    result = kb.query(
        "How can embeddings be generated?",
        retrieval="semantic",
        embedding_provider="local_tfidf",
    )

    assert result
    assert result[0].source_id == "semantic-tfidf-doc"


def test_query_semantic_onnx_provider_missing_model():
    pytest.importorskip("onnxruntime")

    kb = KnowledgeBase()
    kb.ingest("onnx-doc", "ONNX provider path should be validated before query.")

    try:
        kb.query(
            "What is ONNX?",
            retrieval="semantic",
            embedding_provider="onnx_local",
            embedding_model="/tmp/does-not-exist/model.onnx",
        )
    except RuntimeError as exc:
        assert "ONNX model path does not exist" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when ONNX path is invalid")


def test_smart_chunking_preserves_sentence_boundaries():
    kb = KnowledgeBase(chunk_size=8, chunk_overlap=2)
    indexed = kb.ingest(
        "smart-doc",
        "RAG improves answers. It retrieves relevant context first.\n\nChunking should respect paragraphs and sentences.",
        chunking_strategy="smart",
    )

    assert indexed >= 2
    result = kb.query("How does RAG improve answers?", top_k=2)
    assert result
    assert "." in result[0].text
    assert result[0].metadata["chunking_strategy"] == "smart"


def test_hybrid_query_with_reranker_returns_traceable_results():
    kb = KnowledgeBase(chunk_size=6, chunk_overlap=1)
    kb.ingest("alpha", "RAG systems retrieve grounded context before generating answers.")
    kb.ingest("beta", "Databases store rows and tables for transactional workloads.")

    results, trace = kb.query_with_trace(
        "Which system retrieves grounded context?",
        retrieval="hybrid",
        embedding_provider="local",
        reranker="term_overlap",
        top_k=2,
        candidate_pool_size=4,
    )

    assert results
    assert results[0].source_id == "alpha"
    assert trace["retrieval"] == "hybrid"
    assert trace["reranker"] == "term_overlap"
    assert any(stage["stage"] == "rrf_fusion" for stage in trace["stages"])


def test_eval_harness_reports_hit_rate_and_mrr():
    kb = KnowledgeBase(chunk_size=8, chunk_overlap=2)
    kb.ingest("guide", "RAG uses retrieval to ground generation in retrieved documents.")
    kb.ingest("cookbook", "Recipes explain ingredients, temperatures, and timing.")

    report = run_retrieval_eval(
        kb,
        [
            {
                "question": "What grounds generation in documents?",
                "expected_source_ids": ["guide"],
                "retrieval": "hybrid",
                "embedding_provider": "local",
                "reranker": "term_overlap",
                "top_k": 3,
                "candidate_pool_size": 6,
            }
        ],
    )

    assert report["cases"] == 1
    assert report["hits"] == 1
    assert report["hit_rate"] == 1.0
    assert report["mrr"] == 1.0
