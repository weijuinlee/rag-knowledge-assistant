from rag_assistant.knowledge_base import KnowledgeBase
import pytest


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
