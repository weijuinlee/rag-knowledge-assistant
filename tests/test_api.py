from fastapi.testclient import TestClient
import pytest

from rag_assistant.api import app


def test_full_query_flow():
    client = TestClient(app)
    ingest_payload = {
        "source_id": "api-doc",
        "content": "RAG means retrieval augmented generation.",
        "chunk_size": 6,
        "chunk_overlap": 2,
    }

    response = client.post("/ingest", json=ingest_payload)
    assert response.status_code == 201

    query = client.post("/query", json={"question": "What does RAG mean?", "top_k": 3})
    assert query.status_code == 200
    data = query.json()
    assert data["count"] >= 1
    assert "RAG" in data["answer"]


def test_semantic_endpoint_with_local_provider():
    client = TestClient(app)
    client.post(
        "/ingest",
        json={
            "source_id": "semantic-doc",
            "content": "RAG means retrieval augmented generation.",
            "chunk_size": 6,
            "chunk_overlap": 2,
        },
    )

    query = client.post(
        "/query/semantic",
        json={
            "question": "What is RAG?",
            "top_k": 3,
            "embedding_provider": "local",
            "local_dimensions": 24,
        },
    )
    assert query.status_code == 200
    data = query.json()
    assert data["count"] >= 1


def test_semantic_endpoint_with_local_tfidf_provider():
    pytest.importorskip("numpy")

    client = TestClient(app)
    client.post(
        "/ingest",
        json={
            "source_id": "semantic-tfidf-doc",
            "content": "Embeddings can be generated locally without remote services.",
            "chunk_size": 8,
            "chunk_overlap": 2,
        },
    )

    query = client.post(
        "/query/semantic",
        json={
            "question": "How can embeddings be generated?",
            "top_k": 2,
            "embedding_provider": "local_tfidf",
            "local_dimensions": 64,
        },
    )
    assert query.status_code == 200
    data = query.json()
    assert data["count"] >= 1


def test_query_onnx_provider_requires_existing_model():
    pytest.importorskip("onnxruntime")

    client = TestClient(app)
    client.post(
        "/ingest",
        json={
            "source_id": "onnx-doc",
            "content": "ONNX provider path should be validated.",
            "chunk_size": 8,
            "chunk_overlap": 2,
        },
    )

    query = client.post(
        "/query/semantic",
        json={
            "question": "How does ONNX work?",
            "top_k": 2,
            "embedding_provider": "onnx_local",
            "embedding_model": "/tmp/does-not-exist/model.onnx",
        },
    )
    assert query.status_code == 503
