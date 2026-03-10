from __future__ import annotations

import importlib
import sys

from fastapi.testclient import TestClient
import pytest


@pytest.fixture
def fresh_client(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_STORAGE_PATH", str(tmp_path / "index.json"))
    monkeypatch.delenv("RAG_API_KEY", raising=False)

    sys.modules.pop("rag_assistant.api", None)
    importlib.invalidate_caches()
    api_module = importlib.import_module("rag_assistant.api")

    with TestClient(api_module.app) as client:
        yield client


def test_health_and_healthz_smoke(fresh_client):
    health = fresh_client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["status"] == "ok"
    assert "version" in payload
    assert "storage" in payload
    assert "uptime_seconds" in payload

    healthz = fresh_client.get("/healthz")
    assert healthz.status_code == 200
    healthz_payload = healthz.json()
    assert healthz_payload["status"] == "ok"
    assert healthz_payload["version"] == payload["version"]


def test_query_returns_404_without_documents(fresh_client):
    response = fresh_client.post("/query", json={"question": "What is RAG?", "top_k": 3})
    assert response.status_code == 404
    payload = response.json()
    assert payload["error_code"] == "not_found"
    assert payload["path"] == "/query"


def test_query_returns_result_for_ingested_content(fresh_client):
    ingest = fresh_client.post(
        "/ingest",
        json={
            "source_id": "smoke",
            "content": "RAG means Retrieval-Augmented Generation.",
            "chunk_size": 8,
            "chunk_overlap": 2,
            "chunking_strategy": "smart",
        },
    )
    assert ingest.status_code == 201

    query = fresh_client.post(
        "/query",
        json={
            "question": "What does RAG mean?",
            "top_k": 3,
            "retrieval": "hybrid",
            "embedding_provider": "local",
            "reranker": "term_overlap",
            "candidate_pool_size": 5,
        },
    )
    assert query.status_code == 200
    payload = query.json()
    assert payload["count"] >= 1
    assert "RAG" in payload["answer"]
    assert payload["trace"]["retrieval"] == "hybrid"


def test_eval_endpoint_reports_metrics(fresh_client):
    ingest = fresh_client.post(
        "/ingest",
        json={
            "source_id": "eval-doc",
            "content": "RAG retrieves context before generating an answer.",
            "chunk_size": 8,
            "chunk_overlap": 2,
        },
    )
    assert ingest.status_code == 201

    response = fresh_client.post(
        "/evals/run",
        json={
            "cases": [
                {
                    "question": "What retrieves context before generating an answer?",
                    "expected_source_ids": ["eval-doc"],
                    "retrieval": "hybrid",
                    "embedding_provider": "local",
                    "reranker": "term_overlap",
                }
            ]
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["cases"] == 1
    assert payload["hit_rate"] == 1.0
