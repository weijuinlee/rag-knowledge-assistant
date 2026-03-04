from __future__ import annotations

import importlib
import json
import sys

import pytest
from fastapi.testclient import TestClient

from rag_assistant.knowledge_base import KnowledgeBase, KnowledgeBaseStorageError


def test_persistence_save_and_reload(tmp_path):
    storage_path = tmp_path / "index.json"
    kb = KnowledgeBase(storage_path=str(storage_path))
    kb.ingest(
        "semantic-doc",
        "RAG combines retrieval with generation.",
        metadata={"source": "unit-test"},
    )

    assert storage_path.exists()
    raw = json.loads(storage_path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert raw["chunks"][0]["source_id"] == "semantic-doc"

    reloaded = KnowledgeBase(storage_path=str(storage_path))
    result = reloaded.query("What is RAG?", top_k=1)

    assert result
    assert result[0].source_id == "semantic-doc"
    assert result[0].metadata["source"] == "unit-test"


def test_persistence_probe(tmp_path):
    storage_path = tmp_path / "index.json"
    kb = KnowledgeBase(storage_path=str(storage_path))
    kb.ingest("probe-doc", "Probe index persistence health check.")

    assert kb.storage_probe() is True


def test_persistence_invalid_file_raises(tmp_path):
    storage_path = tmp_path / "bad.json"
    storage_path.write_text("{not-json}", encoding="utf-8")

    with pytest.raises(KnowledgeBaseStorageError):
        KnowledgeBase(storage_path=str(storage_path))


@pytest.fixture
def api_with_api_key_and_rate_limit(monkeypatch):
    monkeypatch.setenv("RAG_API_KEY", "dev-token")
    monkeypatch.setenv("RAG_RATE_LIMIT_REQUESTS", "1")
    monkeypatch.setenv("RAG_RATE_LIMIT_WINDOW_SECONDS", "60")

    sys.modules.pop("rag_assistant.api", None)
    importlib.invalidate_caches()
    api_module = importlib.import_module("rag_assistant.api")

    yield api_module


def test_api_key_enforcement(api_with_api_key_and_rate_limit):
    app = api_with_api_key_and_rate_limit.app
    client = TestClient(app)

    unauthorized = client.post("/query", json={"question": "What is RAG?"})
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error_code"] == "unauthorized"

    unauthorized = client.post(
        "/query",
        json={"question": "What is RAG?"},
        headers={"x-api-key": "wrong"},
    )
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error_code"] == "unauthorized"

    authorized = client.post(
        "/query",
        json={"question": "What is RAG?"},
        headers={"x-api-key": "dev-token"},
    )
    assert authorized.status_code in {404, 200}


def test_rate_limit_middleware_blocks_excess_calls(api_with_api_key_and_rate_limit):
    app = api_with_api_key_and_rate_limit.app
    client = TestClient(app)
    headers = {"x-api-key": "dev-token"}

    first = client.post("/query", json={"question": "What is RAG?"}, headers=headers)
    assert first.status_code in {404, 200}

    second = client.post("/query", json={"question": "What is RAG?"}, headers=headers)
    assert second.status_code == 429
    assert second.json()["error_code"] == "rate_limit_exceeded"
