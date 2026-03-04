# RAG Knowledge Assistant

`rag-knowledge-assistant` is a minimal open-source retrieval API for experimentation and production-ready prototyping.

The API supports lexical TF-IDF retrieval and optional semantic retrieval using pluggable embedding providers.

---

## Current status

- Version: `0.2.0`
- Storage can be run in-memory (default) or with file-backed persistence.
- Auth, rate limiting, request validation responses, metrics, and health endpoints included.

---

## Features

- Ingest and chunk documents via HTTP.
- TF-IDF and semantic retrieval endpoints.
- File-backed persistence with resumable startup index loading.
- Pluggable semantic providers (`sentence_transformers`, `local`, `local_tfidf`, `onnx_local`).
- Config/env-first runtime configuration.
- Optional static API key middleware.
- Input abuse controls (payload size), request ID headers, and rate limiting.
- Structured error payloads and OpenAPI-documented failure modes.
- Observability endpoint (`/metrics`) plus health and readiness checks.
- Docker/Compose defaults for durable storage mount.

---

## Environment configuration

The app reads settings from environment variables, including:

- `RAG_SERVICE_NAME`
- `RAG_STORAGE_PATH` (optional; when set, chunks are persisted to disk)
- `RAG_API_KEY` (optional static API key)
- `RAG_RATE_LIMIT_REQUESTS`
- `RAG_RATE_LIMIT_WINDOW_SECONDS`
- `RAG_MAX_REQUEST_BYTES`
- `RAG_DEFAULT_CHUNK_SIZE`
- `RAG_DEFAULT_CHUNK_OVERLAP`
- `RAG_DEFAULT_EMBEDDING_MODEL`
- `RAG_HOST`
- `RAG_PORT`
- `RAG_RELOAD`
- `RAG_LOG_LEVEL`
- `RAG_DOCS_ENABLED`
- `RAG_ALLOWED_ORIGINS`

See `.env.example` for a complete starter configuration.

---

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For semantic dependencies:

```bash
pip install -e ".[dev,embeddings]"
```

For ONNX support:

```bash
pip install -e ".[dev,onnx]"
```

### 2) Run API

```bash
cp .env.example .env
uvicorn rag_assistant.api:app --reload --app-dir src
```

### 2a) Smoke check (persistence + API key)

```bash
RAG_API_KEY=dev RAG_STORAGE_PATH=./data/rag-index.json uvicorn rag_assistant.api:app --reload --app-dir src
```

### 3) Use the built-in UI

```text
http://localhost:8000/ui/
```

The page can ingest content and run both TF-IDF and semantic queries.
If you configured an API key, paste it into the UI.

### 4) Or run with CLI helper

```bash
python -m rag_assistant.cli
```

### 5) Ingest

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${RAG_API_KEY}" \
  -d '{"source_id":"guide-1","content":"RAG combines retrieval with generation."}'
```

### 6) Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${RAG_API_KEY}" \
  -d '{"question":"What is RAG?","top_k":3}'
```

### Semantic query

```bash
curl -X POST http://localhost:8000/query/semantic \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${RAG_API_KEY}" \
  -d '{"question":"What is RAG?","top_k":3,"embedding_provider":"sentence_transformers","embedding_model":"sentence-transformers/all-MiniLM-L6-v2"}'
```

Local TF-IDF alternative:

```bash
curl -X POST http://localhost:8000/query/semantic \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${RAG_API_KEY}" \
  -d '{"question":"How can embeddings be generated?","top_k":3,"embedding_provider":"local_tfidf","local_dimensions":64}'
```

### 7) Optional: tests

```bash
pytest
```

---

## API reference

- `GET /health` — service health and runtime summary
- `GET /ready` — readiness probe (checks storage availability when persistence is enabled)
- `GET /stats` — index stats and provider/runtime metadata
- `GET /metrics` — in-memory request metrics
- `POST /ingest` — ingest one document
- `POST /ingest/bulk` — ingest many documents
- `POST /query` — retrieve top-k relevant chunks
- `POST /query/semantic` — semantic-only query endpoint
- `DELETE /documents/{source_id}` — remove all chunks for a source
- `DELETE /clear` — clear index

Docs UI: `http://localhost:8000/docs`

---

## Persistence notes

- In-memory mode remains the default for quick experiments.
- Set `RAG_STORAGE_PATH` to enable persistent index writes and reload on restart.
- Current persistence format is a local JSON snapshot of chunks and metadata.

---

## Deployment

- For API and UI together, use the Render blueprint in `render.yaml`.
- Backend (`rag-knowledge-assistant-api`) exposes `http://...` and serves `/healthz` for readiness probes.
- UI (`rag-knowledge-assistant-ui`) publishes a static site with `apiBaseUrl` injected at build time.
- Set backend environment variables in Render:
  - `RAG_ALLOWED_ORIGINS`: frontend origin
  - `RAG_GITHUB_PAGES_ORIGIN` (legacy alias): frontend origin
  - `RAG_API_KEY`: secret value if API key is enabled
  - `RAG_STORAGE_PATH`: persistent path, such as `/var/data/rag-index.json`
- Set frontend environment variable in Render:
  - `VITE_API_BASE_URL`: backend public URL (for example `https://rag-knowledge-assistant-api.onrender.com`)

---

## Contributing

- See `CONTRIBUTING.md`.
- For planned work, see `ROADMAP.md`.
- For release history, see `CHANGELOG.md`.

## License

This project is MIT-licensed. See [LICENSE](LICENSE).
