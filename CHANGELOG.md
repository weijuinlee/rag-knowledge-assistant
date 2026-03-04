# Changelog

## [0.2.0] - 2026-03-04

- Added persistence support with file-backed storage for indexed chunks.
- Added config/env loading (`AppConfig`) for deployment settings.
- Added optional static API key auth and rate limiting middleware.
- Added input abuse controls via request size checks.
- Added structured error responses and global error handlers.
- Added observability via request metrics, request ID headers, and `/metrics`.
- Added improved health endpoints: `/health` and `/ready`.
- Hardened local embedding providers (`local_tfidf`, `onnx_local`).

## [0.1.0]

- Initial minimal RAG retrieval implementation.
- In-memory storage and TF-IDF/semantic query baseline.
- FastAPI endpoints for ingest, semantic query, and chunk stats.
