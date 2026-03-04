# Roadmap

## Near term

- Add structured persistence migration from file-backed chunks to optional SQLite/PostgreSQL backends.
- Add OpenTelemetry tracing and Prometheus metric export.
- Add admin endpoints to clear per source and inspect chunk IDs safely.
- Add auth strategies beyond static API key (OAuth2/JWT).

## Mid term

- Add LLM-backed answer synthesis with provider abstraction.
- Add pluggable rerankers and metadata filters in query requests.
- Add optional async/background ingestion path for large payloads.

## Long term

- Add multi-tenant support for source-scoped credentials.
- Add sharded vector index support and export/import tooling.
