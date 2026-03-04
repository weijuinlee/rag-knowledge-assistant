from __future__ import annotations

import logging
import time
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, Deque

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from rag_assistant import __version__
from rag_assistant.config import AppConfig
from rag_assistant.knowledge_base import KnowledgeBase
from rag_assistant.models import (
    BulkIngestRequest,
    BulkIngestResponse,
    ContextChunk,
    DeleteResponse,
    ErrorResponse,
    IngestRequest,
    IngestResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    RouteMetrics,
    SemanticQueryRequest,
)


app_config = AppConfig.from_env()
kb = KnowledgeBase(
    chunk_size=app_config.default_chunk_size,
    chunk_overlap=app_config.default_chunk_overlap,
    storage_path=app_config.storage_path,
    default_embedding_model=app_config.default_embedding_model,
)

log_level = getattr(logging, app_config.log_level.upper(), logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("rag-knowledge-assistant")

app = FastAPI(
    title="RAG Knowledge Assistant",
    version=__version__,
    description="A simple, open-source RAG knowledge API.",
    docs_url="/docs" if app_config.docs_enabled else None,
    redoc_url="/redoc" if app_config.docs_enabled else None,
    openapi_url="/openapi.json" if app_config.docs_enabled else None,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_UI_DIR = _PROJECT_ROOT / "ui"
if _UI_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")

if app_config.allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(app_config.allowed_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

_START_TIME = time.time()


@dataclass
class _RouteMetric:
    requests: int = 0
    errors: int = 0
    total_duration_ms: float = 0.0
    last_status: int | None = None

    @property
    def avg_response_ms(self) -> float:
        if self.requests == 0:
            return 0.0
        return round(self.total_duration_ms / self.requests, 3)


class _MetricsCollector:
    def __init__(self) -> None:
        self._metrics: Dict[str, _RouteMetric] = {}
        self._requests = 0
        self._errors = 0
        self._lock = Lock()

    def record(self, method: str, path: str, status_code: int, duration_ms: float) -> None:
        key = f"{method} {path}"
        with self._lock:
            metric = self._metrics.setdefault(key, _RouteMetric())
            metric.requests += 1
            metric.total_duration_ms += duration_ms
            metric.last_status = status_code

            self._requests += 1
            if status_code >= 400:
                metric.errors += 1
                self._errors += 1

    def snapshot(self) -> MetricsResponse:
        with self._lock:
            return MetricsResponse(
                timestamp=datetime.now(timezone.utc).isoformat(),
                uptime_seconds=round(time.time() - _START_TIME, 2),
                requests_total=self._requests,
                errors_total=self._errors,
                routes={
                    key: RouteMetrics(
                        requests=metric.requests,
                        errors=metric.errors,
                        avg_response_ms=metric.avg_response_ms,
                        last_status=metric.last_status,
                    )
                    for key, metric in self._metrics.items()
                },
            )


class _RateLimiter:
    def __init__(self, requests: int, window_seconds: int) -> None:
        self._requests = requests
        self._window_seconds = max(1, window_seconds)
        self._timestamps: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> tuple[bool, int, int]:
        if self._requests <= 0:
            return True, 0, 0

        now = time.time()
        history = self._timestamps[key]
        cutoff = now - self._window_seconds

        while history and history[0] <= cutoff:
            history.popleft()

        if len(history) >= self._requests:
            # Return seconds until the next token is available.
            return False, 0, max(1, int(history[0] + self._window_seconds - now))

        history.append(now)
        return True, self._requests - len(history), max(1, self._window_seconds)


def _route_key_from_request(request: Request) -> str:
    return request.url.path


def _is_exempt_from_rate_limit(path: str) -> bool:
    return path.startswith("/health") or path.startswith("/ready") or path.startswith("/docs") or path.startswith("/redoc") or path.startswith("/openapi.json") or path.startswith("/metrics")


def _should_enforce_rate_limit(request: Request) -> bool:
    if app_config.api_key is None:
        return True
    supplied_key = request.headers.get("x-api-key")
    if not supplied_key:
        return False
    return supplied_key == app_config.api_key


def _is_json_request(request: Request) -> bool:
    return request.method in {"POST", "PUT", "PATCH", "DELETE"}


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "anonymous"


def _status_error_code(status_code: int) -> str:
    if status_code == 400:
        return "invalid_request"
    if status_code == 401:
        return "unauthorized"
    if status_code == 413:
        return "payload_too_large"
    if status_code == 422:
        return "validation_error"
    if status_code == 429:
        return "rate_limit_exceeded"
    if status_code == 500:
        return "internal_error"
    if status_code == 503:
        return "dependency_failure"
    if status_code == 404:
        return "not_found"
    return "request_failed"


def _error_payload(request: Request, status_code: int, message: str, details: Dict[str, str] | None = None) -> ErrorResponse:
    return ErrorResponse(
        error_code=_status_error_code(status_code),
        message=str(message),
        request_id=getattr(request.state, "request_id", None),
        path=request.url.path,
        details=details,
    )


_metrics_collector = _MetricsCollector()
_rate_limiter = _RateLimiter(
    requests=app_config.rate_limit_requests,
    window_seconds=app_config.rate_limit_window_seconds,
)


@app.middleware("http")
async def observability_and_security_middleware(request: Request, call_next):
    start = time.perf_counter()
    request_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    request.state.request_id = request_id
    path = _route_key_from_request(request)

        if _is_exempt_from_rate_limit(path) is False and _should_enforce_rate_limit(request):
            if app_config.rate_limit_requests > 0:
                allowed, remaining, reset_after = _rate_limiter.allow(_client_key(request))
                if not allowed:
                    raise HTTPException(
                        status_code=429,
                    detail="Rate limit exceeded. Retry later.",
                )
            request.state.rate_limit_remaining = remaining
            request.state.rate_limit_reset_after = reset_after

        content_length = request.headers.get("content-length")
        if content_length and _is_json_request(request):
            try:
                body_size = int(content_length)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid Content-Length header.",
                ) from exc

            if body_size > app_config.request_body_limit_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Request body exceeds configured limit ({app_config.request_body_limit_bytes} bytes).",
                )

    try:
        response = await call_next(request)
    except HTTPException as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        _metrics_collector.record(request.method, path, exc.status_code, duration_ms)
        raise
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        _metrics_collector.record(request.method, path, 500, duration_ms)
        logger.exception("Unhandled request failure for %s %s", request.method, path)
        raise

    duration_ms = (time.perf_counter() - start) * 1000
    _metrics_collector.record(request.method, path, response.status_code, duration_ms)
    response.headers["x-request-id"] = request_id
    response.headers["x-ratelimit-limit"] = str(app_config.rate_limit_requests)
    if path not in {"/docs", "/redoc", "/openapi.json", "/metrics", "/ready", "/health"}:
        response.headers["x-ratelimit-remaining"] = str(getattr(request.state, "rate_limit_remaining", 0))
        response.headers["x-ratelimit-reset-after"] = str(getattr(request.state, "rate_limit_reset_after", 0))
    return response


def _format_answer(context: list[ContextChunk]) -> str:
    return "\n\n".join(
        f"[{idx+1}] {entry.source_id}#{entry.chunk_index}: {entry.text}"
        for idx, entry in enumerate(context)
    )


def _query_response(chunks: list) -> QueryResponse:
    context = [
        ContextChunk(
            source_id=item.source_id,
            chunk_index=item.chunk_index,
            text=item.text,
            score=round(item.score, 4),
            metadata=item.metadata,
        )
        for item in chunks
    ]

    return QueryResponse(answer=_format_answer(context), context=context, count=len(context))


async def _guard_api_key(x_api_key: str | None = Header(default=None, alias="x-api-key")) -> None:
    if app_config.api_key is None:
        return
    if not x_api_key or x_api_key != app_config.api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def _wrap_kb_errors(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(
        status_code=503,
        detail=str(exc),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    payload = _error_payload(request, exc.status_code, str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    payload = _error_payload(
        request,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "Request validation failed.",
        details={"errors": exc.errors()},
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    payload = _error_payload(
        request,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "An unexpected error occurred while processing the request.",
        details={"error": str(exc)},
    )
    return JSONResponse(status_code=500, content=payload.model_dump())


@app.get("/health")
def health_check() -> dict:
    status_data = kb.health()
    status_data.update(
        {
            "status": "ok",
            "service": app_config.service_name,
            "version": __version__,
            "uptime_seconds": round(time.time() - _START_TIME, 2),
            "docs_enabled": app_config.docs_enabled,
        }
    )
    return status_data


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse | dict:
    if _UI_DIR.is_dir():
        return RedirectResponse(url="/ui/")
    return {
        "status": "ok",
        "service": app_config.service_name,
        "version": __version__,
        "ui": "disabled (missing ui directory)",
    }


@app.get("/ready")
def readiness() -> dict:
    if not kb.storage_probe():
        raise HTTPException(status_code=503, detail="Storage backend is not available.")
    return {
        "status": "ready",
        "service": app_config.service_name,
        "version": __version__,
        "storage": kb.storage_status(),
    }


@app.get("/stats")
def stats() -> dict:
    return kb.stats()


@app.post(
    "/ingest",
    response_model=IngestResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    status_code=201,
    dependencies=[Depends(_guard_api_key)],
)
def ingest(payload: IngestRequest):
    try:
        chunks = kb.ingest(
            payload.source_id,
            payload.content,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            metadata=payload.metadata,
        )
    except Exception as exc:
        raise _wrap_kb_errors(exc) from exc

    stats_data = kb.stats()
    return IngestResponse(
        source_id=payload.source_id,
        chunks_indexed=chunks,
        total_chunks=stats_data["chunks"],
    )


@app.post(
    "/ingest/bulk",
    response_model=BulkIngestResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    dependencies=[Depends(_guard_api_key)],
)
def ingest_bulk(payload: BulkIngestRequest):
    total_chunks = 0
    for document in payload.documents:
        try:
            total_chunks += kb.ingest(
                document.source_id,
                document.content,
                chunk_size=document.chunk_size,
                chunk_overlap=document.chunk_overlap,
                metadata=document.metadata,
            )
        except Exception as exc:
            raise _wrap_kb_errors(exc) from exc

    return BulkIngestResponse(
        documents=len(payload.documents),
        chunks_indexed=total_chunks,
        total_chunks=kb.stats()["chunks"],
    )


@app.post(
    "/query",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    dependencies=[Depends(_guard_api_key)],
)
def query(payload: QueryRequest):
    try:
        chunks = kb.query(
            payload.question,
            top_k=payload.top_k,
            min_score=payload.min_score,
            retrieval=payload.retrieval,
            embedding_model=payload.embedding_model,
        )
    except Exception as exc:
        raise _wrap_kb_errors(exc) from exc
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No matching chunks found. Ingest documents first using /ingest.",
        )

    return _query_response(chunks)


@app.post(
    "/query/semantic",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    dependencies=[Depends(_guard_api_key)],
)
def query_semantic(payload: SemanticQueryRequest):
    try:
        chunks = kb.query(
            payload.question,
            top_k=payload.top_k,
            min_score=payload.min_score,
            retrieval="semantic",
            embedding_model=payload.embedding_model,
            embedding_provider=payload.embedding_provider,
            local_dimensions=payload.local_dimensions,
        )
    except Exception as exc:
        raise _wrap_kb_errors(exc) from exc
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No matching chunks found. Ingest documents first using /ingest.",
        )

    return _query_response(chunks)


@app.delete(
    "/documents/{source_id}",
    response_model=DeleteResponse,
    responses={
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    dependencies=[Depends(_guard_api_key)],
)
def delete_source(source_id: str):
    try:
        removed = kb.remove_source(source_id)
    except Exception as exc:
        raise _wrap_kb_errors(exc) from exc
    return DeleteResponse(
        source_id=source_id,
        removed_chunks=removed,
        total_chunks=kb.stats()["chunks"],
    )


@app.delete(
    "/clear",
    responses={
        401: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    dependencies=[Depends(_guard_api_key)],
)
def clear_all():
    try:
        removed = kb.clear()
    except Exception as exc:
        raise _wrap_kb_errors(exc) from exc
    return JSONResponse({"removed_chunks": removed, "remaining_chunks": kb.stats()["chunks"]})


@app.get(
    "/metrics",
    response_model=MetricsResponse,
)
def metrics():
    return _metrics_collector.snapshot()
