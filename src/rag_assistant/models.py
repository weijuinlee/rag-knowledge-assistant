from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, conint


class IngestRequest(BaseModel):
    source_id: str = Field(min_length=1, strip_whitespace=True)
    content: str = Field(min_length=1)
    chunk_size: conint(ge=64, le=4000) = 800
    chunk_overlap: conint(ge=0, le=3000) = 120
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    source_id: str
    chunks_indexed: int
    total_chunks: int


class BulkIngestRequest(BaseModel):
    documents: List[IngestRequest] = Field(min_length=1, max_length=100)


class BulkIngestResponse(BaseModel):
    documents: int
    chunks_indexed: int
    total_chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, strip_whitespace=True)
    top_k: conint(ge=1, le=20) = 5
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval: Literal["tfidf", "semantic"] = "tfidf"
    embedding_model: Optional[str] = Field(default=None, min_length=1)


class SemanticQueryRequest(BaseModel):
    question: str = Field(min_length=1, strip_whitespace=True)
    top_k: conint(ge=1, le=20) = 5
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    embedding_model: Optional[str] = Field(default=None, min_length=1)
    embedding_provider: Literal[
        "sentence_transformers",
        "local",
        "local_hash",
        "local_tfidf",
        "onnx_local",
    ] = "sentence_transformers"
    local_dimensions: conint(ge=8, le=4096) = 256


class ContextChunk(BaseModel):
    source_id: str
    chunk_index: int
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    context: List[ContextChunk]
    count: int


class DeleteResponse(BaseModel):
    source_id: str
    removed_chunks: int
    total_chunks: int


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: str | None = None
    path: str | None = None
    details: Dict[str, Any] | None = None


class RouteMetrics(BaseModel):
    requests: int
    errors: int
    avg_response_ms: float
    last_status: int | None = None


class MetricsResponse(BaseModel):
    timestamp: str
    uptime_seconds: float
    requests_total: int
    errors_total: int
    routes: Dict[str, RouteMetrics]
