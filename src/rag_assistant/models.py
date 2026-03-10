from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, conint, field_validator


ChunkingStrategy = Literal["tokens", "sentence", "paragraph", "smart"]
RetrievalMode = Literal["tfidf", "semantic", "hybrid"]
RerankerMode = Literal["none", "term_overlap"]


class IngestRequest(BaseModel):
    source_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    chunk_size: conint(ge=1, le=4000) = 800
    chunk_overlap: conint(ge=0, le=4000) = 120
    chunking_strategy: ChunkingStrategy = "tokens"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id")
    @classmethod
    def _strip_source_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("source_id cannot be empty")
        return value


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
    question: str = Field(min_length=1)
    top_k: conint(ge=1, le=20) = 5
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval: RetrievalMode = "tfidf"
    embedding_model: Optional[str] = Field(default=None, min_length=1)
    embedding_provider: Literal[
        "sentence_transformers",
        "local",
        "local_hash",
        "local_tfidf",
        "onnx_local",
    ] = "sentence_transformers"
    local_dimensions: conint(ge=8, le=4096) = 256
    reranker: RerankerMode = "none"
    candidate_pool_size: conint(ge=1, le=100) = 10

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be empty")
        return value


class SemanticQueryRequest(BaseModel):
    question: str = Field(min_length=1)
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
    reranker: RerankerMode = "none"
    candidate_pool_size: conint(ge=1, le=100) = 10

    @field_validator("question")
    @classmethod
    def _strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be empty")
        return value


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
    trace: Dict[str, Any] = Field(default_factory=dict)


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


class EvalCase(BaseModel):
    question: str = Field(min_length=1)
    expected_source_ids: List[str] = Field(min_length=1, max_length=20)
    top_k: conint(ge=1, le=20) = 5
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieval: RetrievalMode = "hybrid"
    embedding_model: Optional[str] = Field(default=None, min_length=1)
    embedding_provider: Literal[
        "sentence_transformers",
        "local",
        "local_hash",
        "local_tfidf",
        "onnx_local",
    ] = "local"
    local_dimensions: conint(ge=8, le=4096) = 256
    reranker: RerankerMode = "term_overlap"
    candidate_pool_size: conint(ge=1, le=100) = 10

    @field_validator("question")
    @classmethod
    def _strip_eval_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be empty")
        return value


class EvalRequest(BaseModel):
    cases: List[EvalCase] = Field(min_length=1, max_length=200)


class EvalCaseResult(BaseModel):
    question: str
    expected_source_ids: List[str]
    returned_source_ids: List[str]
    matched: bool
    reciprocal_rank: float
    top_hit_source_id: str | None = None
    trace: Dict[str, Any] = Field(default_factory=dict)


class EvalResponse(BaseModel):
    cases: int
    hits: int
    hit_rate: float
    mrr: float
    results: List[EvalCaseResult]
