from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from rag_assistant.embeddings import EmbeddingProviderError, ProviderSpec, build_embedding_provider

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_PATTERN = re.compile(r"\n\s*\n+")
SUPPORTED_CHUNKING_STRATEGIES = {"tokens", "sentence", "paragraph", "smart"}
SUPPORTED_RETRIEVAL_MODES = {"tfidf", "semantic", "hybrid"}
SUPPORTED_RERANKERS = {"none", "term_overlap"}


class KnowledgeBaseError(RuntimeError):
    """Base class for knowledge-base runtime failures."""


class KnowledgeBaseStorageError(KnowledgeBaseError):
    """Raised when the persistence layer cannot be loaded or written."""


@dataclass(frozen=True)
class RetrievedChunk:
    source_id: str
    chunk_index: int
    text: str
    score: float
    metadata: Dict[str, Any]


class KnowledgeBase:
    """In-memory retrieval engine with TF-IDF and pluggable semantic providers."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        storage_path: str | None = None,
        default_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self._validate_chunk_config(chunk_size, chunk_overlap)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        self._storage_path = Path(storage_path).expanduser() if storage_path else None
        self._lock = threading.Lock()
        self._chunks: List[Dict[str, Any]] = []
        self._doc_freq: Dict[str, int] = defaultdict(int)

        self._default_embedding_model = default_embedding_model.strip()
        self._embedding_provider = None
        self._chunk_embeddings: List[List[float]] = []
        self._provider_cache_key: tuple[str, str | None, int | None] | None = None

        if self._storage_path is not None:
            self._load_from_storage()

    @staticmethod
    def _validate_chunk_config(chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    @staticmethod
    def _validate_chunking_strategy(chunking_strategy: str) -> str:
        normalized = chunking_strategy.strip().lower()
        if normalized not in SUPPORTED_CHUNKING_STRATEGIES:
            raise ValueError(
                "chunking_strategy must be one of: tokens, sentence, paragraph, smart"
            )
        return normalized

    @staticmethod
    def _validate_query_config(retrieval: str, reranker: str) -> tuple[str, str]:
        normalized_retrieval = retrieval.strip().lower()
        normalized_reranker = reranker.strip().lower()
        if normalized_retrieval not in SUPPORTED_RETRIEVAL_MODES:
            raise ValueError("retrieval must be one of: tfidf, semantic, hybrid")
        if normalized_reranker not in SUPPORTED_RERANKERS:
            raise ValueError("reranker must be one of: none, term_overlap")
        return normalized_retrieval, normalized_reranker

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return TOKEN_PATTERN.findall(KnowledgeBase._normalize(text).lower())

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        normalized = KnowledgeBase._normalize(text)
        if not normalized:
            return []
        sentences = [segment.strip() for segment in SENTENCE_PATTERN.split(normalized) if segment.strip()]
        return sentences or [normalized]

    @staticmethod
    def _split_paragraphs(text: str) -> List[str]:
        paragraphs = [KnowledgeBase._normalize(block) for block in PARAGRAPH_PATTERN.split(text) if block.strip()]
        return [paragraph for paragraph in paragraphs if paragraph]

    @staticmethod
    def _normalize_metadata(metadata: Dict[str, Any] | None) -> Dict[str, Any]:
        if not metadata:
            return {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON-like object")
        try:
            json.dumps(metadata)
        except TypeError as exc:
            raise ValueError("metadata must be JSON-serializable") from exc
        return dict(metadata)

    @property
    def storage_path(self) -> str | None:
        return str(self._storage_path) if self._storage_path else None

    @property
    def persistent(self) -> bool:
        return self._storage_path is not None

    def _split_chunks_tokens(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        tokens = re.findall(TOKEN_PATTERN.pattern, self._normalize(text))
        if not tokens:
            return []

        stride = chunk_size - chunk_overlap
        chunks: List[str] = []

        for start in range(0, len(tokens), stride):
            chunk_tokens = tokens[start : start + chunk_size]
            if not chunk_tokens:
                continue
            chunks.append(" ".join(chunk_tokens))
            if start + chunk_size >= len(tokens):
                break

        return chunks

    def _chunk_units(self, units: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        chunks: List[str] = []
        current_units: List[str] = []
        current_token_count = 0

        for unit in units:
            clean_unit = self._normalize(unit)
            unit_tokens = self._tokenize(clean_unit)
            if not unit_tokens:
                continue

            if len(unit_tokens) >= chunk_size:
                if current_units:
                    chunks.append(self._normalize(" ".join(current_units)))
                    current_units = []
                    current_token_count = 0
                chunks.extend(self._split_chunks_tokens(clean_unit, chunk_size, chunk_overlap))
                continue

            if current_units and current_token_count + len(unit_tokens) > chunk_size:
                chunks.append(self._normalize(" ".join(current_units)))
                overlap_units: List[str] = []
                overlap_tokens = 0
                for previous_unit in reversed(current_units):
                    previous_tokens = self._tokenize(previous_unit)
                    overlap_units.insert(0, previous_unit)
                    overlap_tokens += len(previous_tokens)
                    if overlap_tokens >= chunk_overlap:
                        break
                current_units = overlap_units
                current_token_count = sum(len(self._tokenize(item)) for item in current_units)

            current_units.append(clean_unit)
            current_token_count += len(unit_tokens)

        if current_units:
            chunks.append(self._normalize(" ".join(current_units)))

        return chunks

    def _split_chunks(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        chunking_strategy: str,
    ) -> List[str]:
        strategy = self._validate_chunking_strategy(chunking_strategy)
        if strategy == "tokens":
            return self._split_chunks_tokens(text, chunk_size, chunk_overlap)

        if strategy == "sentence":
            return self._chunk_units(self._split_sentences(text), chunk_size, chunk_overlap)

        paragraphs = self._split_paragraphs(text)
        if strategy == "paragraph":
            return self._chunk_units(paragraphs, chunk_size, chunk_overlap)

        smart_units: List[str] = []
        for paragraph in paragraphs or [text]:
            paragraph_tokens = self._tokenize(paragraph)
            if len(paragraph_tokens) <= max(1, chunk_size // 2):
                smart_units.append(paragraph)
                continue
            smart_units.extend(self._split_sentences(paragraph))
        return self._chunk_units(smart_units, chunk_size, chunk_overlap)

    @staticmethod
    def _vectorize(tokens: List[str], doc_freq: Dict[str, int], n_chunks: int) -> Dict[str, float]:
        if not tokens or n_chunks <= 0:
            return {}

        frequencies = Counter(tokens)
        vector: Dict[str, float] = {}
        for token, tf in frequencies.items():
            idf = math.log((1 + n_chunks) / (1 + doc_freq[token]))
            vector[token] = tf * idf
        return vector

    @staticmethod
    def _cosine_similarity(v1: Dict[str, float], v2: Dict[str, float]) -> float:
        if not v1 or not v2:
            return 0.0

        if len(v1) > len(v2):
            smaller, larger = v2, v1
        else:
            smaller, larger = v1, v2

        dot = 0.0
        for token, weight in smaller.items():
            dot += weight * larger.get(token, 0.0)

        norm1 = math.sqrt(sum(weight * weight for weight in v1.values()))
        norm2 = math.sqrt(sum(weight * weight for weight in v2.values()))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    @staticmethod
    def _cosine_similarity_dense(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            return 0.0

        dot = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for left, right in zip(v1, v2):
            dot += left * right
            norm1 += left * left
            norm2 += right * right

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / math.sqrt(norm1 * norm2)

    def _rebuild_doc_freq(self) -> None:
        self._doc_freq = defaultdict(int)
        for chunk in self._chunks:
            for token in set(self._tokenize(chunk["text"])):
                self._doc_freq[token] += 1

    def _invalidate_semantic_index(self) -> None:
        self._chunk_embeddings = []

    def _load_from_storage(self) -> None:
        if self._storage_path is None:
            return

        if not self._storage_path.exists():
            return

        try:
            raw_payload = json.loads(self._storage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise KnowledgeBaseStorageError(
                f"failed to read persistence file: {self._storage_path}"
            ) from exc

        chunks = (
            raw_payload["chunks"]
            if isinstance(raw_payload, dict) and isinstance(raw_payload.get("chunks"), list)
            else raw_payload if isinstance(raw_payload, list) else None
        )
        if chunks is None:
            raise KnowledgeBaseStorageError("persistence file is not in a supported format")

        loaded: List[Dict[str, Any]] = []
        for entry in chunks:
            if not isinstance(entry, dict):
                continue
            source_id = str(entry.get("source_id", "")).strip()
            if not source_id:
                continue
            try:
                chunk_index = int(entry.get("chunk_index", 0))
            except (TypeError, ValueError) as exc:
                raise KnowledgeBaseStorageError(
                    "corrupt persistence payload: chunk_index must be an int"
                ) from exc
            text = str(entry.get("text", ""))
            metadata = entry.get("metadata", {})
            if not isinstance(metadata, dict):
                raise KnowledgeBaseStorageError("corrupt persistence payload: metadata must be dict")

            loaded.append(
                {
                    "source_id": source_id,
                    "chunk_index": chunk_index,
                    "text": text,
                    "metadata": metadata,
                }
            )

        self._chunks = loaded
        self._rebuild_doc_freq()

    def _persist(self) -> None:
        if self._storage_path is None:
            return

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "chunks": self._chunks,
        }
        temporary = self._storage_path.with_name(f"{self._storage_path.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self._storage_path)
        except OSError as exc:
            raise KnowledgeBaseStorageError(
                f"failed to persist index to: {self._storage_path}"
            ) from exc

    def storage_status(self) -> Dict[str, Any]:
        return {
            "enabled": self._storage_path is not None,
            "path": str(self._storage_path) if self._storage_path else None,
        }

    def storage_probe(self) -> bool:
        if self._storage_path is None:
            return True

        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            probe = self._storage_path.with_suffix(".probe")
            probe.write_text("{}", encoding="utf-8")
            probe.unlink()
            return True
        except OSError:
            return False

    def _build_provider_cache_key(
        self,
        embedding_provider: str,
        embedding_model: str | None,
        local_dimensions: int | None,
    ) -> tuple[str, str | None, int | None]:
        normalized = embedding_provider.strip().lower()
        if normalized == "local_tfidf":
            return (normalized, None, local_dimensions or 256)
        if normalized in {"local", "local_hash", "hash"}:
            return ("local_hash", None, local_dimensions or 64)
        if normalized in {"onnx", "onnx_local"}:
            return ("onnx_local", (embedding_model or "").strip(), local_dimensions or 256)
        return (
            "sentence_transformers",
            (embedding_model or self._default_embedding_model).strip(),
            None,
        )

    def _load_embedding_provider(
        self,
        *,
        embedding_provider: str = "sentence_transformers",
        embedding_model: str | None = None,
        local_dimensions: int = 64,
    ):
        normalized_provider = embedding_provider.strip().lower()
        cache_key = self._build_provider_cache_key(
            embedding_provider=normalized_provider,
            embedding_model=embedding_model,
            local_dimensions=local_dimensions,
        )

        if self._provider_cache_key == cache_key and self._embedding_provider is not None:
            return self._embedding_provider

        if cache_key != self._provider_cache_key:
            self._invalidate_semantic_index()

        try:
            provider = build_embedding_provider(
                ProviderSpec(
                    provider=normalized_provider,
                    model=embedding_model,
                    dimensions=local_dimensions,
                )
            )
        except EmbeddingProviderError as exc:
            raise RuntimeError(str(exc)) from exc
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

        self._embedding_provider = provider
        self._provider_cache_key = cache_key
        return provider

    def ingest(
        self,
        source_id: str,
        content: str,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        chunking_strategy: str = "tokens",
        metadata: Dict[str, Any] | None = None,
    ) -> int:
        if not source_id or not source_id.strip():
            raise ValueError("source_id cannot be empty")
        if not content or not content.strip():
            raise ValueError("content cannot be empty")

        normalized_source_id = source_id.strip()
        safe_metadata = self._normalize_metadata(metadata)
        chunk_size = chunk_size or self._chunk_size
        chunk_overlap = chunk_overlap or self._chunk_overlap
        self._validate_chunk_config(chunk_size, chunk_overlap)
        normalized_chunking_strategy = self._validate_chunking_strategy(chunking_strategy)

        chunks = self._split_chunks(
            content,
            chunk_size,
            chunk_overlap,
            normalized_chunking_strategy,
        )

        with self._lock:
            indexed = 0
            for idx, chunk_text in enumerate(chunks):
                chunk_metadata = dict(safe_metadata)
                chunk_metadata.setdefault("chunking_strategy", normalized_chunking_strategy)
                chunk_metadata.setdefault("token_count", len(self._tokenize(chunk_text)))
                self._chunks.append(
                    {
                        "source_id": normalized_source_id,
                        "chunk_index": idx,
                        "text": chunk_text,
                        "metadata": chunk_metadata,
                    }
                )
                indexed += 1

            self._rebuild_doc_freq()
            self._invalidate_semantic_index()
            self._persist()

        return indexed

    def _query_tfidf(self, question: str, top_k: int, min_score: float) -> List[RetrievedChunk]:
        question_tokens = self._tokenize(question)
        if not question_tokens:
            return []

        n_chunks = len(self._chunks)
        question_vector = self._vectorize(question_tokens, self._doc_freq, n_chunks)

        scored: List[RetrievedChunk] = []
        for chunk in self._chunks:
            chunk_vector = self._vectorize(self._tokenize(chunk["text"]), self._doc_freq, n_chunks)
            score = self._cosine_similarity(question_vector, chunk_vector)
            if score >= min_score:
                scored.append(
                    RetrievedChunk(
                        source_id=chunk["source_id"],
                        chunk_index=chunk["chunk_index"],
                        text=chunk["text"],
                        score=score,
                        metadata=chunk["metadata"],
                    )
                )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def _build_semantic_index(
        self,
        *,
        embedding_provider: str,
        embedding_model: str | None = None,
        local_dimensions: int = 64,
    ) -> List[List[float]]:
        provider = self._load_embedding_provider(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            local_dimensions=local_dimensions,
        )

        if not self._chunks:
            self._chunk_embeddings = []
            return self._chunk_embeddings

        if self._chunk_embeddings:
            return self._chunk_embeddings

        provider.fit([chunk["text"] for chunk in self._chunks])

        try:
            self._chunk_embeddings = [
                list(map(float, vector))
                for vector in provider.encode([chunk["text"] for chunk in self._chunks], normalize=True)
            ]
        except Exception as exc:
            raise RuntimeError("Embedding provider failed to encode indexed chunks.") from exc

        return self._chunk_embeddings

    def _encode_question(
        self,
        question: str,
        *,
        embedding_provider: str,
        embedding_model: str | None = None,
        local_dimensions: int = 64,
    ) -> List[float]:
        provider = self._load_embedding_provider(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            local_dimensions=local_dimensions,
        )
        try:
            vector = provider.encode([question], normalize=True)[0]
        except Exception as exc:
            raise RuntimeError("Embedding provider failed to encode query.") from exc
        return list(map(float, vector))

    def _query_semantic(
        self,
        question: str,
        top_k: int,
        min_score: float,
        embedding_model: str | None = None,
        embedding_provider: str = "sentence_transformers",
        local_dimensions: int = 64,
    ) -> List[RetrievedChunk]:
        _chunk_embeddings = self._build_semantic_index(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            local_dimensions=local_dimensions,
        )
        question_vector = self._encode_question(
            question,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            local_dimensions=local_dimensions,
        )

        scored: List[RetrievedChunk] = []
        for chunk, chunk_vector in zip(self._chunks, _chunk_embeddings):
            score = self._cosine_similarity_dense(question_vector, chunk_vector)
            if score >= min_score:
                scored.append(
                    RetrievedChunk(
                        source_id=chunk["source_id"],
                        chunk_index=chunk["chunk_index"],
                        text=chunk["text"],
                        score=score,
                        metadata=chunk["metadata"],
                    )
                )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _chunk_key(chunk: RetrievedChunk) -> tuple[str, int]:
        return (chunk.source_id, chunk.chunk_index)

    def _merge_rankings_rrf(
        self,
        rankings: List[List[RetrievedChunk]],
        *,
        top_k: int,
        rank_constant: int = 60,
    ) -> List[RetrievedChunk]:
        fused_scores: Dict[tuple[str, int], float] = defaultdict(float)
        canonical_chunk: Dict[tuple[str, int], RetrievedChunk] = {}

        for ranking in rankings:
            for rank, chunk in enumerate(ranking, start=1):
                key = self._chunk_key(chunk)
                fused_scores[key] += 1.0 / (rank_constant + rank)
                canonical_chunk[key] = chunk

        fused = [
            RetrievedChunk(
                source_id=chunk.source_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=fused_scores[key],
                metadata=chunk.metadata,
            )
            for key, chunk in canonical_chunk.items()
        ]
        fused.sort(key=lambda item: item.score, reverse=True)
        return fused[:top_k]

    def _rerank_chunks(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        *,
        reranker: str,
        top_k: int,
    ) -> List[RetrievedChunk]:
        if reranker == "none":
            return chunks[:top_k]

        question_tokens = self._tokenize(question)
        if not question_tokens:
            return chunks[:top_k]

        question_terms = set(question_tokens)
        question_bigrams = set(zip(question_tokens, question_tokens[1:]))
        normalized_question = self._normalize(question).lower()

        reranked: List[RetrievedChunk] = []
        for chunk in chunks:
            chunk_tokens = self._tokenize(chunk.text)
            if not chunk_tokens:
                reranked.append(chunk)
                continue

            chunk_terms = set(chunk_tokens)
            chunk_bigrams = set(zip(chunk_tokens, chunk_tokens[1:]))
            overlap_score = len(question_terms & chunk_terms) / max(1, len(question_terms))
            bigram_score = len(question_bigrams & chunk_bigrams) / max(1, len(question_bigrams))
            exact_phrase_bonus = (
                1.0 if normalized_question and normalized_question in self._normalize(chunk.text).lower() else 0.0
            )
            reranked.append(
                RetrievedChunk(
                    source_id=chunk.source_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=(0.55 * chunk.score) + (0.3 * overlap_score) + (0.1 * bigram_score) + (0.05 * exact_phrase_bonus),
                    metadata=chunk.metadata,
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]

    def query_with_trace(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.0,
        retrieval: str = "tfidf",
        embedding_model: str | None = None,
        embedding_provider: str = "sentence_transformers",
        local_dimensions: int = 64,
        reranker: str = "none",
        candidate_pool_size: int | None = None,
    ) -> tuple[List[RetrievedChunk], Dict[str, Any]]:
        if not question or not question.strip():
            return [], {"retrieval": retrieval, "reranker": reranker, "stages": []}

        normalized_retrieval, normalized_reranker = self._validate_query_config(retrieval, reranker)
        candidate_limit = max(top_k, candidate_pool_size or top_k)

        with self._lock:
            if not self._chunks:
                return [], {
                    "retrieval": normalized_retrieval,
                    "reranker": normalized_reranker,
                    "stages": [],
                    "candidates_considered": 0,
                }

            stages: List[Dict[str, Any]] = []
            if normalized_retrieval == "tfidf":
                candidates = self._query_tfidf(question=question, top_k=candidate_limit, min_score=min_score)
                stages.append({"stage": "tfidf", "candidates": len(candidates)})
            elif normalized_retrieval == "semantic":
                candidates = self._query_semantic(
                    question,
                    top_k=candidate_limit,
                    min_score=min_score,
                    embedding_model=embedding_model,
                    embedding_provider=embedding_provider,
                    local_dimensions=local_dimensions,
                )
                stages.append(
                    {
                        "stage": "semantic",
                        "candidates": len(candidates),
                        "embedding_provider": embedding_provider,
                        "embedding_model": embedding_model or self._default_embedding_model,
                    }
                )
            else:
                tfidf_candidates = self._query_tfidf(
                    question=question,
                    top_k=candidate_limit,
                    min_score=min_score,
                )
                semantic_candidates = self._query_semantic(
                    question,
                    top_k=candidate_limit,
                    min_score=min_score,
                    embedding_model=embedding_model,
                    embedding_provider=embedding_provider,
                    local_dimensions=local_dimensions,
                )
                stages.extend(
                    [
                        {"stage": "tfidf", "candidates": len(tfidf_candidates)},
                        {
                            "stage": "semantic",
                            "candidates": len(semantic_candidates),
                            "embedding_provider": embedding_provider,
                            "embedding_model": embedding_model or self._default_embedding_model,
                        },
                    ]
                )
                candidates = self._merge_rankings_rrf(
                    [tfidf_candidates, semantic_candidates],
                    top_k=candidate_limit,
                )
                stages.append({"stage": "rrf_fusion", "candidates": len(candidates)})

            reranked = self._rerank_chunks(
                question,
                candidates,
                reranker=normalized_reranker,
                top_k=top_k,
            )
            if normalized_reranker != "none":
                stages.append({"stage": normalized_reranker, "candidates": len(reranked)})

            trace = {
                "retrieval": normalized_retrieval,
                "reranker": normalized_reranker,
                "candidate_pool_size": candidate_limit,
                "candidates_considered": len(candidates),
                "returned_chunks": len(reranked),
                "stages": stages,
            }
            return reranked, trace

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.0,
        retrieval: str = "tfidf",
        embedding_model: str | None = None,
        embedding_provider: str = "sentence_transformers",
        local_dimensions: int = 64,
        reranker: str = "none",
        candidate_pool_size: int | None = None,
    ) -> List[RetrievedChunk]:
        chunks, _ = self.query_with_trace(
            question,
            top_k=top_k,
            min_score=min_score,
            retrieval=retrieval,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            local_dimensions=local_dimensions,
            reranker=reranker,
            candidate_pool_size=candidate_pool_size,
        )
        return chunks

    def remove_source(self, source_id: str) -> int:
        with self._lock:
            before = len(self._chunks)
            normalized = source_id.strip()
            self._chunks = [chunk for chunk in self._chunks if chunk["source_id"] != normalized]
            removed = before - len(self._chunks)
            if removed:
                self._rebuild_doc_freq()
                self._invalidate_semantic_index()
                self._persist()
            return removed

    def clear(self) -> int:
        with self._lock:
            removed = len(self._chunks)
            self._chunks.clear()
            self._doc_freq.clear()
            self._invalidate_semantic_index()
            self._persist()
            return removed

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            unique_sources = len({chunk["source_id"] for chunk in self._chunks})
            provider_name = None
            model_name = None
            provider_dims = None

            if self._provider_cache_key is not None:
                provider_name, model_name, provider_dims = self._provider_cache_key

            return {
                "documents": unique_sources,
                "chunks": len(self._chunks),
                "embedding_provider": provider_name,
                "embedding_model": model_name,
                "embedding_dimensions": provider_dims,
                "embedding_chunks_indexed": len(self._chunk_embeddings),
                "default_chunk_size": self._chunk_size,
                "default_chunk_overlap": self._chunk_overlap,
                "storage": self.storage_status(),
            }

    def health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "storage": self.storage_status(),
                "documents": len({chunk["source_id"] for chunk in self._chunks}),
                "chunks": len(self._chunks),
            }
