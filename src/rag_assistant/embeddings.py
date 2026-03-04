from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from typing import Protocol, runtime_checkable

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embedding provider interface used by the knowledge base."""

    @property
    def name(self) -> str:
        ...

    def fit(self, texts: Sequence[str]) -> None:
        ...

    def encode(self, texts: Sequence[str], *, normalize: bool = True) -> list[list[float]]:
        ...


class EmbeddingProviderError(RuntimeError):
    """Raised when an embedding provider cannot be initialized or used."""


class _BaseProvider:
    def fit(self, texts: Sequence[str]) -> None:
        return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(_BaseProvider._normalize_text(text).lower())


class LocalHashProvider(_BaseProvider):
    """Deterministic local provider with no heavy dependencies."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions <= 1:
            raise ValueError("dimensions must be greater than 1")
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return f"local_hash_{self._dimensions}"

    def encode(self, texts: Sequence[str], *, normalize: bool = True) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self._dimensions
            for token in self._tokenize(text):
                if not token:
                    continue
                vector[(ord(token[0]) + ord(token[-1])) % self._dimensions] += 1.0
                for char in token:
                    vector[ord(char) % self._dimensions] += 0.2

            if normalize:
                norm = sum(value * value for value in vector) ** 0.5
                if norm > 0.0:
                    vector = [value / norm for value in vector]

            vectors.append(vector)
        return vectors


class LocalTfIdfProvider(_BaseProvider):
    """Numpy-backed local TF-IDF provider."""

    def __init__(self, dimensions: int = 512) -> None:
        if np is None:
            raise EmbeddingProviderError(
                "numpy is required for local_tfidf. Install with `pip install -e '.[embeddings]'`."
            )
        if dimensions <= 1:
            raise ValueError("dimensions must be greater than 1")

        self._dimensions = dimensions
        self._vocabulary: dict[str, int] = {}
        self._idf: Any = None

    @property
    def name(self) -> str:
        return f"local_tfidf_{self._dimensions}"

    def fit(self, texts: Sequence[str]) -> None:
        if np is None:
            raise EmbeddingProviderError(
                "numpy is required for local_tfidf. Install with `pip install -e '.[embeddings]'`."
            )
        if not texts:
            self._vocabulary = {}
            self._idf = np.zeros(self._dimensions, dtype=float)
            return

        tokenized = [self._tokenize(text) for text in texts]
        doc_frequency: dict[str, int] = {}
        for row in tokenized:
            for token in set(row):
                doc_frequency[token] = doc_frequency.get(token, 0) + 1

        ranked_tokens = sorted(
            doc_frequency.items(),
            key=lambda item: ((len(tokenized) + 1) / (item[1] + 1), item[0]),
            reverse=True,
        )
        vocab_order = [token for token, _ in ranked_tokens][: self._dimensions]

        self._vocabulary = {token: idx for idx, token in enumerate(vocab_order)}

        n_docs = max(1, len(tokenized))
        idf = [0.0] * self._dimensions
        for token, idx in self._vocabulary.items():
            df = doc_frequency[token]
            idf[idx] = (n_docs + 1) / (df + 1)
            idf[idx] = float(math.log(idf[idx]) + 1.0)

        self._idf = np.array(idf, dtype=float)

    def encode(self, texts: Sequence[str], *, normalize: bool = True) -> list[list[float]]:
        if np is None:
            raise EmbeddingProviderError(
                "numpy is required for local_tfidf. Install with `pip install -e '.[embeddings]'`."
            )
        if not self._vocabulary:
            self.fit(texts)
            if not self._vocabulary:
                return [[0.0 for _ in range(self._dimensions)] for _ in texts]

        dim = self._dimensions
        vectors = np.zeros((len(texts), dim), dtype=float)

        for row, text in enumerate(texts):
            token_counts: dict[str, int] = {}
            for token in self._tokenize(text):
                token_counts[token] = token_counts.get(token, 0) + 1
            for token, count in token_counts.items():
                idx = self._vocabulary.get(token)
                if idx is None:
                    continue
                vectors[row, idx] = count * self._idf[idx]

        if normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            vectors = vectors / norms

        return [list(map(float, row)) for row in vectors]


class SentenceTransformerProvider(_BaseProvider):
    """Wrapper around sentence-transformers with lazy import."""

    def __init__(self, model_name: str) -> None:
        if not model_name or not model_name.strip():
            raise ValueError("embedding_model cannot be empty")

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - import guard
            raise EmbeddingProviderError(
                "sentence-transformers is not installed. Install with `pip install -e '.[embeddings]'`."
            ) from exc

        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingProviderError(f"Could not initialize embedding model '{model_name}'.") from exc
        self._model_name = model_name

    @property
    def name(self) -> str:
        return self._model_name

    def encode(self, texts: Sequence[str], *, normalize: bool = True) -> list[list[float]]:
        try:
            vectors = self._model.encode(texts, normalize_embeddings=normalize)
        except Exception as exc:
            raise EmbeddingProviderError("Embedding model failed to encode texts.") from exc
        return [list(map(float, vector)) for vector in vectors]


class OnnxTransformerProvider(_BaseProvider):
    """Local ONNX embedding provider for exported transformer checkpoints."""

    def __init__(self, model_path: str, tokenizer_path: str | None = None) -> None:
        if np is None:
            raise EmbeddingProviderError("numpy is required for onnx_local provider.")
        if not model_path or not model_path.strip():
            raise ValueError("embedding_model cannot be empty for onnx provider")

        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise EmbeddingProviderError(
                "onnxruntime is not installed. Install with `pip install -e '.[onnx]'` or add onnxruntime to your env."
            ) from exc

        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise EmbeddingProviderError(
                "transformers is required for onnx_local provider. Install with `pip install transformers`."
            ) from exc

        self._model_path = str(Path(model_path).expanduser())
        if not Path(self._model_path).exists():
            raise EmbeddingProviderError(f"ONNX model path does not exist: {self._model_path}")

        self._tokenizer_path = tokenizer_path or str(Path(self._model_path).parent)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Failed to load tokenizer for ONNX provider from '{self._tokenizer_path}'."
            ) from exc

        try:
            self._session = ort.InferenceSession(self._model_path)
        except Exception as exc:
            raise EmbeddingProviderError(f"Could not initialize ONNX session from '{self._model_path}'.") from exc

        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._output_names = [out.name for out in self._session.get_outputs()]

    @property
    def name(self) -> str:
        return f"onnx_local:{self._model_path}"

    def _prepare_inputs(self, texts: Sequence[str]):
        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="np",
        )

        # Some tokenizer objects expose a dict-like interface directly;
        # keep this generic to avoid strict assumptions.
        encoded_dict = dict(encoded)

        input_map = {}
        if "input_ids" in self._input_names and "input_ids" in encoded_dict:
            input_map["input_ids"] = np.asarray(encoded_dict["input_ids"])
        if "attention_mask" in self._input_names and "attention_mask" in encoded_dict:
            input_map["attention_mask"] = np.asarray(encoded_dict["attention_mask"])
        if "token_type_ids" in self._input_names and "token_type_ids" in encoded_dict:
            input_map["token_type_ids"] = np.asarray(encoded_dict["token_type_ids"])

        if not input_map:
            raise EmbeddingProviderError("ONNX model inputs did not match tokenizer output.")
        return input_map, np.asarray(encoded_dict.get("attention_mask")) if "attention_mask" in encoded_dict else None

    @staticmethod
    def _reduce_embedding(vectors: np.ndarray, attention_mask: np.ndarray | None = None) -> np.ndarray:
        if vectors.ndim == 2:
            return vectors

        if vectors.ndim == 1:
            return vectors.reshape(1, -1)

        if vectors.ndim == 3:
            if attention_mask is not None:
                mask = attention_mask.astype(float)
                seq_len = min(vectors.shape[1], mask.shape[1])
                reduced = vectors[:, :seq_len, :]
                masked = reduced * mask[:, :seq_len, None]
                mask_sum = mask[:, :seq_len].sum(axis=1, keepdims=True)
                mask_sum[mask_sum == 0.0] = 1.0
                return masked.sum(axis=1) / mask_sum

            return vectors.mean(axis=1)

        while vectors.ndim > 2:
            vectors = vectors.mean(axis=1)
        return vectors

    def encode(self, texts: Sequence[str], *, normalize: bool = True) -> list[list[float]]:
        if not texts:
            return []

        input_map, attention_mask = self._prepare_inputs(texts)
        outputs = self._session.run(self._output_names or None, input_map)
        if not outputs:
            return []

        candidate = None
        for output in outputs:
            if np.asarray(output).ndim >= 2:
                candidate = np.asarray(output)
                if candidate.ndim <= 3:
                    break

        if candidate is None:
            raise EmbeddingProviderError("ONNX model output has no supported array shape.")

        embedding = self._reduce_embedding(candidate, attention_mask=attention_mask)
        if embedding.ndim != 2:
            raise EmbeddingProviderError(
                f"ONNX embedding output shape unsupported: {embedding.shape}"
            )

        if normalize:
            norms = (embedding ** 2).sum(axis=1, keepdims=True) ** 0.5
            norms[norms == 0.0] = 1.0
            embedding = embedding / norms

        return [list(map(float, row)) for row in embedding]


@dataclass(frozen=True)
class ProviderSpec:
    provider: str
    model: str | None = None
    dimensions: int | None = None


def build_embedding_provider(spec: ProviderSpec) -> EmbeddingProvider:
    """Build a provider from request-like config."""
    provider = spec.provider.strip().lower()

    if provider in {"sentence_transformers", "sentence-transformer", "sentence"}:
        return SentenceTransformerProvider(spec.model or "sentence-transformers/all-MiniLM-L6-v2")
    if provider in {"local", "local_hash", "hash"}:
        return LocalHashProvider(dimensions=spec.dimensions or 64)
    if provider in {"local_tfidf", "tfidf_local", "local-tfidf"}:
        return LocalTfIdfProvider(dimensions=spec.dimensions or 256)
    if provider in {"onnx", "onnx_local"}:
        if not spec.model:
            raise ValueError("embedding_model is required for onnx_local provider")
        return OnnxTransformerProvider(spec.model)

    raise ValueError(f"unknown embedding provider '{spec.provider}'")
