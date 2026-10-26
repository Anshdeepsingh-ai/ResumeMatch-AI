"""
Similarity models and embedding backends for ResumeMatch AI.
Course: CSE472 Deep Learning for NLP
Milestone: Day 5 Semantic Matching Engine

Provides pluggable embedding backends:
1. SentenceTransformerEngine: Deep semantic embeddings via Sentence-BERT / Transformer models.
2. TFIDFSimilarityEngine: Fast, pure-NumPy lexical vector space model fallback.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

from config.settings import settings
from src.matching.vectorizer import TFIDFVectorizer, compute_cosine_similarity
from src.utils.logger import get_logger

logger = get_logger("resumematch.matching")


class BaseSimilarityEngine(ABC):
    """Abstract base class defining the similarity engine interface."""

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Human-readable identifier of the underlying model/algorithm."""
        ...

    @property
    @abstractmethod
    def is_fallback(self) -> bool:
        """Indicate whether this engine represents a fallback method."""
        ...

    @abstractmethod
    def compute_similarity(
        self, text_a: str, text_b: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute similarity between two text passages.

        Args:
            text_a: First text document (e.g., candidate resume).
            text_b: Second text document (e.g., job description).

        Returns:
            Tuple of:
                - similarity_score (float): Score on a 0.0 to 100.0 scale.
                - metadata (dict): Diagnostic metadata (e.g. raw cosine, dimensions).
        """
        ...


class TFIDFSimilarityEngine(BaseSimilarityEngine):
    """
    Lexical similarity engine using pure-NumPy TF-IDF vector space modeling.

    Clearly designated as a lexical fallback when deep neural embeddings are unavailable.
    Does NOT claim deep learning or semantic comprehension.
    """

    def __init__(
        self,
        sublinear_tf: bool = True,
        smooth_idf: bool = True,
    ) -> None:
        self._vectorizer = TFIDFVectorizer(
            sublinear_tf=sublinear_tf,
            smooth_idf=smooth_idf,
            lowercase=True,
            remove_stopwords=True,
        )

    @property
    def method_name(self) -> str:
        return "TF-IDF Vector Space Model (Lexical Fallback)"

    @property
    def is_fallback(self) -> bool:
        return True

    def compute_similarity(
        self, text_a: str, text_b: str
    ) -> Tuple[float, Dict[str, Any]]:
        if not isinstance(text_a, str) or not isinstance(text_b, str):
            raise TypeError(
                f"Expected string inputs, got text_a={type(text_a).__name__}, text_b={type(text_b).__name__}"
            )
        if not text_a.strip() or not text_b.strip():
            return 0.0, {
                "raw_cosine": 0.0,
                "reason": "empty_or_whitespace_input",
                "model_type": "lexical_fallback",
            }

        vec_a, vec_b, raw_cosine = self._vectorizer.fit_transform_pair(text_a, text_b)
        scaled_score = max(0.0, min(100.0, raw_cosine * 100.0))

        metadata = {
            "raw_cosine": round(raw_cosine, 4),
            "vocab_size": int(len(vec_a)),
            "model_type": "lexical_fallback",
        }
        return round(scaled_score, 2), metadata


class SentenceTransformerEngine(BaseSimilarityEngine):
    """
    Deep semantic embedding engine using Sentence-BERT / Transformer models
    from the sentence-transformers library.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name or getattr(
            settings, "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
        )
        self.device = device
        self._model: Optional[Any] = None

    @property
    def method_name(self) -> str:
        return f"sentence-transformers ({self._model_name})"

    @property
    def is_fallback(self) -> bool:
        return False

    def load_model(self) -> Any:
        """Lazily load the pretrained sentence-transformer model."""
        if self._model is not None:
            return self._model

        logger.info("Loading SentenceTransformer model: %s", self._model_name)
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device=self.device)
            return self._model
        except ImportError as err:
            raise ImportError(
                f"The 'sentence-transformers' package is required for deep semantic embeddings. "
                f"Error: {err}"
            ) from err
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load sentence-transformers model '{self._model_name}': {exc}"
            ) from exc

    def compute_similarity(
        self, text_a: str, text_b: str
    ) -> Tuple[float, Dict[str, Any]]:
        if not isinstance(text_a, str) or not isinstance(text_b, str):
            raise TypeError(
                f"Expected string inputs, got text_a={type(text_a).__name__}, text_b={type(text_b).__name__}"
            )
        if not text_a.strip() or not text_b.strip():
            return 0.0, {
                "raw_cosine": 0.0,
                "reason": "empty_or_whitespace_input",
                "model_type": "deep_semantic",
            }

        model = self.load_model()
        embeddings = model.encode(
            [text_a, text_b],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings_arr = np.asarray(embeddings, dtype=np.float64)
        if embeddings_arr.ndim != 2 or embeddings_arr.shape[0] < 2:
            raise ValueError(
                f"Expected 2-dimensional embeddings for 2 inputs, got shape {embeddings_arr.shape}"
            )

        vec_a = embeddings_arr[0]
        vec_b = embeddings_arr[1]

        raw_cosine = compute_cosine_similarity(vec_a, vec_b)
        scaled_score = max(0.0, min(100.0, raw_cosine * 100.0))

        metadata = {
            "raw_cosine": round(raw_cosine, 4),
            "embedding_dim": int(vec_a.shape[0]),
            "model_name": self._model_name,
            "model_type": "deep_semantic",
        }
        return round(scaled_score, 2), metadata


class AutoSimilarityEngine(BaseSimilarityEngine):
    """
    Adaptive similarity engine that prioritizes deep SentenceTransformer embeddings
    and automatically falls back to pure-NumPy TF-IDF lexical modeling if model
    loading or runtime inference/encoding fails.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self._target_model = model_name or getattr(
            settings, "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
        )
        self._device = device
        self.primary_engine = SentenceTransformerEngine(
            model_name=self._target_model, device=self._device
        )
        self.fallback_engine = TFIDFSimilarityEngine()
        self._active_engine: BaseSimilarityEngine = self.primary_engine
        self._fallback_occurred: bool = False
        self._last_fallback_reason: Optional[str] = None

        # Check availability of sentence_transformers package
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as err:
            self._fallback_occurred = True
            self._active_engine = self.fallback_engine
            self._last_fallback_reason = f"ImportError: {err}"

    @property
    def method_name(self) -> str:
        return self._active_engine.method_name

    @property
    def is_fallback(self) -> bool:
        return self._active_engine.is_fallback

    def compute_similarity(
        self, text_a: str, text_b: str
    ) -> Tuple[float, Dict[str, Any]]:
        if not isinstance(text_a, str) or not isinstance(text_b, str):
            raise TypeError(
                f"Expected string inputs, got text_a={type(text_a).__name__}, text_b={type(text_b).__name__}"
            )

        # If already operating in fallback mode, route directly to fallback engine
        if self._fallback_occurred:
            score, meta = self.fallback_engine.compute_similarity(text_a, text_b)
            meta["fallback_from"] = self.primary_engine.method_name
            meta["fallback_reason"] = self._last_fallback_reason
            meta["method_name"] = self.fallback_engine.method_name
            meta["is_fallback"] = True
            return score, meta

        try:
            score, meta = self.primary_engine.compute_similarity(text_a, text_b)
            self._active_engine = self.primary_engine
            return score, meta
        except TypeError:
            # Preserve programming errors and invalid types (e.g. non-string inputs)
            raise
        except (RuntimeError, OSError, ImportError, ValueError) as exc:
            logger.warning(
                "SentenceTransformer runtime execution failed (%s: %s); falling back to TF-IDF vector space model.",
                type(exc).__name__,
                exc,
            )
            self._fallback_occurred = True
            self._active_engine = self.fallback_engine
            self._last_fallback_reason = f"{type(exc).__name__}: {str(exc)}"
            score, meta = self.fallback_engine.compute_similarity(text_a, text_b)
            meta["fallback_from"] = self.primary_engine.method_name
            meta["fallback_reason"] = self._last_fallback_reason
            meta["method_name"] = self.fallback_engine.method_name
            meta["is_fallback"] = True
            return score, meta


# Cache of instantiated similarity engines
_ENGINES_CACHE: Dict[str, BaseSimilarityEngine] = {}


def clear_engine_cache() -> None:
    """Clear cached similarity engine instances (useful for testing)."""
    _ENGINES_CACHE.clear()


def get_similarity_engine(
    method: str = "auto",
    model_name: Optional[str] = None,
) -> BaseSimilarityEngine:
    """
    Factory function retrieving or creating a cached similarity engine.

    Args:
        method: Selection strategy ('auto', 'sentence-transformers', or 'tfidf').
        model_name: Pretrained model name for SentenceTransformer.

    Returns:
        Configured BaseSimilarityEngine instance.
    """
    effective_method = (
        method if method != "auto" else getattr(settings, "MATCHING_METHOD", "auto")
    ).lower()
    target_model = model_name or getattr(
        settings, "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
    )

    cache_key = f"{effective_method}:{target_model}"
    if cache_key in _ENGINES_CACHE:
        return _ENGINES_CACHE[cache_key]

    if effective_method in ("auto", "adaptive"):
        engine = AutoSimilarityEngine(model_name=target_model)
        _ENGINES_CACHE[cache_key] = engine
        return engine

    if effective_method in ("sentence-transformers", "deep-learning"):
        engine = SentenceTransformerEngine(model_name=target_model)
        _ENGINES_CACHE[cache_key] = engine
        return engine

    if effective_method in ("tfidf", "lexical", "fallback"):
        engine = TFIDFSimilarityEngine()
        _ENGINES_CACHE[cache_key] = engine
        return engine

    raise ValueError(
        f"Unsupported matching method '{method}'. Valid options: 'auto', 'sentence-transformers', 'tfidf'."
    )
