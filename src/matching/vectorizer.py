"""
Vector similarity and TF-IDF lexical modeling for ResumeMatch AI.
Course: CSE472 Deep Learning for NLP
Milestone: Day 5 Semantic Matching Engine

Provides pure-NumPy TF-IDF vectorization and cosine similarity calculations
without external deep learning or heavy library dependencies.
"""

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.nlp.tokenizer import DEFAULT_STOPWORDS, tokenize


def compute_cosine_similarity(
    vec_a: Union[np.ndarray, List[float]],
    vec_b: Union[np.ndarray, List[float]],
) -> float:
    """
    Compute the cosine similarity between two 1-dimensional numeric vectors:

        Cosine(u, v) = (u . v) / (||u||_2 * ||v||_2)

    Args:
        vec_a: First vector (numpy array or float sequence).
        vec_b: Second vector (numpy array or float sequence).

    Returns:
        Cosine similarity bounded within [-1.0, 1.0], or 0.0 if either vector has zero magnitude.

    Raises:
        TypeError: If inputs cannot be converted to 1D float numpy arrays.
        ValueError: If vector dimensions are mismatched or non-1D.
    """
    try:
        a = np.asarray(vec_a, dtype=np.float64)
        b = np.asarray(vec_b, dtype=np.float64)
    except Exception as exc:
        raise TypeError(f"Vectors must be numeric sequences: {exc}") from exc

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(
            f"Expected 1-dimensional vectors, got shapes {a.shape} and {b.shape}"
        )

    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Vector dimension mismatch: vector A has dimension {a.shape[0]}, "
            f"but vector B has dimension {b.shape[0]}"
        )

    if a.size == 0:
        return 0.0

    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot_product = float(np.dot(a, b))
    raw_cosine = dot_product / (norm_a * norm_b)

    # Bound against floating-point inaccuracies
    return max(-1.0, min(1.0, raw_cosine))


class TFIDFVectorizer:
    """
    Lightweight, deterministic TF-IDF vectorizer built on NumPy and the
    ResumeMatch AI technical tokenizer.

    Preserves domain terms like C++, C#, .NET, Node.js, and version numbers.
    Used as an explicit fallback when neural sentence embeddings are unavailable.
    """

    def __init__(
        self,
        sublinear_tf: bool = True,
        smooth_idf: bool = True,
        lowercase: bool = True,
        remove_stopwords: bool = True,
    ) -> None:
        """
        Initialize the TF-IDF vectorizer.

        Args:
            sublinear_tf: Apply 1 + log(tf) scaling to term frequencies.
            smooth_idf: Add 1 to document counts to prevent division by zero.
            lowercase: Lowercase tokens while preserving technical symbols.
            remove_stopwords: Filter common linguistic stopwords.
        """
        self.sublinear_tf = sublinear_tf
        self.smooth_idf = smooth_idf
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.vocabulary_: Dict[str, int] = {}
        self.idf_diag_: Optional[np.ndarray] = None

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text preserving technical identifiers and filtering stopwords."""
        if not isinstance(text, str):
            raise TypeError(f"Expected string input, got {type(text).__name__}")
        raw_tokens = tokenize(text)
        tokens: List[str] = []
        for tok in raw_tokens:
            tok_norm = tok.lower() if self.lowercase else tok
            if self.remove_stopwords and tok_norm.lower() in DEFAULT_STOPWORDS:
                continue
            if len(tok_norm) < 2:
                continue
            tokens.append(tok_norm)
        return tokens

    def fit_transform_pair(
        self, doc_a: str, doc_b: str
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute pairwise TF-IDF vectors and cosine similarity for two documents.

        Args:
            doc_a: Text of the first document (e.g. resume).
            doc_b: Text of the second document (e.g. job description).

        Returns:
            Tuple of (vector_a, vector_b, cosine_similarity_float).
        """
        tokens_a = self._tokenize(doc_a)
        tokens_b = self._tokenize(doc_b)

        if not tokens_a or not tokens_b:
            empty_vec = np.zeros(0, dtype=np.float64)
            return empty_vec, empty_vec, 0.0

        counts_a = Counter(tokens_a)
        counts_b = Counter(tokens_b)

        # Build combined vocabulary
        all_terms = sorted(list(set(counts_a.keys()) | set(counts_b.keys())))
        vocab = {term: idx for idx, term in enumerate(all_terms)}
        vocab_size = len(all_terms)

        # Calculate smooth IDF over the 2-document corpus
        n_docs = 2.0
        idf_vec = np.zeros(vocab_size, dtype=np.float64)
        for term, idx in vocab.items():
            df = (1 if term in counts_a else 0) + (1 if term in counts_b else 0)
            if self.smooth_idf:
                idf_val = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
            else:
                idf_val = math.log(n_docs / df) + 1.0
            idf_vec[idx] = idf_val

        # Vector calculation helper
        def _build_tfidf(counts: Counter) -> np.ndarray:
            vec = np.zeros(vocab_size, dtype=np.float64)
            for term, count in counts.items():
                if term in vocab:
                    idx = vocab[term]
                    if self.sublinear_tf:
                        tf_val = 1.0 + math.log(count) if count > 0 else 0.0
                    else:
                        tf_val = float(count)
                    vec[idx] = tf_val * idf_vec[idx]
            norm = np.linalg.norm(vec)
            return (vec / norm) if norm > 0.0 else vec

        vec_a = _build_tfidf(counts_a)
        vec_b = _build_tfidf(counts_b)

        sim = compute_cosine_similarity(vec_a, vec_b)
        return vec_a, vec_b, sim
