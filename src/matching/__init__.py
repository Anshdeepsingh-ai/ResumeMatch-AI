"""
Matching and similarity scoring module for ResumeMatch AI.
Course: CSE472 Deep Learning for NLP
Milestone: Day 5 Semantic Matching Engine
"""

from src.matching.engine import (
    SemanticMatchingEngine,
    generate_explanation,
    match_resume_to_job,
)
from src.matching.models import (
    AutoSimilarityEngine,
    BaseSimilarityEngine,
    SentenceTransformerEngine,
    TFIDFSimilarityEngine,
    clear_engine_cache,
    get_similarity_engine,
)
from src.matching.schemas import MatchResult
from src.matching.vectorizer import TFIDFVectorizer, compute_cosine_similarity

__all__ = [
    "SemanticMatchingEngine",
    "match_resume_to_job",
    "generate_explanation",
    "MatchResult",
    "AutoSimilarityEngine",
    "BaseSimilarityEngine",
    "SentenceTransformerEngine",
    "TFIDFSimilarityEngine",
    "get_similarity_engine",
    "clear_engine_cache",
    "compute_cosine_similarity",
    "TFIDFVectorizer",
]
