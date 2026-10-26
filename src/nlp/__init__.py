"""
NLP Feature Extraction Module for ResumeMatch AI.
Course: CSE472 Deep Learning for NLP
Milestone: Day 4 NLP Feature Extraction
"""

from src.nlp.feature_extractor import NLPFeatureExtractor, extract_features
from src.nlp.schemas import DocumentStatistics, NLPFeatures, SkillMention
from src.nlp.skills import DEFAULT_SKILL_TAXONOMY, SkillExtractor
from src.nlp.tokenizer import (
    DEFAULT_STOPWORDS,
    extract_candidate_keyphrases,
    extract_candidate_keywords,
    segment_sentences,
    tokenize,
)

__all__ = [
    "NLPFeatureExtractor",
    "extract_features",
    "NLPFeatures",
    "SkillMention",
    "DocumentStatistics",
    "SkillExtractor",
    "DEFAULT_SKILL_TAXONOMY",
    "tokenize",
    "segment_sentences",
    "extract_candidate_keywords",
    "extract_candidate_keyphrases",
    "DEFAULT_STOPWORDS",
]
