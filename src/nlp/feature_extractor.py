"""
Core NLP Feature Extraction Service for ResumeMatch AI.
Extracts lexical, structural, keyword, and technical skill features.
Course: CSE472 Deep Learning for NLP
Milestone: Day 4 NLP Feature Extraction
"""

from typing import Any, Optional

from src.nlp.schemas import DocumentStatistics, NLPFeatures
from src.nlp.skills import DEFAULT_SKILL_TAXONOMY, SkillExtractor
from src.nlp.tokenizer import (
    extract_candidate_keyphrases,
    extract_candidate_keywords,
    segment_sentences,
    tokenize,
)
from src.preprocessing import normalize_text
from src.utils.logger import get_logger

logger = get_logger("nlp")


def _validate_input_str(text: Any) -> None:
    """Validate that input text is a string."""
    if not isinstance(text, str):
        raise TypeError(f"Input text must be a string, got {type(text).__name__}")


class NLPFeatureExtractor:
    """
    NLP Feature Extraction service for resumes and job descriptions.
    Integrates preprocessing, tokenization, sentence segmentation,
    candidate keyword/phrase mining, and technical skill matching.
    """

    def __init__(
        self,
        skill_extractor: Optional[SkillExtractor] = None,
        top_keywords: int = 20,
        top_keyphrases: int = 15,
    ):
        """
        Initialize NLPFeatureExtractor.

        Args:
            skill_extractor: Configured SkillExtractor instance. If None, uses default.
            top_keywords: Number of candidate keywords to retain.
            top_keyphrases: Number of candidate keyphrases to retain.
        """
        self.skill_extractor = (
            skill_extractor if skill_extractor is not None else SkillExtractor()
        )
        self.top_keywords = top_keywords
        self.top_keyphrases = top_keyphrases

    def extract_features(
        self,
        text: str,
        preprocess: bool = True,
    ) -> NLPFeatures:
        """
        Extract structured NLP features from document text.

        Args:
            text: Raw or pre-normalized document text.
            preprocess: Whether to run text normalization via src.preprocessing.

        Returns:
            NLPFeatures schema containing extracted tokens, sentences, keywords,
            phrases, skills, and structural document statistics.

        Raises:
            TypeError: If text is not a string.
        """
        _validate_input_str(text)

        # 1. Preprocessing stage
        clean_text = normalize_text(text) if preprocess else text.strip()

        # Handle empty or whitespace-only documents predictably
        if not clean_text or clean_text.isspace():
            logger.debug("Document is empty or whitespace; returning zeroed NLP features.")
            zero_stats = DocumentStatistics(
                character_count=0,
                word_count=0,
                unique_word_count=0,
                token_count=0,
                sentence_count=0,
                paragraph_count=0,
                lexical_diversity=0.0,
                avg_sentence_length=0.0,
                skill_count=0,
            )
            return NLPFeatures(
                clean_text="",
                tokens=[],
                sentences=[],
                keywords=[],
                keyphrases=[],
                skills=[],
                statistics=zero_stats,
            )

        # 2. Tokenization and Sentence Segmentation
        tokens = tokenize(clean_text)
        sentences = segment_sentences(clean_text)

        # 3. Candidate Keyword & Keyphrase Mining
        keywords = extract_candidate_keywords(tokens, top_k=self.top_keywords)
        keyphrases = extract_candidate_keyphrases(tokens, top_k=self.top_keyphrases)

        # 4. Configurable Technical Skill Matching
        skills = self.skill_extractor.extract_skills(clean_text)

        # 5. Document Statistics
        words = clean_text.split()
        word_count = len(words)
        unique_word_count = len(set(w.lower() for w in words))
        token_count = len(tokens)
        sentence_count = len(sentences)
        paragraphs = [p for p in clean_text.split("\n\n") if p.strip()]
        paragraph_count = len(paragraphs)

        lexical_diversity = (
            round(unique_word_count / word_count, 4) if word_count > 0 else 0.0
        )
        avg_sentence_length = (
            round(word_count / sentence_count, 2) if sentence_count > 0 else 0.0
        )

        statistics = DocumentStatistics(
            character_count=len(clean_text),
            word_count=word_count,
            unique_word_count=unique_word_count,
            token_count=token_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            lexical_diversity=lexical_diversity,
            avg_sentence_length=avg_sentence_length,
            skill_count=len(skills),
        )

        # Log safe non-PII metadata only
        logger.info(
            "Extracted NLP features: %d tokens, %d sentences, %d skills detected",
            token_count,
            sentence_count,
            len(skills),
        )

        return NLPFeatures(
            clean_text=clean_text,
            tokens=tokens,
            sentences=sentences,
            keywords=keywords,
            keyphrases=keyphrases,
            skills=skills,
            statistics=statistics,
        )


def extract_features(
    text: str,
    preprocess: bool = True,
    skill_extractor: Optional[SkillExtractor] = None,
    top_keywords: int = 20,
    top_keyphrases: int = 15,
) -> NLPFeatures:
    """
    Convenience function to extract NLP features from document text.

    Args:
        text: Input text string.
        preprocess: Whether to run text normalization first.
        skill_extractor: Optional custom SkillExtractor instance.
        top_keywords: Max candidate keywords to return.
        top_keyphrases: Max candidate keyphrases to return.

    Returns:
        NLPFeatures payload.
    """
    extractor = NLPFeatureExtractor(
        skill_extractor=skill_extractor,
        top_keywords=top_keywords,
        top_keyphrases=top_keyphrases,
    )
    return extractor.extract_features(text, preprocess=preprocess)
