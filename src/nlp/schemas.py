"""
Data models and output schemas for the ResumeMatch AI NLP feature extraction module.
Course: CSE472 Deep Learning for NLP
Milestone: Day 4 NLP Feature Extraction
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class SkillMention:
    """
    Representation of an identified technical skill mention in a document.

    Attributes:
        name: Surface form or name matched in the source text.
        canonical_name: Standardized canonical skill name (e.g., 'Node.js', 'PostgreSQL').
        category: Taxonomy domain (e.g., 'programming_languages', 'cloud_devops').
        match_type: Type of match ('exact' or 'alias').
        count: Number of times this skill appeared in the text.
    """

    name: str
    canonical_name: str
    category: str
    match_type: str
    count: int


@dataclass(frozen=True)
class DocumentStatistics:
    """
    Structural and lexical statistics of an analyzed document.

    Attributes:
        character_count: Total number of characters in the normalized text.
        word_count: Total word count.
        unique_word_count: Count of unique words (case-insensitive).
        token_count: Total token count extracted by the technical tokenizer.
        sentence_count: Number of segmented sentences or logical units.
        paragraph_count: Number of paragraph blocks.
        lexical_diversity: Ratio of unique words to total words (0.0 to 1.0).
        avg_sentence_length: Average word count per sentence.
        skill_count: Total number of unique technical skills detected.
    """

    character_count: int
    word_count: int
    unique_word_count: int
    token_count: int
    sentence_count: int
    paragraph_count: int
    lexical_diversity: float
    avg_sentence_length: float
    skill_count: int


@dataclass
class NLPFeatures:
    """
    Complete NLP feature extraction payload for resumes and job descriptions.

    Attributes:
        clean_text: Clean, preprocessed document text.
        tokens: Sequence of tokens preserving technical terms and punctuation.
        sentences: Segmented list of sentences, bullet items, and section headings.
        keywords: Top candidate keywords with their occurrence counts.
        keyphrases: Top candidate multi-word keyphrases (2-grams, 3-grams) with counts.
        skills: List of detected SkillMention objects.
        statistics: DocumentStatistics summary.
        metadata: Optional technical or provenance metadata.
    """

    clean_text: str
    tokens: List[str]
    sentences: List[str]
    keywords: List[Tuple[str, int]]
    keyphrases: List[Tuple[str, int]]
    skills: List[SkillMention]
    statistics: DocumentStatistics
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_skill_names(self) -> List[str]:
        """Return a sorted list of unique canonical skill names detected."""
        return sorted(list({s.canonical_name for s in self.skills}))

    def get_skills_by_category(self) -> Dict[str, List[str]]:
        """Return detected canonical skills grouped by taxonomy category."""
        grouped: Dict[str, List[str]] = {}
        for skill in self.skills:
            grouped.setdefault(skill.category, []).append(skill.canonical_name)
        # De-duplicate and sort
        for cat in grouped:
            grouped[cat] = sorted(list(set(grouped[cat])))
        return grouped

    def to_dict(self) -> Dict[str, Any]:
        """Convert features payload to a serializable dictionary."""
        return {
            "clean_text": self.clean_text,
            "tokens": self.tokens,
            "sentences": self.sentences,
            "keywords": self.keywords,
            "keyphrases": self.keyphrases,
            "skills": [asdict(s) for s in self.skills],
            "skill_names": self.get_skill_names(),
            "skills_by_category": self.get_skills_by_category(),
            "statistics": asdict(self.statistics),
            "metadata": self.metadata,
        }
