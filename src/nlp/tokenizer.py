"""
Tokenization, sentence segmentation, and keyword extraction module.
Preserves technical terms (C++, C#, .NET, Node.js, version numbers).
Course: CSE472 Deep Learning for NLP
Milestone: Day 4 NLP Feature Extraction
"""

import re
from collections import Counter
from typing import Any, List, Set, Tuple

# Curated English stopwords to avoid external package download dependencies
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "i'd", "i'll", "i'm", "i've", "if",
    "in", "into", "is", "isn't", "it", "it's", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll",
    "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their",
    "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
    "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "when", "where", "which", "while", "who", "whom", "why", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
}

# Protected abbreviation set for sentence segmentation
PROTECTED_ABBREVIATIONS: Set[str] = {
    "dr", "mr", "mrs", "ms", "ph.d", "m.sc", "b.sc", "b.tech", "m.tech", "mba",
    "e.g", "i.e", "inc", "corp", "ltd", "vs", "etc", "approx", "dept", "univ",
}

# Regex pattern specifically tuned for technical resumes and job descriptions
TECHNICAL_TOKEN_PATTERN = re.compile(
    r"(?i)"
    r"\bC\+\+(?!\w)"                              # C++
    r"|\bC\#(?!\w)"                                # C#
    r"|(?:\.NET|\bASP\.NET)(?!\w)"                # .NET, ASP.NET
    r"|\b[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+\b"    # Node.js, Vue.js, v2.1.0, 3.12.14
    r"|\b[A-Za-z0-9]+(?:/[A-Za-z0-9]+)+\b"        # CI/CD, TCP/IP
    r"|\b[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*\b"      # Standard and hyphenated words
)


def _validate_input_str(text: Any) -> None:
    """Validate that input text is a string."""
    if not isinstance(text, str):
        raise TypeError(f"Input text must be a string, got {type(text).__name__}")


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into technical tokens, preserving programming languages and version numbers.

    Args:
        text: Input text string.

    Returns:
        List of extracted string tokens.

    Raises:
        TypeError: If text is not a string.
    """
    _validate_input_str(text)
    if not text or text.isspace():
        return []

    return TECHNICAL_TOKEN_PATTERN.findall(text)


def segment_sentences(text: str) -> List[str]:
    """
    Segment text into logical sentence units, list items, and section headings.
    Protects abbreviations (Ph.D., e.g., Dr.), version numbers, and technical dots.

    Args:
        text: Input text string.

    Returns:
        List of segmented sentences or logical units.

    Raises:
        TypeError: If text is not a string.
    """
    _validate_input_str(text)
    if not text or text.isspace():
        return []

    raw_blocks = [block.strip() for block in text.split("\n") if block.strip()]
    sentences: List[str] = []

    for block in raw_blocks:
        # Preserve bulleted list items and numbered list items as distinct units
        if block.startswith(("•", "-", "*")) or re.match(r"^\d+\.\s", block):
            sentences.append(block)
            continue

        # Split paragraph into sentences on terminal punctuation (.!?) followed by space and uppercase
        candidate_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9•\-])", block)

        merged: List[str] = []
        for part in candidate_parts:
            # Check if previous segment ended with an abbreviation (e.g., 'Ph.D.' or 'Dr.')
            if merged and any(
                merged[-1].lower().rstrip(".").endswith(abbr.rstrip("."))
                for abbr in PROTECTED_ABBREVIATIONS
            ):
                merged[-1] = f"{merged[-1]} {part}"
            else:
                merged.append(part)

        for s in merged:
            s_clean = s.strip()
            if s_clean:
                sentences.append(s_clean)

    return sentences


def extract_candidate_keywords(
    tokens: List[str],
    top_k: int = 20,
    stopwords: Set[str] = DEFAULT_STOPWORDS,
    min_length: int = 2,
) -> List[Tuple[str, int]]:
    """
    Extract frequency-ranked candidate keywords excluding stopwords.

    Args:
        tokens: Sequence of tokens from document.
        top_k: Maximum number of candidate keywords to return.
        stopwords: Set of words to ignore.
        min_length: Minimum character length for candidate keywords (except recognized 1-char symbols).

    Returns:
        List of (keyword, count) tuples sorted by descending frequency.
    """
    filtered = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in stopwords:
            continue
        # Allow technical symbols like C even if 1 char, otherwise enforce min_length
        if len(token) < min_length and token not in {"C", "R"}:
            continue
        filtered.append(token_lower)

    counts = Counter(filtered)
    return counts.most_common(top_k)


def extract_candidate_keyphrases(
    tokens: List[str],
    top_k: int = 15,
    stopwords: Set[str] = DEFAULT_STOPWORDS,
) -> List[Tuple[str, int]]:
    """
    Extract multi-word candidate keyphrases (2-grams and 3-grams) bounded by content words.

    Args:
        tokens: Sequence of tokens from document.
        top_k: Maximum number of keyphrases to return.
        stopwords: Set of stopwords.

    Returns:
        List of (keyphrase, count) tuples sorted by descending frequency.
    """
    if len(tokens) < 2:
        return []

    phrases = []

    # 1. Bigrams
    for i in range(len(tokens) - 1):
        t1, t2 = tokens[i], tokens[i + 1]
        w1, w2 = t1.lower(), t2.lower()
        # Phrase must not start or end with a stopword
        if w1 not in stopwords and w2 not in stopwords:
            if len(w1) > 1 and len(w2) > 1:
                phrases.append(f"{w1} {w2}")

    # 2. Trigrams
    for i in range(len(tokens) - 2):
        t1, t2, t3 = tokens[i], tokens[i + 1], tokens[i + 2]
        w1, w2, w3 = t1.lower(), t2.lower(), t3.lower()
        # Boundary tokens must not be stopwords; middle token can be preposition (e.g., 'retrieval for generation')
        if w1 not in stopwords and w3 not in stopwords:
            if len(w1) > 1 and len(w3) > 1:
                phrases.append(f"{w1} {w2} {w3}")

    counts = Counter(phrases)
    return counts.most_common(top_k)
