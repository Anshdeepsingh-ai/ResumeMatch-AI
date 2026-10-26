"""
Comprehensive unit tests for the ResumeMatch AI NLP feature extraction module.
Course: CSE472 Deep Learning for NLP
Milestone: Day 4 NLP Feature Extraction
"""

import pytest
from src.nlp import (
    DEFAULT_SKILL_TAXONOMY,
    DocumentStatistics,
    NLPFeatureExtractor,
    NLPFeatures,
    SkillExtractor,
    SkillMention,
    extract_candidate_keyphrases,
    extract_candidate_keywords,
    extract_features,
    segment_sentences,
    tokenize,
)


# ==============================================================================
# 1. Normal and Empty Documents Tests
# ==============================================================================


class TestDocumentHandling:
    """Tests for normal documents, empty documents, and whitespace inputs."""

    def test_normal_resume_extraction(self):
        """Verify feature extraction on a standard synthetic resume."""
        resume_text = (
            "Alex Mercer, M.Sc.\n"
            "Senior NLP Engineer\n"
            "• Developed deep learning models using PyTorch and Python 3.12.\n"
            "• Deployed scalable services to AWS with Docker and Kubernetes."
        )
        features = extract_features(resume_text)

        assert isinstance(features, NLPFeatures)
        assert features.statistics.word_count > 15
        assert features.statistics.token_count > 15
        assert features.statistics.sentence_count >= 3
        assert features.statistics.skill_count >= 4
        assert "Python" in features.get_skill_names()
        assert "PyTorch" in features.get_skill_names()
        assert "AWS" in features.get_skill_names()
        assert "Docker" in features.get_skill_names()

    def test_normal_job_description_extraction(self):
        """Verify feature extraction on a synthetic job description."""
        jd_text = (
            "Job Title: Senior Machine Learning Engineer\n"
            "Requirements:\n"
            "• 5+ years building NLP and deep learning systems.\n"
            "• Strong proficiency in Python, C++, and PyTorch.\n"
            "• Hands-on experience with Docker, Kubernetes, and CI/CD pipelines.\n"
            "• Experience with PostgreSQL or MongoDB databases."
        )
        features = extract_features(jd_text)

        assert "Python" in features.get_skill_names()
        assert "C++" in features.get_skill_names()
        assert "CI/CD" in features.get_skill_names()
        assert "PostgreSQL" in features.get_skill_names()
        assert features.statistics.paragraph_count >= 1

    @pytest.mark.parametrize("empty_input", ["", "   ", "\t", "\n\n", "  \r\n\t  \n  "])
    def test_empty_and_whitespace_documents(self, empty_input):
        """Verify empty and whitespace inputs yield clean zeroed features without errors."""
        features = extract_features(empty_input)

        assert features.clean_text == ""
        assert features.tokens == []
        assert features.sentences == []
        assert features.keywords == []
        assert features.keyphrases == []
        assert features.skills == []
        assert features.statistics.character_count == 0
        assert features.statistics.word_count == 0
        assert features.statistics.token_count == 0
        assert features.statistics.sentence_count == 0
        assert features.statistics.lexical_diversity == 0.0
        assert features.statistics.avg_sentence_length == 0.0


# ==============================================================================
# 2. Technical Terms and Version Numbers Tests
# ==============================================================================


class TestTechnicalTokens:
    """Tests verifying preservation of programming languages, frameworks, and versions."""

    def test_cpp_and_csharp_preservation(self):
        """Verify C++ and C# are tokenized as atomic units and not truncated to C."""
        text = "Extensive background in C++ and C# enterprise development."
        tokens = tokenize(text)
        assert "C++" in tokens
        assert "C#" in tokens
        assert "C" not in tokens

    def test_dotnet_and_nodejs_preservation(self):
        """Verify .NET, ASP.NET, and Node.js are preserved with punctuation."""
        text = "Built backend APIs using .NET Core, ASP.NET, and Node.js."
        tokens = tokenize(text)
        assert ".NET" in tokens
        assert "ASP.NET" in tokens
        assert "Node.js" in tokens

    def test_version_numbers_and_slashes(self):
        """Verify version numbers, CI/CD, and TCP/IP are preserved."""
        text = "Python 3.12.14, PyTorch v2.1.0, CI/CD, TCP/IP protocol."
        tokens = tokenize(text)
        assert "Python" in tokens
        assert "3.12.14" in tokens
        assert "v2.1.0" in tokens
        assert "CI/CD" in tokens
        assert "TCP/IP" in tokens


# ==============================================================================
# 3. Sentence Segmentation Tests
# ==============================================================================


class TestSentenceSegmentation:
    """Tests verifying sentence and bullet item segmentation."""

    def test_bullet_items_as_distinct_sentences(self):
        """Verify bullet points are treated as distinct logical sentence units."""
        text = (
            "Experience:\n"
            "• Led development of semantic search engine.\n"
            "• Implemented dense vector retrieval using Sentence-BERT.\n"
            "• Achieved 99.5% test coverage."
        )
        sentences = segment_sentences(text)
        assert len(sentences) == 4
        assert sentences[1] == "• Led development of semantic search engine."
        assert sentences[2] == "• Implemented dense vector retrieval using Sentence-BERT."
        assert sentences[3] == "• Achieved 99.5% test coverage."

    def test_abbreviations_do_not_break_sentences(self):
        """Verify abbreviations like Ph.D., e.g., and Dr. do not cause false sentence splits."""
        text = "Dr. Jane Doe, Ph.D. presented findings at ACL. The paper was accepted!"
        sentences = segment_sentences(text)
        assert len(sentences) == 2
        assert "Dr. Jane Doe, Ph.D. presented findings at ACL." in sentences[0]


# ==============================================================================
# 4. Keyword and Phrase Candidate Extraction Tests
# ==============================================================================


class TestKeywordAndKeyphraseExtraction:
    """Tests verifying stopword removal, candidate keywords, and n-grams."""

    def test_stopword_filtering(self):
        """Verify common stopwords are omitted from candidate keywords."""
        tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        keywords = extract_candidate_keywords(tokens, top_k=10)
        kw_names = [k[0] for k in keywords]
        assert "the" not in kw_names
        assert "over" not in kw_names
        assert "quick" in kw_names
        assert "fox" in kw_names

    def test_keyphrase_extraction(self):
        """Verify 2-gram and 3-gram keyphrases are correctly extracted and ranked."""
        tokens = [
            "natural", "language", "processing", "engineer", "working", "on",
            "deep", "learning", "and", "natural", "language", "processing"
        ]
        keyphrases = extract_candidate_keyphrases(tokens, top_k=10)
        phrase_names = [kp[0] for kp in keyphrases]

        assert "natural language processing" in phrase_names
        assert "deep learning" in phrase_names
        # Top phrase should be natural language processing (occurs twice)
        nlp_phrase = next(kp for kp in keyphrases if kp[0] == "natural language processing")
        assert nlp_phrase[1] == 2


# ==============================================================================
# 5. Skill Vocabulary Matching Tests
# ==============================================================================


class TestSkillVocabularyMatching:
    """Tests verifying skill detection, alias resolution, and single-letter boundaries."""

    def test_exact_canonical_skills(self):
        """Verify canonical skills match with match_type='exact'."""
        text = "Proficient in Python, Docker, PyTorch, and PostgreSQL."
        extractor = SkillExtractor()
        skills = extractor.extract_skills(text)

        skill_map = {s.canonical_name: s for s in skills}
        assert "Python" in skill_map
        assert skill_map["Python"].match_type == "exact"
        assert "Docker" in skill_map
        assert skill_map["Docker"].category == "cloud_and_devops"

    def test_alias_skill_resolution(self):
        """Verify aliases map to canonical skill names with match_type='alias'."""
        text = "Experienced with k8s clusters and postgres databases."
        extractor = SkillExtractor()
        skills = extractor.extract_skills(text)

        skill_map = {s.canonical_name: s for s in skills}
        assert "Kubernetes" in skill_map
        assert skill_map["Kubernetes"].match_type == "alias"
        assert "PostgreSQL" in skill_map
        assert skill_map["PostgreSQL"].match_type == "alias"

    def test_single_letter_language_c_precision(self):
        """Verify 'C' is matched when standalone, but NOT falsely on 'C++' or 'C#'."""
        # 1. Text with only C++ and C#
        text_cpp = "Expert in C++ and C#."
        skills_cpp = SkillExtractor().extract_skills(text_cpp)
        names_cpp = [s.canonical_name for s in skills_cpp]
        assert "C++" in names_cpp
        assert "C#" in names_cpp
        assert "C" not in names_cpp

        # 2. Text with standalone C
        text_c = "Experienced in C, Python, and C++."
        skills_c = SkillExtractor().extract_skills(text_c)
        names_c = [s.canonical_name for s in skills_c]
        assert "C" in names_c
        assert "C++" in names_c
        assert "Python" in names_c

    def test_single_letter_language_r_precision(self):
        """Verify 'R' is not falsely matched on 'R&D', but matches standalone 'R'."""
        text_rd = "Worked in R&D building ML algorithms."
        skills_rd = SkillExtractor().extract_skills(text_rd)
        names_rd = [s.canonical_name for s in skills_rd]
        assert "R" not in names_rd
        assert "Machine Learning" in names_rd

        text_r = "Statistical modeling in R and Python."
        skills_r = SkillExtractor().extract_skills(text_r)
        names_r = [s.canonical_name for s in skills_r]
        assert "R" in names_r
        assert "Python" in names_r

    def test_custom_skill_taxonomy(self):
        """Verify injecting a custom taxonomy configures extraction properly."""
        custom_taxonomy = {
            "Solidity": {
                "category": "blockchain",
                "aliases": ["ethereum contract"],
                "case_sensitive": False,
            },
            "GraphQL": {
                "category": "api",
                "aliases": ["gql"],
                "case_sensitive": False,
            },
        }
        extractor = SkillExtractor(taxonomy=custom_taxonomy)
        skills = extractor.extract_skills("Created gql APIs and Solidity contracts.")

        skill_map = {s.canonical_name: s for s in skills}
        assert "GraphQL" in skill_map
        assert skill_map["GraphQL"].category == "api"
        assert skill_map["GraphQL"].match_type == "alias"
        assert "Solidity" in skill_map
        assert skill_map["Solidity"].category == "blockchain"
        assert skill_map["Solidity"].match_type == "exact"
        # Standard skills not in custom taxonomy should not appear
        assert "Python" not in skill_map


# ==============================================================================
# 6. Unicode, Punctuation, and Repeated Words Tests
# ==============================================================================


class TestUnicodeAndRepeatedWords:
    """Tests verifying Unicode handling, repeated words, and document statistics."""

    def test_unicode_and_accents(self):
        """Verify Unicode accented text is preserved and processed correctly."""
        text = "François Müller, Ph.D. — Zürich résumé with café."
        features = extract_features(text)
        assert "François" in features.clean_text
        assert "Müller" in features.clean_text
        assert features.statistics.character_count > 0

    def test_repeated_terms_statistics(self):
        """Verify repeated terms are correctly reflected in frequency counts and lexical diversity."""
        text = "Python Python Python Docker Docker"
        features = extract_features(text)

        assert features.statistics.word_count == 5
        assert features.statistics.unique_word_count == 2
        assert features.statistics.lexical_diversity == 0.4

        kw_map = dict(features.keywords)
        assert kw_map.get("python") == 3
        assert kw_map.get("docker") == 2

        skill_map = {s.canonical_name: s for s in features.skills}
        assert skill_map["Python"].count == 3
        assert skill_map["Docker"].count == 2


# ==============================================================================
# 7. Type Validation and Determinism Tests
# ==============================================================================


class TestValidationAndDeterminism:
    """Tests verifying strict type validation and deterministic output."""

    @pytest.mark.parametrize("invalid_input", [None, 123, 4.56, ["str"], {"text": "val"}])
    def test_invalid_type_raises_type_error(self, invalid_input):
        """Verify non-string inputs raise TypeError explicitly."""
        with pytest.raises(TypeError, match="Input text must be a string"):
            extract_features(invalid_input)

        with pytest.raises(TypeError, match="Input text must be a string"):
            tokenize(invalid_input)

        with pytest.raises(TypeError, match="Input text must be a string"):
            segment_sentences(invalid_input)

    def test_deterministic_output(self):
        """Verify running extract_features on identical text yields strictly equal outputs."""
        sample = (
            "Senior NLP Engineer\n"
            "• Built retrieval systems using PyTorch and Sentence-BERT.\n"
            "• Managed Kubernetes clusters on AWS (2020 - 2024)."
        )
        f1 = extract_features(sample)
        f2 = extract_features(sample)

        assert f1.to_dict() == f2.to_dict()
        assert f1.get_skill_names() == f2.get_skill_names()
        assert f1.get_skills_by_category() == f2.get_skills_by_category()


# ==============================================================================
# 8. Integration with Existing Preprocessing Module Tests
# ==============================================================================


class TestPreprocessingIntegration:
    """Tests verifying seamless integration with src.preprocessing."""

    def test_unnormalized_raw_text_integrated(self):
        """Verify raw text with CRLF, wrapped hyphens, and dividers is preprocessed cleanly."""
        raw_text = (
            "EXPERIENCE\r\n"
            "--------------------\r\n"
            "● Built micro-\r\n"
            "services using Node.js and C++.\r\n"
            "\x00\x07"  # control bytes
            "• Managed CI/CD pipelines.\r\n"
        )
        features = extract_features(raw_text, preprocess=True)

        assert "--------------------" not in features.clean_text
        assert "microservices" in features.clean_text  # dehyphenated
        assert "\x00" not in features.clean_text
        assert "Node.js" in features.get_skill_names()
        assert "C++" in features.get_skill_names()
        assert "CI/CD" in features.get_skill_names()

    def test_preprocess_flag_disabled(self):
        """Verify preprocess=False uses provided text directly without normalization."""
        raw_text = "Python Developer."
        features = extract_features(raw_text, preprocess=False)
        assert features.clean_text == "Python Developer."
        assert "Python" in features.get_skill_names()
