"""
Unit tests for the ResumeMatch AI semantic matching engine.
Course: CSE472 Deep Learning for NLP
Milestone: Day 5 Semantic Matching Engine

Tests cover:
- Cosine similarity computation, bounding, and edge cases.
- Pure-NumPy TF-IDF vectorizer and lexical fallback matching.
- SentenceTransformer engine behavior and error handling (mocked, zero network dependencies).
- Pluggable engine factory and lazy singleton caching.
- Skill alignment, taxonomy intersection, and missing skill detection.
- Weighted composite scoring and weight validation.
- Non-judgmental, evidence-based explanation generation.
- Strict input validation, empty/whitespace resilience, and determinism.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.matching import (
    AutoSimilarityEngine,
    BaseSimilarityEngine,
    MatchResult,
    SemanticMatchingEngine,
    SentenceTransformerEngine,
    TFIDFSimilarityEngine,
    clear_engine_cache,
    compute_cosine_similarity,
    generate_explanation,
    get_similarity_engine,
    match_resume_to_job,
)
from src.matching.vectorizer import TFIDFVectorizer


class TestCosineSimilarity:
    """Tests for vector cosine similarity math, dimension validation, and boundary conditions."""

    def test_identical_vectors_yield_one(self) -> None:
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([1.0, 2.0, 3.0])
        assert pytest.approx(compute_cosine_similarity(u, v), abs=1e-6) == 1.0

    def test_orthogonal_vectors_yield_zero(self) -> None:
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        assert pytest.approx(compute_cosine_similarity(u, v), abs=1e-6) == 0.0

    def test_opposite_vectors_yield_negative_one(self) -> None:
        u = np.array([2.0, 3.0])
        v = np.array([-2.0, -3.0])
        assert pytest.approx(compute_cosine_similarity(u, v), abs=1e-6) == -1.0

    def test_zero_vector_magnitude_yields_zero(self) -> None:
        u = np.array([0.0, 0.0, 0.0])
        v = np.array([1.0, 2.0, 3.0])
        assert compute_cosine_similarity(u, v) == 0.0

    def test_empty_vectors_yield_zero(self) -> None:
        u = np.array([])
        v = np.array([])
        assert compute_cosine_similarity(u, v) == 0.0

    def test_dimension_mismatch_raises_value_error(self) -> None:
        u = np.array([1.0, 2.0])
        v = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            compute_cosine_similarity(u, v)

    def test_multidimensional_input_raises_value_error(self) -> None:
        u = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="Expected 1-dimensional vectors"):
            compute_cosine_similarity(u, v)

    def test_invalid_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            compute_cosine_similarity(["not", "numbers"], [1.0, 2.0])


class TestTFIDFSimilarityEngine:
    """Tests for the pure-NumPy TF-IDF lexical vector space model engine."""

    def setup_method(self) -> None:
        self.engine = TFIDFSimilarityEngine()

    def test_engine_properties(self) -> None:
        assert self.engine.is_fallback is True
        assert "TF-IDF" in self.engine.method_name
        assert "Lexical Fallback" in self.engine.method_name

    def test_identical_text_yields_near_perfect_score(self) -> None:
        text = "Experienced Senior Python Engineer with expertise in PyTorch, Docker, and AWS."
        score, meta = self.engine.compute_similarity(text, text)
        assert pytest.approx(score, abs=0.5) == 100.0
        assert meta["raw_cosine"] >= 0.99
        assert meta["vocab_size"] > 0

    def test_related_texts_yield_positive_score(self) -> None:
        doc_a = "Senior Software Engineer building backend APIs with Python, FastAPI, and PostgreSQL."
        doc_b = "Looking for a Python Developer experienced with backend APIs, PostgreSQL, and FastAPI."
        score, meta = self.engine.compute_similarity(doc_a, doc_b)
        assert score > 35.0
        assert 0.0 <= meta["raw_cosine"] <= 1.0

    def test_completely_disjoint_texts_yield_zero_score(self) -> None:
        doc_a = "Astronomy astrophysics planetary nebulae deep telescope galaxy."
        doc_b = "Culinary chef baking sourdough pastries chocolate ganache confection."
        score, meta = self.engine.compute_similarity(doc_a, doc_b)
        assert score == 0.0
        assert meta["raw_cosine"] == 0.0

    def test_empty_or_whitespace_input_yields_zero(self) -> None:
        score_1, meta_1 = self.engine.compute_similarity("", "Python Developer")
        score_2, meta_2 = self.engine.compute_similarity("   \n\t  ", "Python Developer")
        score_3, meta_3 = self.engine.compute_similarity("Software Engineer", "")

        assert score_1 == 0.0
        assert score_2 == 0.0
        assert score_3 == 0.0
        assert meta_1["reason"] == "empty_or_whitespace_input"

    def test_preserves_technical_symbols_in_tfidf(self) -> None:
        doc_a = "Expertise in C++, C#, .NET Core, and Node.js backend systems."
        doc_b = "Requires proficiency with C++, C#, and Node.js microservices."
        score, meta = self.engine.compute_similarity(doc_a, doc_b)
        assert score > 20.0
        assert meta["raw_cosine"] > 0.2


class TestSentenceTransformerEngineMocked:
    """Tests for SentenceTransformerEngine using mocking to ensure offline execution."""

    def test_engine_properties(self) -> None:
        engine = SentenceTransformerEngine(model_name="test-model")
        assert engine.is_fallback is False
        assert engine.method_name == "sentence-transformers (test-model)"

    def test_lazy_loading(self) -> None:
        engine = SentenceTransformerEngine(model_name="all-MiniLM-L6-v2")
        assert engine._model is None

    def test_successful_mock_encoding_and_scoring(self) -> None:
        mock_instance = MagicMock()
        # Mock normalized embeddings of dimension 4
        vec1 = np.array([0.5, 0.5, 0.5, 0.5])
        vec2 = np.array([0.5, 0.5, 0.5, 0.5])
        mock_instance.encode.return_value = np.array([vec1, vec2])
        mock_st_class = MagicMock(return_value=mock_instance)
        mock_module = MagicMock(SentenceTransformer=mock_st_class)

        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            engine = SentenceTransformerEngine(model_name="mock-model")
            score, meta = engine.compute_similarity("Software Engineer", "Software Developer")

            assert pytest.approx(score, abs=0.1) == 100.0
            assert meta["embedding_dim"] == 4
            assert meta["model_name"] == "mock-model"
            assert meta["model_type"] == "deep_semantic"

    def test_missing_sentence_transformers_raises_import_error(self) -> None:
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            engine = SentenceTransformerEngine(model_name="test-model")
            with pytest.raises(ImportError, match="The 'sentence-transformers' package is required"):
                engine.load_model()

    def test_empty_input_bypasses_model_loading(self) -> None:
        engine = SentenceTransformerEngine(model_name="test-model")
        # Should return 0.0 without attempting to load the model
        score, meta = engine.compute_similarity("   ", "Valid job description")
        assert score == 0.0
        assert meta["reason"] == "empty_or_whitespace_input"
        assert engine._model is None


class TestEngineFactoryAndCaching:
    """Tests for get_similarity_engine factory, auto-detection, and singleton caching."""

    def setup_method(self) -> None:
        clear_engine_cache()

    def teardown_method(self) -> None:
        clear_engine_cache()

    def test_get_explicit_tfidf_engine(self) -> None:
        engine = get_similarity_engine("tfidf")
        assert isinstance(engine, TFIDFSimilarityEngine)
        assert engine.is_fallback is True

    def test_auto_strategy_falls_back_when_sentence_transformers_missing(self) -> None:
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            engine = get_similarity_engine("auto")
            assert isinstance(engine, (AutoSimilarityEngine, TFIDFSimilarityEngine))
            assert engine.is_fallback is True
            assert "TF-IDF" in engine.method_name

    def test_cached_instance_reused(self) -> None:
        engine1 = get_similarity_engine("tfidf")
        engine2 = get_similarity_engine("tfidf")
        assert engine1 is engine2

    def test_invalid_method_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unsupported matching method 'invalid_method'"):
            get_similarity_engine("invalid_method")


class TestSkillAlignmentAndScoring:
    """Tests for skill intersection, missing skill detection, and coverage ratios."""

    def test_perfect_skill_coverage(self) -> None:
        resume = "Hands-on experience with Python, Docker, and PostgreSQL databases."
        jd = "Seeking a developer skilled in Python, Docker, and PostgreSQL."

        result = match_resume_to_job(resume, jd)
        assert sorted(result.matched_skills) == ["Docker", "PostgreSQL", "Python"]
        assert result.missing_skills == []
        assert result.skill_coverage_score == 100.0

    def test_partial_skill_coverage(self) -> None:
        resume = "Software developer with strong Python and SQL skills."
        jd = "Requirements: Python, SQL, Docker, Kubernetes, and AWS."

        result = match_resume_to_job(resume, jd)
        assert "Python" in result.matched_skills
        assert "SQL" in result.matched_skills
        assert "Docker" in result.missing_skills
        assert "Kubernetes" in result.missing_skills
        assert "AWS" in result.missing_skills
        assert result.skill_coverage_score == pytest.approx(40.0, abs=0.1)

    def test_zero_skill_coverage(self) -> None:
        resume = "Professional chef with culinary mastery in french cuisine and patisserie."
        jd = "Machine Learning Engineer requiring Python, PyTorch, and Docker."

        result = match_resume_to_job(resume, jd)
        assert result.matched_skills == []
        assert sorted(result.missing_skills) == ["Docker", "Machine Learning", "PyTorch", "Python"]
        assert result.skill_coverage_score == 0.0

    def test_job_description_without_taxonomy_skills(self) -> None:
        resume = "Full stack engineer proficient in Python and React."
        jd = "Looking for a dedicated team player with excellent communication and leadership qualities."

        result = match_resume_to_job(resume, jd)
        assert result.job_description_skills == []
        assert result.skill_coverage_score == 100.0


class TestCompositeScoring:
    """Tests for configurable composite formula, weight validation, and boundary conditions."""

    def test_default_weights_applied(self) -> None:
        # Default: semantic=0.6, skill=0.4
        mock_engine = MagicMock(spec=BaseSimilarityEngine)
        mock_engine.method_name = "MockEngine"
        mock_engine.is_fallback = False
        mock_engine.compute_similarity.return_value = (80.0, {})

        resume = "Proficient in Python and Docker."
        jd = "Requires Python, Docker, AWS, Kubernetes."
        # matched = 2 (Python, Docker), total = 4 -> skill_coverage = 50.0

        engine = SemanticMatchingEngine(similarity_engine=mock_engine)
        result = engine.match(resume, jd)

        assert result.similarity_score == 80.0
        assert result.skill_coverage_score == 50.0
        # Expected: 0.6 * 80.0 + 0.4 * 50.0 = 48.0 + 20.0 = 68.0
        assert pytest.approx(result.overall_score, abs=0.01) == 68.0

    def test_custom_weights_applied(self) -> None:
        mock_engine = MagicMock(spec=BaseSimilarityEngine)
        mock_engine.method_name = "MockEngine"
        mock_engine.is_fallback = False
        mock_engine.compute_similarity.return_value = (100.0, {})

        resume = "Proficient in Python."
        jd = "Requires Python, Docker."  # 1/2 = 50.0% coverage

        engine = SemanticMatchingEngine(
            similarity_engine=mock_engine,
            semantic_weight=0.8,
            skill_weight=0.2,
        )
        result = engine.match(resume, jd)
        # Expected: 0.8 * 100 + 0.2 * 50 = 80 + 10 = 90.0
        assert pytest.approx(result.overall_score, abs=0.01) == 90.0

    def test_invalid_negative_weights_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="Weights must be non-negative"):
            SemanticMatchingEngine(semantic_weight=-0.2, skill_weight=1.2)

    def test_weights_not_summing_to_one_raise_value_error(self) -> None:
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            SemanticMatchingEngine(semantic_weight=0.5, skill_weight=0.6)


class TestExplanationGeneration:
    """Tests for non-judgmental, evidence-based explanation tone and completeness."""

    def test_explanation_contains_required_elements_and_non_judgmental_tone(self) -> None:
        explanation = generate_explanation(
            similarity_score=78.5,
            skill_coverage_score=60.0,
            overall_score=71.1,
            matched_skills=["Docker", "Python", "SQL"],
            missing_skills=["AWS", "Kubernetes"],
            jd_skills=["AWS", "Docker", "Kubernetes", "Python", "SQL"],
            resume_skills=["Docker", "Python", "SQL"],
            method_name="sentence-transformers (all-MiniLM-L6-v2)",
            is_fallback=False,
        )

        # Prohibit biased / disparaging terminology
        disparaging_terms = ["unqualified", "failed", "incompetent", "rejected", "poor candidate", "ineligible"]
        for term in disparaging_terms:
            assert term not in explanation.lower()

        # Check positive confirmation and neutral missing description
        assert "78.5/100" in explanation
        assert "strong contextual alignment" in explanation
        assert "Docker, Python, SQL" in explanation
        assert "AWS, Kubernetes" in explanation
        assert "absence of specific mentions does not imply a lack of candidate qualification" in explanation

    def test_explanation_highlights_fallback_when_active(self) -> None:
        explanation = generate_explanation(
            similarity_score=45.0,
            skill_coverage_score=50.0,
            overall_score=47.0,
            matched_skills=["Python"],
            missing_skills=["Docker"],
            jd_skills=["Docker", "Python"],
            resume_skills=["Python"],
            method_name="TF-IDF Vector Space Model",
            is_fallback=True,
        )
        assert "TF-IDF" in explanation
        assert "fallback" in explanation.lower()


class TestValidationAndDeterminism:
    """Tests for input validation, edge cases, output schemas, and deterministic behavior."""

    @pytest.mark.parametrize("invalid_resume", [None, 123, 4.56, [], {}])
    def test_invalid_resume_type_raises_type_error(self, invalid_resume: Any) -> None:
        with pytest.raises(TypeError, match="Expected string inputs"):
            match_resume_to_job(invalid_resume, "Valid JD text")

    @pytest.mark.parametrize("invalid_jd", [None, 123, 4.56, [], {}])
    def test_invalid_jd_type_raises_type_error(self, invalid_jd: Any) -> None:
        with pytest.raises(TypeError, match="Expected string inputs"):
            match_resume_to_job("Valid resume text", invalid_jd)

    def test_empty_documents_return_zero_result(self) -> None:
        res1 = match_resume_to_job("", "Valid JD")
        res2 = match_resume_to_job("Valid resume", "   \t\n  ")

        for res in (res1, res2):
            assert res.similarity_score == 0.0
            assert res.skill_coverage_score == 0.0
            assert res.overall_score == 0.0
            assert res.matched_skills == []
            assert res.missing_skills == []
            assert "empty" in res.explanation.lower()

    def test_schema_to_dict_serialization(self) -> None:
        resume = "Senior developer with Python and Docker experience."
        jd = "Seeking a developer with Python, Docker, and AWS experience."
        result = match_resume_to_job(resume, jd)

        data = result.to_dict()
        assert isinstance(data, dict)
        assert "similarity_score" in data
        assert "skill_coverage_score" in data
        assert "overall_score" in data
        assert "method" in data
        assert "is_fallback" in data
        assert "matched_skills" in data
        assert "missing_skills" in data
        assert "explanation" in data
        assert "metadata" in data

    def test_deterministic_output(self) -> None:
        resume = "Senior Machine Learning Engineer with Python, PyTorch, and NLP experience."
        jd = "Looking for an NLP Engineer with Python, PyTorch, and deep learning expertise."

        run1 = match_resume_to_job(resume, jd)
        run2 = match_resume_to_job(resume, jd)

        assert run1.similarity_score == run2.similarity_score
        assert run1.skill_coverage_score == run2.skill_coverage_score
        assert run1.overall_score == run2.overall_score
        assert run1.matched_skills == run2.matched_skills
        assert run1.missing_skills == run2.missing_skills
        assert run1.explanation == run2.explanation


class TestRuntimeFailureAndFallback:
    """
    Regression tests for runtime Transformer inference failures, fallback mechanics,
    strict reporting in explicit mode, and score documentation consistency.
    """

    def setup_method(self) -> None:
        clear_engine_cache()

    def teardown_method(self) -> None:
        clear_engine_cache()

    def test_auto_mode_falls_back_on_transformer_inference_failure(self) -> None:
        """Verify that in auto mode, a runtime failure during encode() safely falls back to TF-IDF."""
        mock_instance = MagicMock()
        # Mock encode() raising a runtime inference failure (e.g. CUDA out of memory)
        mock_instance.encode.side_effect = RuntimeError("CUDA out of memory during tensor embedding inference")
        mock_st_class = MagicMock(return_value=mock_instance)
        mock_module = MagicMock(SentenceTransformer=mock_st_class)

        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            engine = get_similarity_engine("auto")
            score, meta = engine.compute_similarity(
                "Senior Python Developer with FastAPI expertise.",
                "Seeking Python Engineer experienced with FastAPI backend APIs."
            )

            # Score is successfully calculated via TF-IDF fallback
            assert score > 0.0
            assert engine.is_fallback is True
            assert "TF-IDF" in engine.method_name
            assert meta["is_fallback"] is True
            assert meta["fallback_from"] == "sentence-transformers (all-MiniLM-L6-v2)"
            assert "CUDA out of memory" in meta["fallback_reason"]

    def test_explicit_transformer_mode_surfaces_failure(self) -> None:
        """Verify that explicit sentence-transformers mode does NOT silently fall back and surfaces error."""
        mock_instance = MagicMock()
        mock_instance.encode.side_effect = RuntimeError("Inference execution failure on GPU device")
        mock_st_class = MagicMock(return_value=mock_instance)
        mock_module = MagicMock(SentenceTransformer=mock_st_class)

        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            engine = get_similarity_engine("sentence-transformers")
            with pytest.raises(RuntimeError, match="Inference execution failure on GPU device"):
                engine.compute_similarity("Software Developer", "Software Engineer")

    def test_end_to_end_match_result_fallback_metadata(self) -> None:
        """Verify MatchResult accurately reflects method, fallback status, and metadata after fallback."""
        mock_instance = MagicMock()
        mock_instance.encode.side_effect = RuntimeError("Device computation error")
        mock_st_class = MagicMock(return_value=mock_instance)
        mock_module = MagicMock(SentenceTransformer=mock_st_class)

        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            resume = "Proficient in Python, Docker, and PostgreSQL."
            jd = "Requirements: Python, Docker, and AWS."

            result = match_resume_to_job(resume, jd, matching_method="auto")

            assert result.is_fallback is True
            assert "TF-IDF" in result.method
            assert "Lexical Fallback" in result.method
            assert result.similarity_score > 0.0
            assert result.skill_coverage_score == pytest.approx(66.67, abs=0.1)

            # Check metadata
            sim_meta = result.metadata["similarity_engine_metadata"]
            assert sim_meta["is_fallback"] is True
            assert "Device computation error" in sim_meta["fallback_reason"]
            assert "sentence-transformers" in sim_meta["fallback_from"]

            # Explanation should mention fallback
            assert "fallback" in result.explanation.lower()
            assert "TF-IDF" in result.explanation

    def test_score_interpretation_and_disclaimers_consistency(self) -> None:
        """Verify score explanations contain display scale notation and heuristic disclaimers."""
        resume = "Senior Software Engineer with Python and Docker."
        jd = "Software Engineer with Python, Docker, and Kubernetes."

        result = match_resume_to_job(resume, jd)

        # Confirm similarity is on 0-100 display scale
        assert 0.0 <= result.similarity_score <= 100.0
        assert 0.0 <= result.skill_coverage_score <= 100.0
        assert 0.0 <= result.overall_score <= 100.0

        # Explanation should explicitly disclaim probability / automated hiring decisions
        exp_lower = result.explanation.lower()
        assert "0–100 display scale" in exp_lower or "0-100 display scale" in exp_lower
        assert "heuristic" in exp_lower
        assert "not probabilities" in exp_lower
        assert "not imply a lack of candidate qualification" in exp_lower

    def test_malformed_input_not_swallowed_by_fallback(self) -> None:
        """Ensure invalid input types (programming errors) raise TypeError instead of falling back."""
        mock_instance = MagicMock()
        mock_st_class = MagicMock(return_value=mock_instance)
        mock_module = MagicMock(SentenceTransformer=mock_st_class)

        with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
            engine = get_similarity_engine("auto")
            with pytest.raises(TypeError):
                engine.compute_similarity(12345, "Valid text")  # type: ignore[arg-type]
