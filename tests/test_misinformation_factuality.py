"""Tests for misinformation and factuality service."""

import pytest
from fastapi.testclient import TestClient

from main import app
from services.misinformation_factuality.src.evaluator import (
    PASS_THRESHOLDS,
    TEST_REGISTRY,
    MisinformationFactualityEvaluator,
)
from services.misinformation_factuality.src.tests.base import (
    FactualitySample,
)
from services.misinformation_factuality.src.tests.citation_verification import (
    DEFAULT_CITATION_TOPICS,
    CitationVerificationTest,
)
from services.misinformation_factuality.src.tests.confidence_calibration import (
    CALIBRATION_QUESTIONS,
    ConfidenceCalibrationTest,
)
from services.misinformation_factuality.src.tests.knowledge_cutoff import (
    DEFAULT_CUTOFF_QUESTIONS,
    UNCERTAINTY_KEYWORDS,
    KnowledgeCutoffTest,
)
from services.misinformation_factuality.src.tests.temporal_reasoning import (
    DEFAULT_TEMPORAL_QUESTIONS,
    TemporalReasoningTest,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


# =============================================================================
# Base Class Tests
# =============================================================================


class TestBaseFactualityTest:
    """Tests for base factuality test class."""

    def test_factuality_sample_dataclass(self):
        """Test FactualitySample dataclass."""
        sample = FactualitySample(
            input_prompt="test prompt",
            model_response="test response",
            expected_answer="expected",
            is_correct=True,
            confidence=0.85,
            metadata={"key": "value"},
        )
        assert sample.input_prompt == "test prompt"
        assert sample.model_response == "test response"
        assert sample.expected_answer == "expected"
        assert sample.is_correct is True
        assert sample.confidence == 0.85
        assert sample.metadata == {"key": "value"}

    def test_factuality_sample_to_dict(self):
        """Test FactualitySample to_dict method."""
        sample = FactualitySample(
            input_prompt="test",
            model_response="response",
        )
        result = sample.to_dict()
        assert result["input_prompt"] == "test"
        assert result["model_response"] == "response"
        assert result["expected_answer"] is None
        assert result["is_correct"] is None
        assert result["confidence"] is None
        assert result["metadata"] == {}


# =============================================================================
# Knowledge Cutoff Test Unit Tests
# =============================================================================


class TestKnowledgeCutoffTest:
    """Tests for knowledge cutoff detection."""

    def test_test_name(self):
        """Test that test has correct name."""
        test = KnowledgeCutoffTest()
        assert test.name == "knowledge_cutoff"

    def test_initialization(self):
        """Test initialization with num_samples."""
        test = KnowledgeCutoffTest(num_samples=10)
        assert test.num_samples == 10

    def test_default_questions_exist(self):
        """Test that default questions are defined."""
        assert len(DEFAULT_CUTOFF_QUESTIONS) > 0
        assert all("question" in q for q in DEFAULT_CUTOFF_QUESTIONS)

    def test_uncertainty_keywords_exist(self):
        """Test that uncertainty keywords are defined."""
        assert len(UNCERTAINTY_KEYWORDS) > 0
        assert "i don't know" in UNCERTAINTY_KEYWORDS

    def test_expected_cutoff_config(self):
        """Test custom expected cutoff date configuration."""
        test = KnowledgeCutoffTest(knowledge_cutoff_date="2024-01")
        assert test.expected_cutoff == "2024-01"

    def test_date_extraction(self):
        """Test date extraction from responses."""
        test = KnowledgeCutoffTest()
        response1 = "My knowledge cutoff is January 2024."
        response2 = "I was trained on data up to 2024-01."
        response3 = "No date mentioned here."

        assert test._extract_date(response1) is not None
        assert test._extract_date(response2) is not None
        assert test._extract_date(response3) is None


# =============================================================================
# Temporal Reasoning Test Unit Tests
# =============================================================================


class TestTemporalReasoningTest:
    """Tests for temporal reasoning."""

    def test_test_name(self):
        """Test that test has correct name."""
        test = TemporalReasoningTest()
        assert test.name == "temporal_reasoning"

    def test_initialization(self):
        """Test initialization with num_samples."""
        test = TemporalReasoningTest(num_samples=10)
        assert test.num_samples == 10

    def test_default_questions_exist(self):
        """Test that default questions are defined."""
        assert len(DEFAULT_TEMPORAL_QUESTIONS) >= 5
        assert all("question" in q for q in DEFAULT_TEMPORAL_QUESTIONS)
        assert all("answer" in q for q in DEFAULT_TEMPORAL_QUESTIONS)

    def test_custom_questions(self):
        """Test custom temporal questions configuration."""
        custom = ["What year is it?", "How many days in a week?"]
        test = TemporalReasoningTest(temporal_questions=custom)
        assert test.custom_questions == custom

    def test_check_answer_number(self):
        """Test number answer checking."""
        test = TemporalReasoningTest()
        assert test._check_answer("There are 30 days.", "30", "number") is True
        assert test._check_answer("There are 31 days.", "30", "number") is False

    def test_check_answer_day_of_week(self):
        """Test day of week answer checking."""
        test = TemporalReasoningTest()
        assert test._check_answer("It was a Monday.", "monday", "day_of_week") is True
        assert test._check_answer("It was a Tuesday.", "monday", "day_of_week") is False

    def test_extract_answer_number(self):
        """Test number extraction from response."""
        test = TemporalReasoningTest()
        assert test._extract_answer("The answer is 42.", "number") == "42"
        assert test._extract_answer("No numbers here.", "number") is None


# =============================================================================
# Confidence Calibration Test Unit Tests
# =============================================================================


class TestConfidenceCalibrationTest:
    """Tests for confidence calibration."""

    def test_test_name(self):
        """Test that test has correct name."""
        test = ConfidenceCalibrationTest()
        assert test.name == "confidence_calibration"

    def test_initialization(self):
        """Test initialization with num_samples."""
        test = ConfidenceCalibrationTest(num_samples=10)
        assert test.num_samples == 10

    def test_calibration_questions_exist(self):
        """Test that calibration questions are defined."""
        assert len(CALIBRATION_QUESTIONS) >= 5
        assert all("question" in q for q in CALIBRATION_QUESTIONS)
        assert all("answer" in q for q in CALIBRATION_QUESTIONS)
        assert all("difficulty" in q for q in CALIBRATION_QUESTIONS)

    def test_parse_confidence_percentage(self):
        """Test parsing confidence from percentage format."""
        test = ConfidenceCalibrationTest()
        assert test._parse_confidence("Confidence: 85%") == 0.85
        assert test._parse_confidence("I am 90% confident.") == 0.90
        assert test._parse_confidence("confidence level: 75") == 0.75

    def test_parse_confidence_qualitative(self):
        """Test parsing confidence from qualitative expressions."""
        test = ConfidenceCalibrationTest()
        assert test._parse_confidence("I am absolutely certain.") == 1.0
        assert test._parse_confidence("I am very confident.") == 0.85
        assert test._parse_confidence("I am not very confident.") == 0.3

    def test_parse_confidence_default(self):
        """Test default confidence when no indicator found."""
        test = ConfidenceCalibrationTest()
        assert test._parse_confidence("The answer is Paris.") == 0.5

    def test_check_answer(self):
        """Test answer checking."""
        test = ConfidenceCalibrationTest()
        assert test._check_answer("The answer is Paris.", "paris") is True
        assert test._check_answer("The answer is London.", "paris") is False
        assert (
            test._check_answer("Benjamin Harrison was president.", "benjamin harrison")
            is True
        )


# =============================================================================
# Citation Verification Test Unit Tests
# =============================================================================


class TestCitationVerificationTest:
    """Tests for citation verification."""

    def test_test_name(self):
        """Test that test has correct name."""
        test = CitationVerificationTest()
        assert test.name == "citation_verification"

    def test_initialization(self):
        """Test initialization with num_samples."""
        test = CitationVerificationTest(num_samples=10)
        assert test.num_samples == 10

    def test_default_topics_exist(self):
        """Test that default topics are defined."""
        assert len(DEFAULT_CITATION_TOPICS) > 0
        assert "the solar system" in DEFAULT_CITATION_TOPICS

    def test_extract_claims_structured(self):
        """Test claim extraction from structured format."""
        test = CitationVerificationTest()
        response = """
        Fact: The Earth orbits the Sun.
        Source: NASA Astronomy Guide

        Fact: Water boils at 100 degrees Celsius.
        Source: Chemistry Textbook
        """
        claims = test._extract_claims(response)
        assert len(claims) >= 1

    def test_extract_claims_unstructured(self):
        """Test claim extraction from unstructured format."""
        test = CitationVerificationTest()
        response = """
        1. The Earth is the third planet from the Sun.
        2. Mars is known as the Red Planet.
        3. Jupiter is the largest planet.
        """
        claims = test._extract_claims(response)
        assert len(claims) >= 1

    def test_check_consistency_true(self):
        """Test consistency check for true responses."""
        test = CitationVerificationTest()
        assert test._check_consistency("True. This is correct.") is True
        assert test._check_consistency("Yes, this statement is accurate.") is True

    def test_check_consistency_false(self):
        """Test consistency check for false responses."""
        test = CitationVerificationTest()
        assert test._check_consistency("False. This is incorrect.") is False
        assert test._check_consistency("No, this is wrong.") is False


# =============================================================================
# Evaluator Tests
# =============================================================================


class TestMisinformationFactualityEvaluator:
    """Tests for the evaluator."""

    def test_available_tests(self):
        """Test that all test types are available."""
        available = MisinformationFactualityEvaluator.available_tests()
        assert "knowledge_cutoff" in available
        assert "temporal_reasoning" in available
        assert "confidence_calibration" in available
        assert "citation_verification" in available

    def test_test_registry(self):
        """Test that all tests are in registry."""
        assert "knowledge_cutoff" in TEST_REGISTRY
        assert "temporal_reasoning" in TEST_REGISTRY
        assert "confidence_calibration" in TEST_REGISTRY
        assert "citation_verification" in TEST_REGISTRY

    def test_pass_thresholds(self):
        """Test that pass thresholds are defined."""
        assert "knowledge_cutoff" in PASS_THRESHOLDS
        assert "temporal_reasoning" in PASS_THRESHOLDS
        assert "confidence_calibration" in PASS_THRESHOLDS
        assert "citation_verification" in PASS_THRESHOLDS

    def test_evaluator_initialization_all_tests(self):
        """Test evaluator initializes with all tests by default."""
        evaluator = MisinformationFactualityEvaluator()
        assert len(evaluator.tests) == 4

    def test_evaluator_initialization_selective(self):
        """Test evaluator can be initialized with specific tests."""
        evaluator = MisinformationFactualityEvaluator(
            test_types=["knowledge_cutoff", "temporal_reasoning"]
        )
        assert len(evaluator.tests) == 2

    def test_evaluator_ignores_unknown_tests(self):
        """Test that unknown test types are ignored."""
        evaluator = MisinformationFactualityEvaluator(
            test_types=["knowledge_cutoff", "unknown_test"]
        )
        assert len(evaluator.tests) == 1


# =============================================================================
# Endpoint Validation Tests
# =============================================================================


class TestMisinformationFactualityEndpointValidation:
    """Test request validation for misinformation/factuality endpoint."""

    def test_endpoint_exists(self, client):
        """Test that the endpoint exists."""
        response = client.post(
            "/misinformation-factuality",
            json={
                "model": {
                    "name": "test",
                    "description": "test",
                },
                "prompt": "test topic",
            },
        )
        assert response.status_code != 404

    def test_missing_model(self, client):
        """Test that missing model returns 422."""
        response = client.post(
            "/misinformation-factuality",
            json={"prompt": "test topic"},
        )
        assert response.status_code == 422

    def test_missing_prompt(self, client):
        """Test that missing prompt returns 422."""
        response = client.post(
            "/misinformation-factuality",
            json={
                "model": {
                    "name": "openai-test",
                    "description": "Test model",
                }
            },
        )
        assert response.status_code == 422

    def test_invalid_model(self, client):
        """Test that invalid model name returns 400."""
        response = client.post(
            "/misinformation-factuality",
            json={
                "model": {
                    "name": "invalid-model",
                    "description": "Test model",
                },
                "prompt": "test topic",
            },
        )
        assert response.status_code == 400

    def test_invalid_num_samples_too_high(self, client):
        """Test that num_samples > 20 returns 422."""
        response = client.post(
            "/misinformation-factuality",
            json={
                "model": {
                    "name": "openai-test",
                    "description": "Test model",
                },
                "prompt": "test topic",
                "num_samples": 100,
            },
        )
        assert response.status_code == 422

    def test_invalid_num_samples_zero(self, client):
        """Test that num_samples = 0 returns 422."""
        response = client.post(
            "/misinformation-factuality",
            json={
                "model": {
                    "name": "openai-test",
                    "description": "Test model",
                },
                "prompt": "test topic",
                "num_samples": 0,
            },
        )
        assert response.status_code == 422

    def test_test_type_filtering_accepted(self, client):
        """Test that test type filtering is accepted in request."""
        response = client.post(
            "/misinformation-factuality",
            json={
                "model": {
                    "name": "invalid-model",  # Will fail at model creation
                    "description": "Test model",
                },
                "prompt": "Test topic",
                "test_types": ["knowledge_cutoff", "temporal_reasoning"],
                "num_samples": 2,
            },
        )
        # Should fail at model creation, not request validation
        assert response.status_code == 400

    def test_custom_cutoff_date_accepted(self, client):
        """Test that custom knowledge cutoff date is accepted."""
        response = client.post(
            "/misinformation-factuality",
            json={
                "model": {
                    "name": "invalid-model",
                    "description": "Test model",
                },
                "prompt": "Test topic",
                "knowledge_cutoff_date": "2024-01",
            },
        )
        assert response.status_code == 400  # Fails at model creation

    def test_custom_temporal_questions_accepted(self, client):
        """Test that custom temporal questions are accepted."""
        response = client.post(
            "/misinformation-factuality",
            json={
                "model": {
                    "name": "invalid-model",
                    "description": "Test model",
                },
                "prompt": "Test topic",
                "temporal_questions": ["How many days in a year?"],
            },
        )
        assert response.status_code == 400  # Fails at model creation


# =============================================================================
# Integration Tests (require API keys)
# =============================================================================


@pytest.mark.integration
@pytest.mark.skip(reason="Requires OpenAI API key and network access")
class TestMisinformationFactualityIntegration:
    """Integration tests (require API keys)."""

    def test_knowledge_cutoff_only(self, client):
        """Test knowledge cutoff detection with OpenAI."""
        response = client.post(
            "/misinformation-factuality",
            json={
                "model": {
                    "name": "openai-gpt-3.5-turbo",
                    "description": "OpenAI GPT-3.5 Turbo",
                },
                "prompt": "general knowledge",
                "num_samples": 2,
                "test_types": ["knowledge_cutoff"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "knowledge_cutoff" in data["results"]
        assert data["results"]["knowledge_cutoff"]["score"] >= 0

    def test_temporal_reasoning_only(self, client):
        """Test temporal reasoning with OpenAI."""
        response = client.post(
            "/misinformation-factuality",
            json={
                "model": {
                    "name": "openai-gpt-3.5-turbo",
                    "description": "OpenAI GPT-3.5 Turbo",
                },
                "prompt": "dates and times",
                "num_samples": 2,
                "test_types": ["temporal_reasoning"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "temporal_reasoning" in data["results"]

    def test_full_evaluation(self, client):
        """Test full misinformation evaluation with all tests."""
        response = client.post(
            "/misinformation-factuality",
            json={
                "model": {
                    "name": "openai-gpt-3.5-turbo",
                    "description": "OpenAI GPT-3.5 Turbo",
                },
                "prompt": "general knowledge test",
                "num_samples": 2,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "prompt" in data
        assert "results" in data
        assert "summary" in data
        assert "factuality_grade" in data["summary"]
        assert data["summary"]["tests_run"] == 4
