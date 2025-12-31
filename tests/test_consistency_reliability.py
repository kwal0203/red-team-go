"""Tests for consistency and reliability service."""

import pytest
from fastapi.testclient import TestClient

from main import app
from services.consistency_reliability.src.evaluator import (
    PASS_THRESHOLDS,
    TEST_REGISTRY,
    ConsistencyReliabilityEvaluator,
)
from services.consistency_reliability.src.tests.base import EvalSample
from services.consistency_reliability.src.tests.instruction_following import (
    DEFAULT_CONSTRAINTS,
    InstructionFollowingTest,
)
from services.consistency_reliability.src.tests.self_consistency import (
    SelfConsistencyTest,
)
from services.consistency_reliability.src.tests.stability import StabilityTest
from services.consistency_reliability.src.tests.sycophancy import (
    DEFAULT_TOPICS,
    SycophancyTest,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


# =============================================================================
# Base Class Tests
# =============================================================================


class TestBaseConsistencyTest:
    """Tests for base consistency test class."""

    def test_eval_sample_dataclass(self):
        """Test EvalSample dataclass."""
        sample = EvalSample(
            input_prompt="test prompt",
            model_response="test response",
            metadata={"key": "value"},
        )
        assert sample.input_prompt == "test prompt"
        assert sample.model_response == "test response"
        assert sample.metadata == {"key": "value"}

    def test_eval_sample_to_dict(self):
        """Test EvalSample to_dict method."""
        sample = EvalSample(
            input_prompt="test",
            model_response="response",
        )
        result = sample.to_dict()
        assert result["input_prompt"] == "test"
        assert result["model_response"] == "response"
        assert result["metadata"] == {}


# =============================================================================
# Sycophancy Test Unit Tests
# =============================================================================


class TestSycophancyTest:
    """Tests for sycophancy detection."""

    def test_test_name(self):
        """Test that test has correct name."""
        test = SycophancyTest()
        assert test.name == "sycophancy"

    def test_default_topics(self):
        """Test default topics are used."""
        test = SycophancyTest(num_samples=5)
        assert len(test.topics) == 5
        assert test.topics[0] == DEFAULT_TOPICS[0]

    def test_custom_topics(self):
        """Test custom topics can be provided."""
        topics = ["Topic 1", "Topic 2"]
        test = SycophancyTest(sycophancy_topics=topics)
        assert test.topics == topics

    def test_position_extraction_positive(self):
        """Test position extraction for positive stance."""
        test = SycophancyTest()
        response = "Yes, I agree that this is beneficial and we should support it."
        position = test._extract_position(response)
        assert position == "positive"

    def test_position_extraction_negative(self):
        """Test position extraction for negative stance."""
        test = SycophancyTest()
        response = "No, I disagree. This is harmful and we should oppose it."
        position = test._extract_position(response)
        assert position == "negative"

    def test_position_extraction_neutral(self):
        """Test position extraction for neutral stance."""
        test = SycophancyTest()
        response = "There are multiple perspectives to consider here."
        position = test._extract_position(response)
        assert position == "neutral"

    def test_positions_consistent_same(self):
        """Test position consistency for same positions."""
        test = SycophancyTest()
        assert test._positions_consistent("positive", "positive") is True
        assert test._positions_consistent("negative", "negative") is True

    def test_positions_consistent_different(self):
        """Test position consistency for different positions."""
        test = SycophancyTest()
        assert test._positions_consistent("positive", "negative") is False

    def test_positions_consistent_neutral(self):
        """Test that neutral is consistent with anything."""
        test = SycophancyTest()
        assert test._positions_consistent("neutral", "positive") is True
        assert test._positions_consistent("positive", "neutral") is True


# =============================================================================
# Stability Test Unit Tests
# =============================================================================


class TestStabilityTest:
    """Tests for response stability."""

    def test_test_name(self):
        """Test that test has correct name."""
        test = StabilityTest()
        assert test.name == "stability"

    def test_initialization(self):
        """Test initialization with num_samples."""
        test = StabilityTest(num_samples=10)
        assert test.num_samples == 10

    def test_lazy_loading(self):
        """Test that models are not loaded on init."""
        test = StabilityTest()
        assert test._semantic_perturber is None
        assert test._entailment_model is None


# =============================================================================
# Self-Consistency Test Unit Tests
# =============================================================================


class TestSelfConsistencyTest:
    """Tests for self-consistency."""

    def test_test_name(self):
        """Test that test has correct name."""
        test = SelfConsistencyTest()
        assert test.name == "self_consistency"

    def test_initialization(self):
        """Test initialization with num_samples."""
        test = SelfConsistencyTest(num_samples=10)
        assert test.num_samples == 10

    def test_lazy_loading(self):
        """Test that entailment model is not loaded on init."""
        test = SelfConsistencyTest()
        assert test._entailment_model is None


# =============================================================================
# Instruction Following Test Unit Tests
# =============================================================================


class TestInstructionFollowingTest:
    """Tests for instruction following fidelity."""

    def test_test_name(self):
        """Test that test has correct name."""
        test = InstructionFollowingTest()
        assert test.name == "instruction_following"

    def test_default_constraints(self):
        """Test that default constraints are used."""
        test = InstructionFollowingTest(num_samples=5)
        assert len(test.constraints) == 5

    def test_bullet_point_validator(self):
        """Test bullet point constraint validation."""
        constraint = DEFAULT_CONSTRAINTS[0]  # bullet_points_3

        valid = "- Point 1\n- Point 2\n- Point 3"
        invalid = "- Point 1\n- Point 2"

        assert constraint["validator"](valid) is True
        assert constraint["validator"](invalid) is False

    def test_lowercase_validator(self):
        """Test lowercase constraint validation."""
        constraint = DEFAULT_CONSTRAINTS[1]  # lowercase_only

        valid = "this is all lowercase"
        invalid = "This Has Uppercase"

        assert constraint["validator"](valid) is True
        assert constraint["validator"](invalid) is False

    def test_word_limit_validator(self):
        """Test word limit constraint validation."""
        constraint = DEFAULT_CONSTRAINTS[2]  # word_limit_50

        valid = " ".join(["word"] * 50)
        invalid = " ".join(["word"] * 51)

        assert constraint["validator"](valid) is True
        assert constraint["validator"](invalid) is False

    def test_numbered_list_validator(self):
        """Test numbered list constraint validation."""
        constraint = DEFAULT_CONSTRAINTS[3]  # numbered_list

        valid = "1. First item\n2. Second item"
        invalid = "First item\nSecond item"

        assert constraint["validator"](valid) is True
        assert constraint["validator"](invalid) is False

    def test_no_questions_validator(self):
        """Test no questions constraint validation."""
        constraint = DEFAULT_CONSTRAINTS[4]  # no_questions

        valid = "This is a statement."
        invalid = "Is this a question?"

        assert constraint["validator"](valid) is True
        assert constraint["validator"](invalid) is False

    def test_custom_constraints(self):
        """Test initialization with custom constraints."""
        custom = ["Respond in exactly 5 bullet points."]
        test = InstructionFollowingTest(instruction_constraints=custom)
        assert len(test.constraints) == 1
        assert test.constraints[0]["instruction"] == custom[0]


# =============================================================================
# Evaluator Tests
# =============================================================================


class TestConsistencyReliabilityEvaluator:
    """Tests for the evaluator."""

    def test_available_tests(self):
        """Test that all test types are available."""
        available = ConsistencyReliabilityEvaluator.available_tests()
        assert "sycophancy" in available
        assert "stability" in available
        assert "self_consistency" in available
        assert "instruction_following" in available

    def test_test_registry(self):
        """Test that all tests are in registry."""
        assert "sycophancy" in TEST_REGISTRY
        assert "stability" in TEST_REGISTRY
        assert "self_consistency" in TEST_REGISTRY
        assert "instruction_following" in TEST_REGISTRY

    def test_pass_thresholds(self):
        """Test that pass thresholds are defined."""
        assert "sycophancy" in PASS_THRESHOLDS
        assert "stability" in PASS_THRESHOLDS
        assert "self_consistency" in PASS_THRESHOLDS
        assert "instruction_following" in PASS_THRESHOLDS

    def test_evaluator_initialization_all_tests(self):
        """Test evaluator initializes with all tests by default."""
        evaluator = ConsistencyReliabilityEvaluator()
        assert len(evaluator.tests) == 4

    def test_evaluator_initialization_selective(self):
        """Test evaluator can be initialized with specific tests."""
        evaluator = ConsistencyReliabilityEvaluator(
            test_types=["sycophancy", "stability"]
        )
        assert len(evaluator.tests) == 2

    def test_evaluator_ignores_unknown_tests(self):
        """Test that unknown test types are ignored."""
        evaluator = ConsistencyReliabilityEvaluator(
            test_types=["sycophancy", "unknown_test"]
        )
        assert len(evaluator.tests) == 1


# =============================================================================
# Endpoint Validation Tests
# =============================================================================


class TestConsistencyReliabilityEndpointValidation:
    """Test request validation for consistency/reliability endpoint."""

    def test_endpoint_exists(self, client):
        """Test that the endpoint exists."""
        response = client.post(
            "/consistency-reliability",
            json={
                "model": {
                    "name": "test",
                    "description": "test",
                },
                "prompt": "test",
            },
        )
        assert response.status_code != 404

    def test_missing_model(self, client):
        """Test that missing model returns 422."""
        response = client.post(
            "/consistency-reliability",
            json={"prompt": "Hello world"},
        )
        assert response.status_code == 422

    def test_missing_prompt(self, client):
        """Test that missing prompt returns 422."""
        response = client.post(
            "/consistency-reliability",
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
            "/consistency-reliability",
            json={
                "model": {
                    "name": "invalid-model",
                    "description": "Test model",
                },
                "prompt": "Hello world",
            },
        )
        assert response.status_code == 400

    def test_invalid_num_samples_too_high(self, client):
        """Test that num_samples > 20 returns 422."""
        response = client.post(
            "/consistency-reliability",
            json={
                "model": {
                    "name": "openai-test",
                    "description": "Test model",
                },
                "prompt": "Hello world",
                "num_samples": 100,
            },
        )
        assert response.status_code == 422

    def test_invalid_num_samples_zero(self, client):
        """Test that num_samples = 0 returns 422."""
        response = client.post(
            "/consistency-reliability",
            json={
                "model": {
                    "name": "openai-test",
                    "description": "Test model",
                },
                "prompt": "Hello world",
                "num_samples": 0,
            },
        )
        assert response.status_code == 422

    def test_test_type_filtering_accepted(self, client):
        """Test that test type filtering is accepted in request."""
        response = client.post(
            "/consistency-reliability",
            json={
                "model": {
                    "name": "invalid-model",  # Will fail at model creation
                    "description": "Test model",
                },
                "prompt": "Test prompt",
                "test_types": ["sycophancy", "stability"],
                "num_samples": 2,
            },
        )
        # Should fail at model creation, not request validation
        assert response.status_code == 400

    def test_custom_sycophancy_topics_accepted(self, client):
        """Test that custom sycophancy topics are accepted."""
        response = client.post(
            "/consistency-reliability",
            json={
                "model": {
                    "name": "invalid-model",
                    "description": "Test model",
                },
                "prompt": "Test prompt",
                "sycophancy_topics": ["Topic 1", "Topic 2"],
            },
        )
        assert response.status_code == 400  # Fails at model creation

    def test_custom_instruction_constraints_accepted(self, client):
        """Test that custom instruction constraints are accepted."""
        response = client.post(
            "/consistency-reliability",
            json={
                "model": {
                    "name": "invalid-model",
                    "description": "Test model",
                },
                "prompt": "Test prompt",
                "instruction_constraints": ["Respond in 3 words."],
            },
        )
        assert response.status_code == 400  # Fails at model creation


# =============================================================================
# Integration Tests (require API keys)
# =============================================================================


@pytest.mark.integration
@pytest.mark.skip(reason="Requires OpenAI API key and network access")
class TestConsistencyReliabilityIntegration:
    """Integration tests (require API keys)."""

    def test_sycophancy_only(self, client):
        """Test sycophancy detection with OpenAI."""
        response = client.post(
            "/consistency-reliability",
            json={
                "model": {
                    "name": "openai-gpt-3.5-turbo",
                    "description": "OpenAI GPT-3.5 Turbo",
                },
                "prompt": "What is the best programming language?",
                "num_samples": 2,
                "test_types": ["sycophancy"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "sycophancy" in data["results"]
        assert data["results"]["sycophancy"]["score"] >= 0

    def test_instruction_following_only(self, client):
        """Test instruction following with OpenAI."""
        response = client.post(
            "/consistency-reliability",
            json={
                "model": {
                    "name": "openai-gpt-3.5-turbo",
                    "description": "OpenAI GPT-3.5 Turbo",
                },
                "prompt": "Explain what Python is.",
                "num_samples": 2,
                "test_types": ["instruction_following"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "instruction_following" in data["results"]

    def test_full_evaluation(self, client):
        """Test full consistency evaluation with all tests."""
        response = client.post(
            "/consistency-reliability",
            json={
                "model": {
                    "name": "openai-gpt-3.5-turbo",
                    "description": "OpenAI GPT-3.5 Turbo",
                },
                "prompt": "What is the capital of France?",
                "num_samples": 2,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "model" in data
        assert "prompt" in data
        assert "results" in data
        assert "summary" in data
        assert "reliability_grade" in data["summary"]
        assert data["summary"]["tests_run"] == 4
