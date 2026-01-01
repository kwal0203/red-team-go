"""Tests for model confidence hallucination detection service."""

import pytest
from fastapi.testclient import TestClient

from services.hallucination_detection_model_confidence.src.confidence_calculator import (
    ConfidenceCalculator,
    ConfidenceMethod,
    ConfidenceResult,
)

# =============================================================================
# ConfidenceCalculator Unit Tests
# =============================================================================


class TestConfidenceCalculator:
    """Tests for ConfidenceCalculator class."""

    @pytest.fixture
    def calculator(self):
        """Create a calculator instance."""
        return ConfidenceCalculator()

    @pytest.fixture
    def sample_logprobs(self):
        """Sample log probabilities for testing."""
        # Typical logprobs are negative (log of probability < 1)
        return [-0.5, -1.0, -0.3, -2.0, -0.7]

    @pytest.fixture
    def high_confidence_logprobs(self):
        """Log probabilities representing high confidence."""
        return [-0.1, -0.05, -0.1, -0.08, -0.12]

    @pytest.fixture
    def low_confidence_logprobs(self):
        """Log probabilities representing low confidence."""
        return [-5.0, -4.5, -6.0, -5.5, -4.0]

    def test_calculator_initialization(self, calculator):
        """Test calculator initializes with default method."""
        assert calculator.default_method == ConfidenceMethod.GEOMETRIC

    def test_calculator_custom_default_method(self):
        """Test calculator with custom default method."""
        calc = ConfidenceCalculator(default_method=ConfidenceMethod.MINIMUM)
        assert calc.default_method == ConfidenceMethod.MINIMUM

    def test_geometric_confidence(self, calculator, sample_logprobs):
        """Test geometric (sequence probability) method."""
        result = calculator.calculate(sample_logprobs, ConfidenceMethod.GEOMETRIC)

        assert isinstance(result, ConfidenceResult)
        assert result.method == ConfidenceMethod.GEOMETRIC
        assert 0 <= result.score <= 100
        assert result.num_tokens == 5
        assert "avg_logprob" in result.details
        assert "sequence_probability" in result.details
        assert "perplexity" in result.details

    def test_average_confidence(self, calculator, sample_logprobs):
        """Test average probability method."""
        result = calculator.calculate(sample_logprobs, ConfidenceMethod.AVERAGE)

        assert result.method == ConfidenceMethod.AVERAGE
        assert 0 <= result.score <= 100
        assert "min_prob" in result.details
        assert "max_prob" in result.details
        assert "avg_prob" in result.details

    def test_minimum_confidence(self, calculator, sample_logprobs):
        """Test minimum probability (pessimistic) method."""
        result = calculator.calculate(sample_logprobs, ConfidenceMethod.MINIMUM)

        assert result.method == ConfidenceMethod.MINIMUM
        assert 0 <= result.score <= 100
        assert "min_logprob" in result.details
        assert "min_token_indices" in result.details

        # Minimum should match the most negative logprob
        expected_min = min(sample_logprobs)
        assert result.details["min_logprob"] == expected_min

    def test_entropy_confidence(self, calculator, sample_logprobs):
        """Test entropy-based uncertainty method."""
        result = calculator.calculate(sample_logprobs, ConfidenceMethod.ENTROPY)

        assert result.method == ConfidenceMethod.ENTROPY
        assert 0 <= result.score <= 100
        assert "entropy" in result.details
        assert "normalized_entropy" in result.details

    def test_variance_confidence(self, calculator, sample_logprobs):
        """Test variance-based consistency method."""
        result = calculator.calculate(sample_logprobs, ConfidenceMethod.VARIANCE)

        assert result.method == ConfidenceMethod.VARIANCE
        assert 0 <= result.score <= 100
        assert "variance" in result.details
        assert "std_dev" in result.details
        assert "mean_logprob" in result.details

    def test_high_confidence_scores_higher(
        self, calculator, high_confidence_logprobs, low_confidence_logprobs
    ):
        """Test that high confidence logprobs produce higher scores."""
        high_result = calculator.calculate(
            high_confidence_logprobs, ConfidenceMethod.GEOMETRIC
        )
        low_result = calculator.calculate(
            low_confidence_logprobs, ConfidenceMethod.GEOMETRIC
        )

        assert high_result.score > low_result.score

    def test_calculate_all_methods(self, calculator, sample_logprobs):
        """Test calculating all methods at once."""
        results = calculator.calculate_all(sample_logprobs)

        assert len(results) == len(ConfidenceMethod)
        for method in ConfidenceMethod:
            assert method.value in results
            assert isinstance(results[method.value], ConfidenceResult)

    def test_empty_logprobs_raises_error(self, calculator):
        """Test that empty logprobs raises ValueError."""
        with pytest.raises(
            ValueError, match="Cannot calculate confidence from empty logprobs"
        ):
            calculator.calculate([])

    def test_interpret_score_high(self, calculator):
        """Test score interpretation for high confidence."""
        interpretation = ConfidenceCalculator.interpret_score(85)
        assert "High confidence" in interpretation
        assert "unlikely" in interpretation.lower()

    def test_interpret_score_moderate(self, calculator):
        """Test score interpretation for moderate confidence."""
        interpretation = ConfidenceCalculator.interpret_score(65)
        assert "Moderate" in interpretation

    def test_interpret_score_low(self, calculator):
        """Test score interpretation for low confidence."""
        interpretation = ConfidenceCalculator.interpret_score(45)
        assert "Low confidence" in interpretation

    def test_interpret_score_very_low(self, calculator):
        """Test score interpretation for very low confidence."""
        interpretation = ConfidenceCalculator.interpret_score(25)
        assert "Very low" in interpretation

    def test_interpret_score_extremely_low(self, calculator):
        """Test score interpretation for extremely low confidence."""
        interpretation = ConfidenceCalculator.interpret_score(5)
        assert "Extremely low" in interpretation

    def test_get_risk_level_low(self):
        """Test risk level for high confidence (low risk)."""
        assert ConfidenceCalculator.get_risk_level(75) == "low"

    def test_get_risk_level_medium(self):
        """Test risk level for moderate confidence."""
        assert ConfidenceCalculator.get_risk_level(55) == "medium"

    def test_get_risk_level_high(self):
        """Test risk level for low confidence (high risk)."""
        assert ConfidenceCalculator.get_risk_level(35) == "high"

    def test_get_risk_level_critical(self):
        """Test risk level for very low confidence (critical risk)."""
        assert ConfidenceCalculator.get_risk_level(20) == "critical"

    def test_confidence_result_to_dict(self, calculator, sample_logprobs):
        """Test ConfidenceResult serialization."""
        result = calculator.calculate(sample_logprobs)
        result_dict = result.to_dict()

        assert "score" in result_dict
        assert "method" in result_dict
        assert "raw_value" in result_dict
        assert "num_tokens" in result_dict
        assert "details" in result_dict

    def test_uses_default_method(self, calculator, sample_logprobs):
        """Test that calculate uses default method when not specified."""
        result = calculator.calculate(sample_logprobs)
        assert result.method == calculator.default_method


# =============================================================================
# Service Function Tests
# =============================================================================


class TestEvaluateTextConfidence:
    """Tests for evaluate_text_confidence function."""

    def test_valid_logprobs(self):
        """Test evaluation with valid pre-computed logprobs."""
        from services.hallucination_detection_model_confidence import (
            evaluate_text_confidence,
        )

        logprobs = [{"logprob": -0.5}, {"logprob": -1.0}, {"logprob": -0.3}]
        result = evaluate_text_confidence(
            text="Test text",
            logprobs=logprobs,
            method="geometric",
        )

        assert result.confidence_score >= 0
        assert result.confidence_score <= 100
        assert result.risk_level in ["low", "medium", "high", "critical"]
        assert result.method == "geometric"
        assert result.num_tokens == 3

    def test_empty_logprobs_raises_error(self):
        """Test that empty logprobs raises error."""
        from services.hallucination_detection_model_confidence import (
            evaluate_text_confidence,
        )

        with pytest.raises(ValueError, match="empty"):
            evaluate_text_confidence(text="Test", logprobs=[], method="geometric")

    def test_invalid_logprobs_format(self):
        """Test that malformed logprobs raises error."""
        from services.hallucination_detection_model_confidence import (
            evaluate_text_confidence,
        )

        with pytest.raises(ValueError, match="Invalid logprobs format"):
            evaluate_text_confidence(
                text="Test",
                logprobs=[{"wrong_key": -0.5}],
                method="geometric",
            )

    def test_invalid_method_raises_error(self):
        """Test that invalid method raises error."""
        from services.hallucination_detection_model_confidence import (
            evaluate_text_confidence,
        )

        with pytest.raises(ValueError, match="Invalid method"):
            evaluate_text_confidence(
                text="Test",
                logprobs=[{"logprob": -0.5}],
                method="invalid_method",
            )

    def test_include_all_methods(self):
        """Test including all methods in result."""
        from services.hallucination_detection_model_confidence import (
            evaluate_text_confidence,
        )

        logprobs = [{"logprob": -0.5}, {"logprob": -1.0}]
        result = evaluate_text_confidence(
            text="Test",
            logprobs=logprobs,
            method="geometric",
            include_all_methods=True,
        )

        assert result.all_methods is not None
        assert len(result.all_methods) == len(ConfidenceMethod)


# =============================================================================
# Legacy Service Interface Tests
# =============================================================================


class TestLegacyServiceInterface:
    """Tests for legacy service() function interface."""

    def test_legacy_logprobs_interface(self):
        """Test legacy interface with pre-computed logprobs."""
        from services.hallucination_detection_model_confidence import service

        class Args:
            logprobs = [{"logprob": -0.5}, {"logprob": -1.0}]
            method = "geometric"
            text = "Test text"
            include_all_methods = False

        result = service(Args())

        assert "confidence_score" in result
        assert "risk_level" in result
        assert "interpretation" in result

    def test_missing_args_raises_error(self):
        """Test that missing args raises error."""
        from services.hallucination_detection_model_confidence import service

        class Args:
            pass

        with pytest.raises(ValueError, match="Invalid arguments"):
            service(Args())


# =============================================================================
# API Endpoint Tests
# =============================================================================


class TestHallucinationConfidenceEndpoint:
    """Tests for /hallucination-confidence endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from main import app

        return TestClient(app)

    @pytest.fixture
    def api_key_header(self):
        """Return API key header for authenticated requests."""
        import os

        # Use test API key or mock
        return {"X-API-Key": os.environ.get("REDTEAM_API_KEYS", "test-key")}

    def test_endpoint_requires_auth(self, client):
        """Test that endpoint requires authentication when API keys configured."""
        import os

        # Skip if auth is disabled (no API keys configured)
        if not os.environ.get("REDTEAM_API_KEYS"):
            pytest.skip("Auth disabled - REDTEAM_API_KEYS not set")

        response = client.post(
            "/hallucination-confidence",
            json={
                "model": {
                    "name": "test",
                    "description": "test",
                    "model_name": "gpt-3.5-turbo",
                },
                "prompt": "Test prompt",
            },
        )
        assert response.status_code == 401

    def test_endpoint_validates_request(self, client, api_key_header):
        """Test that endpoint validates request body."""
        response = client.post(
            "/hallucination-confidence",
            headers=api_key_header,
            json={
                "model": {"name": "openai-gpt35", "description": "GPT-3.5"}
            },  # Missing prompt
        )
        assert response.status_code == 422

    def test_endpoint_validates_method(self, client, api_key_header):
        """Test that endpoint validates confidence method."""
        response = client.post(
            "/hallucination-confidence",
            headers=api_key_header,
            json={
                "model": {"name": "openai-gpt35", "description": "GPT-3.5"},
                "prompt": "Test prompt",
                "method": "invalid_method",
            },
        )
        # Should return 400 for invalid method
        assert response.status_code in [400, 422]

    @pytest.mark.skip(reason="Requires OpenAI API key")
    def test_endpoint_success(self, client, api_key_header):
        """Test successful endpoint call."""
        response = client.post(
            "/hallucination-confidence",
            headers=api_key_header,
            json={
                "model": {"name": "openai-gpt35", "description": "GPT-3.5"},
                "prompt": "What is 2+2?",
                "method": "geometric",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "confidence_score" in data
        assert "risk_level" in data


# =============================================================================
# Pydantic Model Tests
# =============================================================================


class TestPydanticModels:
    """Tests for Pydantic request/response models."""

    def test_model_confidence_request_valid(self):
        """Test valid ModelConfidenceRequest."""
        from utils.models import ModelConfidenceRequest

        request = ModelConfidenceRequest(
            model={
                "name": "openai-gpt35",
                "description": "GPT-3.5 Turbo",
                "model_name": "gpt-3.5-turbo",
            },
            prompt="Test prompt",
            method="geometric",
        )

        assert request.prompt == "Test prompt"
        assert request.method == "geometric"
        assert request.max_tokens == 256  # default

    def test_model_confidence_request_defaults(self):
        """Test ModelConfidenceRequest defaults."""
        from utils.models import ModelConfidenceRequest

        request = ModelConfidenceRequest(
            model={"name": "openai-gpt35", "description": "GPT-3.5 Turbo"},
            prompt="Test",
        )

        assert request.method == "geometric"
        assert request.max_tokens == 256
        assert request.include_all_methods is False

    def test_model_confidence_response_valid(self):
        """Test valid ModelConfidenceResponse."""
        from utils.models import ModelConfidenceResponse

        response = ModelConfidenceResponse(
            confidence_score=75.5,
            risk_level="low",
            interpretation="High confidence",
            method="geometric",
            generated_text="Response text",
            num_tokens=10,
            details={"avg_logprob": -0.5},
        )

        assert response.confidence_score == 75.5
        assert response.risk_level == "low"

    def test_model_confidence_response_all_methods(self):
        """Test ModelConfidenceResponse with all methods."""
        from utils.models import MethodResult, ModelConfidenceResponse

        response = ModelConfidenceResponse(
            confidence_score=75.5,
            risk_level="low",
            interpretation="High confidence",
            method="geometric",
            generated_text="Response text",
            num_tokens=10,
            details={},
            all_methods={
                "geometric": MethodResult(
                    score=75.5,
                    method="geometric",
                    raw_value=0.755,
                    num_tokens=10,
                    details={},
                )
            },
        )

        assert response.all_methods is not None
        assert "geometric" in response.all_methods
