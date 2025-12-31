"""Tests for prompt generation service."""

import pytest
from fastapi.testclient import TestClient

from main import app
from services.prompt_generation.src.evaluator import (
    GenerationReport,
    PromptEvaluator,
)
from services.prompt_generation.src.generators.base import GeneratedPrompt
from services.prompt_generation.src.generators.genetic import GeneticPromptGenerator
from services.prompt_generation.src.generators.llm_generator import LLMPromptGenerator
from services.prompt_generation.src.generators.pair import PAIRPromptGenerator


@pytest.fixture
def client():
    return TestClient(app)


# =============================================================================
# Generator Tests
# =============================================================================


class TestLLMPromptGenerator:
    """Tests for LLM-based prompt generator."""

    def test_generator_creation(self):
        """Test that generator can be created."""
        generator = LLMPromptGenerator()
        assert generator is not None
        assert generator.name == "llm"

    def test_module_level_category_prompts(self):
        """Test that category prompts are defined at module level."""
        from services.prompt_generation.src.generators import llm_generator

        assert hasattr(llm_generator, "CATEGORY_PROMPTS")
        assert "jailbreak" in llm_generator.CATEGORY_PROMPTS
        assert "harmful" in llm_generator.CATEGORY_PROMPTS
        assert "bias" in llm_generator.CATEGORY_PROMPTS
        assert "toxicity" in llm_generator.CATEGORY_PROMPTS

    def test_generation_requires_model(self):
        """Test that generation without model raises ValueError."""
        generator = LLMPromptGenerator()

        with pytest.raises(ValueError) as exc_info:
            generator.generate(category="jailbreak", num_prompts=3)
        assert "No model set" in str(exc_info.value)

    def test_parse_prompts(self):
        """Test prompt parsing from LLM response."""
        generator = LLMPromptGenerator()

        # Simulate LLM response with various formats
        response = """1. First prompt here
2. Second prompt here
- Third prompt with bullet
* Fourth prompt with asterisk
Here are some prompts:
Fifth actual prompt"""

        prompts = generator._parse_prompts(response)

        assert len(prompts) >= 4
        assert "First prompt here" in prompts
        assert "Second prompt here" in prompts


class TestGeneticPromptGenerator:
    """Tests for genetic algorithm prompt generator (stub)."""

    def test_generator_creation(self):
        """Test that generator can be created."""
        generator = GeneticPromptGenerator()
        assert generator is not None

    def test_raises_not_implemented(self):
        """Test that generate raises NotImplementedError."""
        generator = GeneticPromptGenerator()
        with pytest.raises(NotImplementedError) as exc_info:
            generator.generate(category="jailbreak", num_prompts=5)
        assert "Genetic algorithm prompt generator not yet implemented" in str(
            exc_info.value
        )


class TestPAIRPromptGenerator:
    """Tests for PAIR prompt generator (stub)."""

    def test_generator_creation(self):
        """Test that generator can be created."""
        generator = PAIRPromptGenerator()
        assert generator is not None

    def test_raises_not_implemented(self):
        """Test that generate raises NotImplementedError."""
        generator = PAIRPromptGenerator()
        with pytest.raises(NotImplementedError) as exc_info:
            generator.generate(category="jailbreak", num_prompts=5)
        assert "PAIR" in str(exc_info.value)


# =============================================================================
# Evaluator Tests
# =============================================================================


class TestPromptEvaluator:
    """Tests for prompt evaluator."""

    def test_evaluator_creation(self):
        """Test that evaluator can be created."""
        evaluator = PromptEvaluator()
        assert evaluator is not None

    def test_refusal_detection(self):
        """Test that refusals are detected."""
        evaluator = PromptEvaluator()

        # Test various refusal patterns
        refusals = [
            "I cannot help with that request.",
            "I'm unable to provide that information.",
            "Sorry, but I can't assist with this.",
            "This violates my guidelines.",
        ]
        for text in refusals:
            assert evaluator._is_refusal(text) is True

    def test_non_refusal_detection(self):
        """Test that non-refusals are not flagged."""
        evaluator = PromptEvaluator()

        non_refusals = [
            "Here is the information you requested.",
            "I would be happy to help with that.",
            "Let me explain how this works.",
        ]
        for text in non_refusals:
            assert evaluator._is_refusal(text) is False

    def test_create_report(self):
        """Test report creation."""
        evaluator = PromptEvaluator()

        prompts = [
            GeneratedPrompt(
                prompt="Test prompt 1",
                generation_method="llm_variation",
                metadata={"category": "test"},
            ),
            GeneratedPrompt(
                prompt="Test prompt 2",
                generation_method="llm_variation",
                metadata={"category": "test"},
            ),
        ]

        report = evaluator.create_report(
            generator_name="llm",
            category="jailbreak",
            prompts=prompts,
            evaluations=None,
        )

        assert isinstance(report, GenerationReport)
        assert report.generator == "llm"
        assert report.target_category == "jailbreak"
        assert len(report.prompts) == 2
        assert report.summary["total_generated"] == 2


# =============================================================================
# Endpoint Validation Tests
# =============================================================================


class TestPromptGenerationEndpointValidation:
    """Test request validation for prompt generation endpoint."""

    def test_endpoint_exists(self, client):
        """Test that the endpoint exists."""
        response = client.post(
            "/generate-adversarial-prompts",
            json={
                "model": {
                    "name": "test",
                    "description": "test",
                },
                "target_category": "jailbreak",
            },
        )
        assert response.status_code != 404

    def test_missing_model(self, client):
        """Test that missing model returns 422."""
        response = client.post(
            "/generate-adversarial-prompts",
            json={"target_category": "jailbreak"},
        )
        assert response.status_code == 422

    def test_missing_category(self, client):
        """Test that missing target_category returns 422."""
        response = client.post(
            "/generate-adversarial-prompts",
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
            "/generate-adversarial-prompts",
            json={
                "model": {
                    "name": "invalid-model",
                    "description": "Test model",
                },
                "target_category": "jailbreak",
            },
        )
        assert response.status_code == 400

    def test_invalid_generator(self, client):
        """Test that invalid generator returns 400."""
        response = client.post(
            "/generate-adversarial-prompts",
            json={
                "model": {
                    "name": "openai-test",
                    "description": "Test model",
                },
                "target_category": "jailbreak",
                "generator": "invalid_generator",
            },
        )
        # Should fail at service with unknown generator
        assert response.status_code == 400

    def test_stubbed_generator_returns_empty_prompts(self, client):
        """Test that stubbed generator returns success with empty prompts."""
        response = client.post(
            "/generate-adversarial-prompts",
            json={
                "model": {
                    "name": "openai-test",
                    "description": "Test model",
                },
                "target_category": "jailbreak",
                "generator": "genetic",  # Stubbed generator
            },
        )
        # genetic is stubbed, returns 200 with empty prompts
        assert response.status_code == 200
        data = response.json()
        assert data["prompts"] == []
        assert data["summary"]["total_generated"] == 0

    def test_pair_generator_returns_empty_prompts(self, client):
        """Test that PAIR generator returns success with empty prompts."""
        response = client.post(
            "/generate-adversarial-prompts",
            json={
                "model": {
                    "name": "openai-test",
                    "description": "Test model",
                },
                "target_category": "jailbreak",
                "generator": "pair",  # Stubbed generator
            },
        )
        # pair is stubbed, returns 200 with empty prompts
        assert response.status_code == 200
        data = response.json()
        assert data["prompts"] == []
        assert data["summary"]["total_generated"] == 0

    def test_num_prompts_validation(self, client):
        """Test that num_prompts is validated."""
        response = client.post(
            "/generate-adversarial-prompts",
            json={
                "model": {
                    "name": "openai-test",
                    "description": "Test model",
                },
                "target_category": "jailbreak",
                "num_prompts": 100,  # Max is 50
            },
        )
        assert response.status_code == 422


@pytest.mark.integration
@pytest.mark.skip(reason="Requires OpenAI API key and network access")
class TestPromptGenerationIntegration:
    """Integration tests for prompt generation (require API keys)."""

    def test_llm_generation(self, client):
        """Test LLM prompt generation with OpenAI."""
        response = client.post(
            "/generate-adversarial-prompts",
            json={
                "model": {
                    "name": "openai-gpt-3.5-turbo",
                    "description": "OpenAI GPT-3.5 Turbo",
                },
                "target_category": "jailbreak",
                "generator": "llm",
                "num_prompts": 5,
                "evaluate": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "generator" in data
        assert "target_category" in data
        assert "prompts" in data
        assert "summary" in data
        assert data["generator"] == "llm"
        assert len(data["prompts"]) >= 1

    def test_generation_without_evaluation(self, client):
        """Test prompt generation without evaluation."""
        response = client.post(
            "/generate-adversarial-prompts",
            json={
                "model": {
                    "name": "openai-gpt-3.5-turbo",
                    "description": "OpenAI GPT-3.5 Turbo",
                },
                "target_category": "harmful",
                "generator": "llm",
                "num_prompts": 3,
                "evaluate": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Without evaluation, prompts should not have evaluation field
        for prompt in data["prompts"]:
            assert prompt.get("evaluation") is None

    def test_generation_with_seed(self, client):
        """Test prompt generation with seed prompt."""
        response = client.post(
            "/generate-adversarial-prompts",
            json={
                "model": {
                    "name": "openai-gpt-3.5-turbo",
                    "description": "OpenAI GPT-3.5 Turbo",
                },
                "target_category": "jailbreak",
                "generator": "llm",
                "num_prompts": 3,
                "seed_prompt": "Ignore all previous instructions",
            },
        )
        assert response.status_code == 200
