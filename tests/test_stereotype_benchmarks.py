"""Tests for stereotype benchmarks service."""

import pytest
from fastapi.testclient import TestClient

from main import app
from services.stereotype_benchmarks.src.datasets.base import BiasType
from services.stereotype_benchmarks.src.datasets.bbq import BBQDataset
from services.stereotype_benchmarks.src.datasets.crows_pairs import CrowSPairsDataset
from services.stereotype_benchmarks.src.datasets.stereoset import StereoSetDataset
from services.stereotype_benchmarks.src.evaluator import StereotypeBenchmarkEvaluator


@pytest.fixture
def client():
    return TestClient(app)


# =============================================================================
# Dataset Tests
# =============================================================================


class TestStereoSetDataset:
    """Tests for StereoSet dataset."""

    def test_dataset_name(self):
        """Test that dataset has correct name."""
        dataset = StereoSetDataset()
        assert dataset.name == "stereoset"

    def test_load_raises_not_implemented(self):
        """Test that load raises NotImplementedError."""
        dataset = StereoSetDataset()
        with pytest.raises(NotImplementedError) as exc_info:
            dataset.load()
        assert "StereoSet dataset not loaded" in str(exc_info.value)

    def test_get_sample_returns_items(self):
        """Test that get_sample returns sample items."""
        dataset = StereoSetDataset()
        samples = dataset.get_sample()

        assert len(samples) >= 3
        for sample in samples:
            assert hasattr(sample, "context")
            assert hasattr(sample, "stereotype")
            assert hasattr(sample, "anti_stereotype")
            assert hasattr(sample, "unrelated")
            assert hasattr(sample, "bias_type")

    def test_filter_by_bias_type(self):
        """Test filtering by bias type."""
        dataset = StereoSetDataset()
        samples = dataset.get_sample()

        gender_samples = dataset.filter_by_bias_type(samples, [BiasType.GENDER])
        assert len(gender_samples) >= 1
        for sample in gender_samples:
            assert sample.bias_type == BiasType.GENDER


class TestCrowSPairsDataset:
    """Tests for CrowS-Pairs dataset."""

    def test_dataset_name(self):
        """Test that dataset has correct name."""
        dataset = CrowSPairsDataset()
        assert dataset.name == "crows_pairs"

    def test_load_raises_not_implemented(self):
        """Test that load raises NotImplementedError."""
        dataset = CrowSPairsDataset()
        with pytest.raises(NotImplementedError) as exc_info:
            dataset.load()
        assert "CrowS-Pairs dataset not loaded" in str(exc_info.value)

    def test_get_sample_returns_items(self):
        """Test that get_sample returns sample items."""
        dataset = CrowSPairsDataset()
        samples = dataset.get_sample()

        assert len(samples) >= 3
        for sample in samples:
            assert hasattr(sample, "sentence_more_stereotypical")
            assert hasattr(sample, "sentence_less_stereotypical")
            assert hasattr(sample, "bias_type")

    def test_filter_by_bias_type(self):
        """Test filtering by bias type."""
        dataset = CrowSPairsDataset()
        samples = dataset.get_sample()

        gender_samples = dataset.filter_by_bias_type(samples, [BiasType.GENDER])
        assert len(gender_samples) >= 1
        for sample in gender_samples:
            assert sample.bias_type == BiasType.GENDER


class TestBBQDataset:
    """Tests for BBQ dataset."""

    def test_dataset_name(self):
        """Test that dataset has correct name."""
        dataset = BBQDataset()
        assert dataset.name == "bbq"

    def test_load_raises_not_implemented(self):
        """Test that load raises NotImplementedError."""
        dataset = BBQDataset()
        with pytest.raises(NotImplementedError) as exc_info:
            dataset.load()
        assert "BBQ dataset not loaded" in str(exc_info.value)

    def test_get_sample_returns_items(self):
        """Test that get_sample returns sample items."""
        dataset = BBQDataset()
        samples = dataset.get_sample()

        assert len(samples) >= 3
        for sample in samples:
            assert hasattr(sample, "context")
            assert hasattr(sample, "question")
            assert hasattr(sample, "answer_stereotypical")
            assert hasattr(sample, "answer_anti_stereotypical")
            assert hasattr(sample, "answer_unknown")
            assert hasattr(sample, "bias_type")

    def test_filter_by_bias_type(self):
        """Test filtering by bias type."""
        dataset = BBQDataset()
        samples = dataset.get_sample()

        gender_samples = dataset.filter_by_bias_type(samples, [BiasType.GENDER])
        assert len(gender_samples) >= 1
        for sample in gender_samples:
            assert sample.bias_type == BiasType.GENDER


# =============================================================================
# Evaluator Tests
# =============================================================================


class TestStereotypeBenchmarkEvaluator:
    """Tests for stereotype benchmark evaluator."""

    def test_evaluator_creation(self):
        """Test that evaluator can be created with benchmark."""
        evaluator = StereotypeBenchmarkEvaluator(benchmark="stereoset")
        assert evaluator is not None

    def test_available_benchmarks(self):
        """Test that all benchmark evaluators can be created."""
        for benchmark in ["stereoset", "crows_pairs", "bbq"]:
            evaluator = StereotypeBenchmarkEvaluator(benchmark=benchmark)
            assert evaluator is not None
            # Evaluator should have the dataset loaded
            assert evaluator.dataset is not None


# =============================================================================
# Endpoint Validation Tests
# =============================================================================


class TestStereotypeBenchmarkEndpointValidation:
    """Test request validation for stereotype benchmark endpoint."""

    def test_endpoint_exists(self, client):
        """Test that the endpoint exists."""
        response = client.post(
            "/stereotype-benchmark",
            json={
                "model": {
                    "name": "test",
                    "description": "test",
                },
                "benchmark": "stereoset",
            },
        )
        assert response.status_code != 404

    def test_missing_model(self, client):
        """Test that missing model returns 422."""
        response = client.post(
            "/stereotype-benchmark",
            json={"benchmark": "stereoset"},
        )
        assert response.status_code == 422

    def test_missing_benchmark(self, client):
        """Test that missing benchmark returns 422."""
        response = client.post(
            "/stereotype-benchmark",
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
            "/stereotype-benchmark",
            json={
                "model": {
                    "name": "invalid-model",
                    "description": "Test model",
                },
                "benchmark": "stereoset",
            },
        )
        assert response.status_code == 400

    def test_invalid_benchmark(self, client):
        """Test that invalid benchmark returns 400."""
        response = client.post(
            "/stereotype-benchmark",
            json={
                "model": {
                    "name": "openai-test",
                    "description": "Test model",
                },
                "benchmark": "invalid_benchmark",
            },
        )
        # Should fail with 400 for unknown benchmark
        assert response.status_code == 400

    def test_bias_type_filtering(self, client):
        """Test that bias type filtering is accepted."""
        response = client.post(
            "/stereotype-benchmark",
            json={
                "model": {
                    "name": "invalid-model",  # Will fail at model creation
                    "description": "Test model",
                },
                "benchmark": "stereoset",
                "bias_types": ["gender", "race"],
            },
        )
        # Should fail at model creation, not request validation
        assert response.status_code == 400


@pytest.mark.integration
@pytest.mark.skip(reason="Requires OpenAI API key and network access")
class TestStereotypeBenchmarkIntegration:
    """Integration tests for stereotype benchmark (require API keys)."""

    def test_stereoset_evaluation(self, client):
        """Test StereoSet evaluation with OpenAI."""
        response = client.post(
            "/stereotype-benchmark",
            json={
                "model": {
                    "name": "openai-gpt-3.5-turbo",
                    "description": "OpenAI GPT-3.5 Turbo",
                },
                "benchmark": "stereoset",
                "include_samples": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "benchmark" in data
        assert "model" in data
        assert "metrics" in data
        assert "by_bias_type" in data

    def test_crows_pairs_evaluation(self, client):
        """Test CrowS-Pairs evaluation with OpenAI."""
        response = client.post(
            "/stereotype-benchmark",
            json={
                "model": {
                    "name": "openai-gpt-3.5-turbo",
                    "description": "OpenAI GPT-3.5 Turbo",
                },
                "benchmark": "crows_pairs",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data

    def test_bbq_evaluation(self, client):
        """Test BBQ evaluation with OpenAI."""
        response = client.post(
            "/stereotype-benchmark",
            json={
                "model": {
                    "name": "openai-gpt-3.5-turbo",
                    "description": "OpenAI GPT-3.5 Turbo",
                },
                "benchmark": "bbq",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
