"""Tests for toxicity detection batch endpoint."""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def valid_api_key():
    return "test-api-key"


class TestToxicityBatchEndpointValidation:
    """Tests for request validation on the toxicity batch endpoint."""

    def test_endpoint_exists(self, client, valid_api_key):
        """Test that the endpoint exists and returns appropriate error for invalid input."""
        response = client.post(
            "/toxicity-detection-batch",
            json={},
            headers={"X-API-Key": valid_api_key},
        )
        # Should return 422 for missing required fields, not 404
        assert response.status_code == 422

    def test_missing_model(self, client, valid_api_key):
        """Test that model is required."""
        response = client.post(
            "/toxicity-detection-batch",
            json={"user_prompts": ["test prompt"]},
            headers={"X-API-Key": valid_api_key},
        )
        assert response.status_code == 422
        assert "model" in response.text.lower()

    def test_missing_user_prompts(self, client, valid_api_key):
        """Test that user_prompts is required."""
        response = client.post(
            "/toxicity-detection-batch",
            json={
                "model": {
                    "name": "openai:gpt-4",
                    "description": "Test model",
                }
            },
            headers={"X-API-Key": valid_api_key},
        )
        assert response.status_code == 422
        assert "user_prompts" in response.text.lower()

    def test_empty_user_prompts_rejected(self, client, valid_api_key):
        """Test that empty user_prompts list is rejected."""
        response = client.post(
            "/toxicity-detection-batch",
            json={
                "model": {
                    "name": "openai:gpt-4",
                    "description": "Test model",
                },
                "user_prompts": [],
            },
            headers={"X-API-Key": valid_api_key},
        )
        assert response.status_code == 422

    def test_invalid_model_name(self, client, valid_api_key):
        """Test that invalid model name returns 400 error."""
        response = client.post(
            "/toxicity-detection-batch",
            json={
                "model": {
                    "name": "invalid-model",
                    "description": "Test model",
                },
                "user_prompts": ["test prompt"],
            },
            headers={"X-API-Key": valid_api_key},
        )
        assert response.status_code == 400
        data = response.json()
        assert "invalid" in data["detail"].lower()


class TestToxicityBatchIntegration:
    """Integration tests requiring external model servers."""

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires OpenAI API key")
    def test_openai_model(self, client, valid_api_key):
        """Test toxicity detection with OpenAI model."""
        response = client.post(
            "/toxicity-detection-batch",
            json={
                "model": {
                    "name": "openai:gpt-4",
                    "description": "Test model",
                },
                "user_prompts": ["What is the weather like today?"],
            },
            headers={"X-API-Key": valid_api_key},
        )
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "toxicity_evaluation" in data["result"]

    @pytest.mark.integration
    @pytest.mark.skip(reason="Requires external HuggingFace model server")
    def test_huggingface_model(self, client, valid_api_key):
        """Test toxicity detection with HuggingFace model."""
        response = client.post(
            "/toxicity-detection-batch",
            json={
                "model": {
                    "name": "huggingface:llama",
                    "description": "Test model",
                    "base_url": "http://localhost:8995/v1",
                },
                "user_prompts": ["What is the weather like today?"],
            },
            headers={"X-API-Key": valid_api_key},
        )
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert "toxicity_evaluation" in data["result"]
