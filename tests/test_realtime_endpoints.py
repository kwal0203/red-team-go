"""Tests for realtime detection endpoints."""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestRealtimeEndpointValidation:
    """Test request validation for realtime endpoints."""

    def test_toxicity_realtime_missing_prompt(self, client):
        """Test that missing prompt returns 422."""
        response = client.post(
            "/toxicity-detection-realtime",
            json={
                "model": {
                    "name": "openai-test",
                    "description": "Test model",
                }
            },
        )
        assert response.status_code == 422

    def test_toxicity_realtime_missing_model(self, client):
        """Test that missing model returns 422."""
        response = client.post(
            "/toxicity-detection-realtime",
            json={"prompt": "Hello world"},
        )
        assert response.status_code == 422

    def test_bias_realtime_missing_prompt(self, client):
        """Test that missing prompt returns 422."""
        response = client.post(
            "/bias-detection-realtime",
            json={
                "model": {
                    "name": "openai-test",
                    "description": "Test model",
                }
            },
        )
        assert response.status_code == 422

    def test_bias_realtime_missing_model(self, client):
        """Test that missing model returns 422."""
        response = client.post(
            "/bias-detection-realtime",
            json={"prompt": "Hello world"},
        )
        assert response.status_code == 422

    def test_toxicity_realtime_invalid_model_name(self, client):
        """Test that invalid model name returns 400."""
        response = client.post(
            "/toxicity-detection-realtime",
            json={
                "model": {
                    "name": "invalid-model",  # Must contain 'openai' or 'huggingface'
                    "description": "Test model",
                },
                "prompt": "Hello world",
            },
        )
        assert response.status_code == 400
        assert (
            "openai" in response.json()["detail"].lower()
            or "huggingface" in response.json()["detail"].lower()
        )

    def test_bias_realtime_invalid_model_name(self, client):
        """Test that invalid model name returns 400."""
        response = client.post(
            "/bias-detection-realtime",
            json={
                "model": {
                    "name": "invalid-model",  # Must contain 'openai' or 'huggingface'
                    "description": "Test model",
                },
                "prompt": "Hello world",
            },
        )
        assert response.status_code == 400
        assert (
            "openai" in response.json()["detail"].lower()
            or "huggingface" in response.json()["detail"].lower()
        )


class TestRealtimeEndpointStructure:
    """Test that realtime endpoints exist and accept correct request structure."""

    def test_toxicity_realtime_endpoint_exists(self, client):
        """Test that the toxicity realtime endpoint exists."""
        # This will fail with 400/422/500 but not 404
        response = client.post(
            "/toxicity-detection-realtime",
            json={
                "model": {
                    "name": "test",
                    "description": "test",
                },
                "prompt": "test",
            },
        )
        assert response.status_code != 404

    def test_bias_realtime_endpoint_exists(self, client):
        """Test that the bias realtime endpoint exists."""
        # This will fail with 400/422/500 but not 404
        response = client.post(
            "/bias-detection-realtime",
            json={
                "model": {
                    "name": "test",
                    "description": "test",
                },
                "prompt": "test",
            },
        )
        assert response.status_code != 404


@pytest.mark.integration
@pytest.mark.skip(reason="Requires OpenAI API key and network access")
class TestRealtimeEndpointsIntegration:
    """Integration tests for realtime endpoints (require API keys)."""

    def test_toxicity_realtime_openai(self, client):
        """Test toxicity detection with OpenAI model."""
        response = client.post(
            "/toxicity-detection-realtime",
            json={
                "model": {
                    "name": "openai-gpt-3.5-turbo",
                    "description": "OpenAI GPT-3.5 Turbo",
                },
                "prompt": "What is the capital of France?",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "prompt" in data
        assert "model_response" in data
        assert "toxicity" in data
        assert "openai_score" in data["toxicity"]
        assert "paradetox_score" in data["toxicity"]
        assert "average_score" in data["toxicity"]
        assert "is_toxic" in data["toxicity"]

    def test_bias_realtime_openai(self, client):
        """Test bias detection with OpenAI model."""
        response = client.post(
            "/bias-detection-realtime",
            json={
                "model": {
                    "name": "openai-gpt-3.5-turbo",
                    "description": "OpenAI GPT-3.5 Turbo",
                },
                "prompt": "Tell me about different cultures.",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "prompt" in data
        assert "model_response" in data
        assert "bias" in data
        assert "bias_detected" in data["bias"]
        assert "explanation" in data["bias"]
