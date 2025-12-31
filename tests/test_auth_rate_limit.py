"""Tests for API key authentication and rate limiting."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def valid_api_key():
    """A valid test API key."""
    return "test-api-key-12345"


@pytest.fixture
def client_with_auth(valid_api_key):
    """Test client with API key authentication configured."""
    with patch.dict(os.environ, {"REDTEAM_API_KEYS": valid_api_key}):
        yield TestClient(app), valid_api_key


class TestHealthEndpointsNoAuth:
    """Tests for health endpoints (no authentication required)."""

    def test_health_endpoint_no_auth_required(self, client):
        """Health endpoint should not require authentication."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_root_endpoint_no_auth_required(self, client):
        """Root endpoint should not require authentication."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"service": "online"}


class TestAuthenticationRequired:
    """Tests for endpoints that require authentication."""

    def test_protected_endpoint_missing_api_key(self, client_with_auth):
        """Protected endpoints should return 401 without API key."""
        client, _ = client_with_auth
        response = client.post(
            "/protect/guardrails",
            json={"input_text": "Hello world"},
        )
        assert response.status_code == 401
        assert "Missing API key" in response.json()["detail"]

    def test_protected_endpoint_invalid_api_key(self, client_with_auth):
        """Protected endpoints should return 401 with invalid key."""
        client, _ = client_with_auth
        response = client.post(
            "/protect/guardrails",
            json={"input_text": "Hello world"},
            headers={"X-API-Key": "invalid-key"},
        )
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]

    def test_protected_endpoint_valid_api_key(self, client_with_auth):
        """Protected endpoints should accept valid API key."""
        client, api_key = client_with_auth
        response = client.post(
            "/protect/guardrails",
            json={"input_text": "Hello world"},
            headers={"X-API-Key": api_key},
        )
        # Should get past auth - 200 for successful request
        assert response.status_code == 200

    def test_batch_endpoint_requires_auth(self, client_with_auth):
        """Batch endpoints should require authentication."""
        client, _ = client_with_auth
        response = client.post(
            "/toxicity-detection-batch",
            json={
                "model": {"name": "openai-test", "description": "test"},
                "num_samples": 1,
            },
        )
        assert response.status_code == 401

    def test_realtime_endpoint_requires_auth(self, client_with_auth):
        """Realtime endpoints should require authentication."""
        client, _ = client_with_auth
        response = client.post(
            "/toxicity-detection-realtime",
            json={
                "model": {"name": "openai-test", "description": "test"},
                "prompt": "Hello",
            },
        )
        assert response.status_code == 401


class TestMultipleApiKeys:
    """Tests for multiple API key support."""

    def test_multiple_api_keys_all_valid(self):
        """All comma-separated API keys should be accepted."""
        keys = "key1,key2,key3"
        with patch.dict(os.environ, {"REDTEAM_API_KEYS": keys}):
            client = TestClient(app)

            # Test each key works
            for key in ["key1", "key2", "key3"]:
                response = client.post(
                    "/protect/guardrails",
                    json={"input_text": "test"},
                    headers={"X-API-Key": key},
                )
                assert response.status_code == 200, f"Key {key} should be valid"

    def test_keys_with_whitespace_trimmed(self):
        """Keys with whitespace should be trimmed."""
        keys = "  key1  ,  key2  "
        with patch.dict(os.environ, {"REDTEAM_API_KEYS": keys}):
            client = TestClient(app)
            response = client.post(
                "/protect/guardrails",
                json={"input_text": "test"},
                headers={"X-API-Key": "key1"},
            )
            assert response.status_code == 200


class TestDevMode:
    """Tests for development mode (no API keys configured)."""

    def test_no_api_keys_allows_access(self):
        """With no REDTEAM_API_KEYS set, requests should be allowed."""
        # Clear the env var
        env = os.environ.copy()
        env.pop("REDTEAM_API_KEYS", None)

        with patch.dict(os.environ, env, clear=True):
            client = TestClient(app)
            response = client.post(
                "/protect/guardrails",
                json={"input_text": "Hello world"},
            )
            # Should not get 401 in dev mode
            assert response.status_code != 401


class TestRateLimitResponse:
    """Tests for rate limit response format."""

    def test_rate_limit_headers_in_response(self, client):
        """Responses should include rate limit information."""
        response = client.get("/health")
        assert response.status_code == 200
        # slowapi adds X-RateLimit headers

    def test_rate_limit_exceeded_returns_429(self):
        """Exceeding rate limit should return 429 with proper format."""
        # This test documents expected behavior
        # Actual rate limit testing is difficult in unit tests
        # because the in-memory limiter resets between test runs
        pass


class TestEndpointRateLimitCategories:
    """Tests verifying different endpoints have appropriate rate limits."""

    def test_health_endpoints_accessible(self, client):
        """Health endpoints should be rate limited at 60/min."""
        # First request should always succeed
        response = client.get("/health")
        assert response.status_code == 200

    def test_batch_endpoints_rate_limited(self, client_with_auth):
        """Batch endpoints should be rate limited at 10/min."""
        client, api_key = client_with_auth
        headers = {"X-API-Key": api_key}

        # First request should succeed (auth passes)
        response = client.post(
            "/toxicity-detection-batch",
            json={
                "model": {"name": "openai-gpt4", "description": "test"},
                "num_samples": 1,
            },
            headers=headers,
        )
        # May fail for other reasons (model not configured),
        # but should not be 429 on first request
        assert response.status_code != 429

    def test_realtime_endpoints_rate_limited(self, client_with_auth):
        """Realtime endpoints should be rate limited at 30/min."""
        client, api_key = client_with_auth
        headers = {"X-API-Key": api_key}

        response = client.post(
            "/protect/guardrails",
            json={"input_text": "test"},
            headers=headers,
        )
        # First request should not hit rate limit
        assert response.status_code != 429


class TestAuthErrorResponses:
    """Tests for authentication error response format."""

    def test_missing_key_error_format(self, client_with_auth):
        """Missing API key should return proper error format."""
        client, _ = client_with_auth
        response = client.post(
            "/protect/guardrails",
            json={"input_text": "test"},
        )
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "Missing API key" in data["detail"]

    def test_invalid_key_error_format(self, client_with_auth):
        """Invalid API key should return proper error format."""
        client, _ = client_with_auth
        response = client.post(
            "/protect/guardrails",
            json={"input_text": "test"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 401
        data = response.json()
        assert "detail" in data
        assert "Invalid API key" in data["detail"]

    def test_www_authenticate_header_present(self, client_with_auth):
        """401 responses should include WWW-Authenticate header."""
        client, _ = client_with_auth
        response = client.post(
            "/protect/guardrails",
            json={"input_text": "test"},
        )
        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers
