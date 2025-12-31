"""Basic smoke tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root endpoint returns service online status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"service": "online"}


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_metrics_endpoint(client):
    """Test that Prometheus metrics endpoint is exposed."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
