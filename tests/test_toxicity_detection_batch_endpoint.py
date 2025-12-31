import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.integration
@pytest.mark.skip(reason="Requires external HuggingFace model server on localhost:8995")
def test_database_prompts_huggingface(client):
    model_data = {
        "name": "huggingface model",
        "description": "A model for answering user questions.",
        "base_url": "http://localhost:8995/v1",
    }

    detection_batch = {
        "model": model_data,
        "num_samples": 1,
        "random": True,
        "database_prompts": True,
        "user_prompts": None,
        "user_topics": None,
    }

    response = client.post("/toxicity-detection-batch", json=detection_batch)
    assert response.status_code == 200

    response_data = response.json()
    assert "result" in response_data
    assert "toxicity_evaluation" in response_data["result"]


# def test_user_prompts_prompts(client):
#     model_data = {
#         "name": "Example Model",
#         "description": "A model for answering user questions.",
#     }
#     detection_batch = {
#         "model": model_data,
#         "num_samples": 10,
#         "random": None,
#         "database_prompts": None,
#         "user_prompts": ["How is the weather today?"],
#         "user_topics": None,
#     }

#     response = client.post("/toxicity-detection-batch", json=detection_batch)
#     assert response.status_code == 200

#     response_data = response.json()
#     assert "result" in response_data
#     assert "toxicity_evaluation" in response_data["result"]


# def test_user_topics(client):
#     model_data = {
#         "name": "Example Model",
#         "description": "A model for answering user questions.",
#     }
#     detection_batch = {
#         "model": model_data,
#         "num_samples": 10,
#         "random": None,
#         "database_prompts": None,
#         "user_prompts": None,
#         "user_topics": ["ai", "ml"],
#     }

#     response = client.post("/toxicity-detection-batch", json=detection_batch)
#     assert response.status_code == 200

#     response_data = response.json()
#     assert "result" in response_data
#     assert "toxicity_evaluation" in response_data["result"]
