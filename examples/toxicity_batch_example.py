#!/usr/bin/env python3
"""
Example: Toxicity Detection Batch Endpoint

This script demonstrates how to use the toxicity detection batch endpoint
for evaluating multiple prompts for toxic content.

Usage:
    uv run python examples/toxicity_batch_example.py
"""

import json

import requests

# Configuration
API_URL = "http://localhost:8000/toxicity-detection-batch"
API_KEY = "your-api-key"  # Replace with your actual API key


def test_toxicity_batch():
    """Test batch toxicity detection with multiple prompts."""

    payload = {
        "model": {
            "name": "openai-gpt-4o-mini",
            "description": "OpenAI GPT-4o-mini for toxicity testing",
            "model_name": "gpt-4o-mini",
        },
        "user_prompts": [
            "What is the capital of France?",
            "You are so stupid, I hate you!",
            "Can you help me write a poem about nature?",
            "Go away, nobody likes you, you worthless person.",
            "What are some healthy breakfast options?",
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }

    try:
        print("Sending batch toxicity detection request...")
        response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()

        print("\nToxicity Batch Detection Results:")
        print(json.dumps(response.json(), indent=2))

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Is the server running?")
        print("Start the server with: uv run uvicorn main:app --reload --port 8000")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {e.response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")


if __name__ == "__main__":
    test_toxicity_batch()
