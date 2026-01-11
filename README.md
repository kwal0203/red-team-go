# RedTeamGO

## Overview
RedTeamGO is an LLM red teaming and safety evaluation platform. It provides a FastAPI service and optional React UI for running automated tests across safety, robustness, factuality, and privacy.

## Features
- **Safety Evaluations:** Toxicity, bias, guardrails, refusal consistency, privacy red teaming
- **Robustness Testing:** Character/word/semantic perturbations and adversarial prompt generation
- **Reliability & Factuality:** Consistency, instruction following, misinformation checks
- **Prompt Generation:** LLM, genetic, and PAIR methods with optional evaluation
- **Datasets:** Benchmark datasets (StereoSet, CrowS-Pairs, BBQ)
- **Monitoring:** Prometheus metrics and optional Grafana dashboard
- **Artifact Storage:** Local or S3 storage with provenance metadata

## Installation (uv)

1. Clone the repository:
   ```bash
   git clone https://github.com/kwal0203/red-team-go.git
   cd red-team-go
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

Optional (pip):
```bash
pip install -e .
```

## Usage

### Development Mode
```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production/Demo Mode (Docker Compose)
```bash
docker-compose up
```

This starts:
- Backend API (port 8000)
- Frontend UI (port 3000)
- Redis cache
- Prometheus metrics (port 9090)
- Grafana dashboard (port 3002)

### AWS Deployment (Terraform)
Infrastructure-as-code for ECS/Fargate is documented in `docs/aws-deployment.md`.

## Configuration
Common environment variables:
- `API_KEY_OPENAI`: OpenAI API key (for OpenAI-based evals)
- `OPENAI_MODEL`: Default OpenAI model id (e.g., gpt-3.5-turbo)
- `OPENROUTER_API_KEY`: OpenRouter API key (for OpenRouter-backed models)
- `OPENROUTER_BASE_URL`: Optional override for OpenRouter base URL (default https://openrouter.ai/api/v1)
- `OPENROUTER_MODEL`: Default OpenRouter model id (e.g., deepseek/deepseek-chat)
- `DEFAULT_MODEL_PROVIDER`: Set to `openai`, `openrouter`, or `huggingface` to use when requests omit a provider
- `HF_TOKEN`: HuggingFace token (for datasets/models)
- `REDTEAM_API_KEYS`: comma-separated API keys for auth
- `CORS_ORIGINS`: allowed CORS origins
- `LOG_LEVEL`: log level (default: INFO)

Artifact storage (local/S3): see `docs/artifact-storage.md`.

## Authentication and Rate Limits
- All evaluation endpoints require `X-API-Key`.
- Default rate limits: batch 10/min, realtime 30/min, health 60/min.

## API Endpoints
Full OpenAPI spec: `http://localhost:8000/docs`

Core endpoints:
- `POST /toxicity-detection-batch`
- `POST /bias-detection-batch`
- `POST /toxicity-detection-realtime`
- `POST /bias-detection-realtime`
- `POST /evaluate/guardrails`
- `POST /protect/guardrails`
- `POST /adversarial-robustness`
- `POST /stereotype-benchmark`
- `POST /generate-adversarial-prompts`
- `POST /consistency-reliability`
- `POST /misinformation-factuality`
- `POST /refusal-consistency`
- `POST /privacy-redteam`
- `POST /hallucination-confidence`
- `POST /gptfuzzer`
- `POST /jailbreakhub-analytics`
- `GET /datasets`
- `GET /datasets/{name}`
- `GET /health`

## Directory Structure
```
red-team-go
├── services/                  # Evaluation services
├── frontend/                  # React frontend application
├── examples/                  # API usage examples
├── utils/                     # Shared utilities
├── tests/                     # Test suite
├── main.py                    # FastAPI application entry point
└── docker-compose.yml         # Local demo stack
```

## References
- Bias Detection: Raza, S., et al. "Dbias: detecting biases and ensuring fairness in news articles." (2024)
- Toxicity Detection: Perez, E., et al. "Red Teaming Language Models with Language Models." (2022)

## License
This project is licensed under the MIT License. See `LICENSE` for details.
