# Architecture Overview

RedTeamGO is organized as a FastAPI backend with service-specific modules, shared utilities, and optional React frontend/observability tooling.

## System Components
- **API layer**: `main.py` wires FastAPI routes, CORS, auth (`utils/auth.py`), rate limits (`utils/rate_limit.py`), and Prometheus instrumentation. Endpoints delegate to service modules and persist artifacts via `utils/artifact_storage.py`.
- **Service modules**: Each feature lives under `services/<feature>/service.py`, wrapping research implementations in `services/<feature>/src`. Services now share `utils.model_factory.create_target_model` to build provider-specific wrappers.
- **Model wrappers**: `services/model_wrappers/` defines `APIModel*` classes for OpenAI, OpenRouter, and HuggingFace. `utils/model_factory.py` chooses a wrapper using request fields or env defaults.
- **Datasets**: `services/datasets` registers loaders and exposes metadata/sample retrieval used by benchmark-style services.
- **Shared utilities**: Config/env helpers (`utils/config.py`), prompt constants, text generation helpers, and typed Pydantic models (`utils/models.py`) for request/response validation.
- **Frontend & observability**: `frontend/` React client (optional), Prometheus/Grafana via `docker-compose.yml`, with Redis for caching/future job orchestration.
- **Infra**: Dockerfile, docker-compose for local/demo, and Terraform under `infra/` for AWS ECS/Fargate deployments.

## Request Lifecycle
1. Client hits FastAPI route with `X-API-Key`.
2. Auth and rate-limit dependencies run; CORS and middleware applied.
3. Endpoint passes validated payloads to a service function.
4. Service builds a target model via `create_target_model`, invokes the feature evaluator, and returns structured results.
5. Artifacts are optionally persisted (local/S3) and Prometheus metrics are exposed for scraping.

## Extensibility Hooks
- **New evaluation capability**: Add `services/<feature>/service.py` plus supporting `src/` logic. Reuse `create_target_model` for provider selection, surface types in `utils/models.py`, and register a route in `main.py`.
- **New model providers**: Add a wrapper in `services/model_wrappers/` and extend `utils/model_factory.py` provider detection. Env defaults should be exposed via `utils/config.py`.
- **Datasets**: Implement a loader subclass in `services/datasets/src/`, register it, and it becomes available through the dataset registry endpoints.
- **Frontend/UX**: Extend the React app to call existing endpoints and display results; Prometheus metrics can be surfaced in Grafana.

## Production Hardening Recommendations
- **Configuration**: Move env lookups into a single Pydantic `Settings` object; validate required vars at startup; remove permissive default `allow_origins=["*"]` in production.
- **Service contract**: Standardize a small service interface (e.g., `ServiceContext`, `execute()` returning a typed result with status/latency/metadata). This reduces ad-hoc dict returns and simplifies logging.
- **Error handling**: Centralize error mapping (validation vs. provider errors vs. evaluator errors) and emit consistent error codes/messages.
- **Observability**: Add request IDs and structured logging; trace spans around model calls and dataset access (e.g., OpenTelemetry), and attach metrics per service.
- **Performance**: Introduce async or background tasks for long-running jobs (Celery/RQ + Redis already present), and cache static datasets/models.
- **Security**: Enforce API keys in all non-health endpoints, redact secrets in logs (artifact storage already scrubs), and harden CORS and rate limits per deployment tier.
- **Testing**: Add contract tests per service (happy-path and failure cases), provider fakes/mocks for model wrappers, and integration tests for dataset registry.
- **Release/CI**: Add formatting/linting (ruff/black), type checks (mypy), and automated CI workflows; publish a versioned Docker image for reproducible deploys.
