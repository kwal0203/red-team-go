# RedTeamGo API Examples

Example scripts demonstrating how to use the RedTeamGo API endpoints.

## Prerequisites

1. Start the backend server:
```bash
uv run uvicorn main:app --reload --port 8000
```

2. Set your API key in the example scripts (replace `your-api-key`)

## Available Examples

### Comprehensive Test Script

**`test_all_endpoints.py`** - Tests all API endpoints with sample payloads.

```bash
# Test all endpoints
uv run python examples/test_all_endpoints.py --api-key YOUR_API_KEY

# Test specific category
uv run python examples/test_all_endpoints.py --category detection

# Test multiple categories
uv run python examples/test_all_endpoints.py --category health --category guardrails

# Verbose mode (show response previews)
uv run python examples/test_all_endpoints.py --verbose

# List available categories
uv run python examples/test_all_endpoints.py --list-categories
```

**Categories:**
- `health` - Health check endpoints (no auth required)
- `detection` - Toxicity and bias detection (batch + realtime)
- `guardrails` - Input/output safety guardrails
- `adversarial` - Adversarial robustness testing
- `benchmarks` - Stereotype benchmarks (StereoSet, CrowS-Pairs, BBQ)
- `consistency` - Consistency and reliability tests
- `misinformation` - Misinformation and factuality tests
- `refusal` - Refusal consistency tests
- `privacy` - Privacy red team tests
- `hallucination` - Hallucination detection

### Individual Examples

1. **`toxicity_batch_example.py`** - Batch toxicity detection
   ```bash
   uv run python examples/toxicity_batch_example.py
   ```

2. **`bias_batch_example.py`** - Batch bias detection
   ```bash
   uv run python examples/bias_batch_example.py
   ```

3. **`realtime_examples.py`** - Realtime detection (toxicity, bias, guardrails, hallucination)
   ```bash
   uv run python examples/realtime_examples.py
   ```

## API Endpoints Reference

### Health (No Auth)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service status |
| `/health` | GET | Health check |

### Detection
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/toxicity-detection-batch` | POST | Batch toxicity analysis |
| `/toxicity-detection-realtime` | POST | Single prompt toxicity |
| `/bias-detection-batch` | POST | Batch bias analysis |
| `/bias-detection-realtime` | POST | Single prompt bias |

### Safety & Guardrails
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/evaluate/guardrails` | POST | Red-team guardrail testing |
| `/protect/guardrails` | POST | Production middleware (block/flag/redact) |

### Adversarial Testing
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/adversarial-robustness` | POST | Test with perturbations |
| `/generate-adversarial-prompts` | POST | Generate adversarial prompts |

### Benchmarks & Evaluation
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stereotype-benchmark` | POST | StereoSet, CrowS-Pairs, BBQ |
| `/consistency-reliability` | POST | Sycophancy, stability tests |
| `/misinformation-factuality` | POST | Knowledge cutoff, temporal tests |
| `/refusal-consistency` | POST | Refusal under pressure tests |

### Privacy & Hallucination
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/privacy-redteam` | POST | Data extraction, prompt leakage |
| `/hallucination-confidence` | POST | Token confidence analysis |

## Authentication

All endpoints (except `/` and `/health`) require an API key:

```python
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your-api-key",
}
```

## Rate Limits

- **Batch endpoints**: 10 requests/minute
- **Realtime endpoints**: 30 requests/minute
- **Health endpoints**: 60 requests/minute

## Model Configuration

All endpoints that interact with LLMs require a model configuration:

```python
model = {
    "name": "openai-gpt-4o-mini",       # Must contain 'openai' or 'huggingface'
    "description": "GPT-4o-mini",        # Human-readable description
    "model_name": "gpt-4o-mini",         # Actual model name for API calls
}
```

For HuggingFace models:
```python
model = {
    "name": "huggingface-llama",
    "description": "Llama 2 via TGI",
    "base_url": "http://localhost:8080",  # TGI endpoint
}
```
