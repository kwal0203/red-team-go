# RedTeamGO

## Overview
RedTeamGO is a proof of concept system for evaluating Large Language Models (LLMs) through automated testing of toxicity and bias detection. The system provides both batch processing capabilities for systematic evaluation and real-time analysis for interactive testing.

## Features
- **Toxicity Detection:** Evaluates model outputs for harmful or toxic content
- **Bias Detection:** Analyzes responses for various types of bias (gender, racial, religious, etc.)
- **Batch Processing:** Test multiple prompts systematically
- **Real-time Analysis:** Immediate feedback for individual prompts
- **Multiple Model Support:** Compatible with OpenAI and HuggingFace models
- **Monitoring:** Integrated Prometheus and Grafana dashboards for system metrics

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/RedTeamGO.git
   cd RedTeamGO
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Development Mode
Run the FastAPI server directly with auto-reload for development:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode
Use Docker Compose to run the full stack:
```bash
docker-compose up
```

This will start:
- Backend API (port 8000)
- Frontend UI (port 3000)
- Redis cache
- Prometheus metrics (port 9090)
- Grafana dashboard (port 3002)

### AWS Deployment (Terraform)
Infrastructure-as-code for AWS ECS/Fargate is documented in `docs/aws-deployment.md`.

### Testing the API
You can use the provided test scripts in the `examples` directory:

```bash
# Test bias detection
python test_bias_detection.py

# Test toxicity detection
python examples/toxicity_batch_example.py
```

## API Endpoints

- `POST /toxicity-detection-batch`: Batch toxicity analysis
- `POST /bias-detection-batch`: Batch bias analysis
- `POST /toxicity-detection-realtime`: Real-time toxicity check
- `POST /bias-detection-realtime`: Real-time bias check
- `GET /health`: Service health check

## Directory Structure
```
RedTeamGO
├── services/                  # Core services
│   ├── bias_detection_dbias/  # Bias detection implementation
│   ├── toxicity_detection/    # Toxicity detection implementation
│   └── model_wrappers/        # Model interface implementations
├── frontend/                  # React frontend application
├── examples/                  # API usage examples
├── utils/                     # Shared utilities
├── main.py                    # FastAPI application entry point
└── docker-compose.yml         # Container orchestration
```

## Configuration
The system can be configured through environment variables or Docker Compose:
- Model endpoints and API keys
- CORS settings
- Resource limits
- Monitoring configuration
- Artifact storage configuration (local or S3) in `docs/artifact-storage.md`

## References
The system implements methods from:
- Bias Detection: Raza, S., et al. "Dbias: detecting biases and ensuring fairness in news articles." (2024)
- Toxicity Detection: Perez, E., et al. "Red Teaming Language Models with Language Models." (2022)

## License
This project is licensed under the MIT License. See `LICENSE` for details.
