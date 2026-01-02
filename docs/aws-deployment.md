# AWS Deployment (Terraform)

This guide deploys the full RedTeamGO stack (API, frontend, Redis, Prometheus, Grafana) on AWS ECS Fargate behind an Application Load Balancer.

## Architecture Summary
- ECS Fargate task with multiple containers: `app`, `frontend`, `redis`, `prometheus`, `grafana`
- ALB listeners:
  - `80` -> frontend
  - `8080` -> API
  - `9090` -> Prometheus
  - `3002` -> Grafana
- ECR repositories for backend and frontend images
- CloudWatch log groups per container

## Prerequisites
- AWS account + credentials configured (`aws configure` or env vars)
- Docker
- Terraform >= 1.5

## Deploy

### 1) Initialize Terraform
```bash
cd infra/aws
terraform init
```

### 2) Apply Infrastructure
This creates VPC networking, ECS, ALB, and ECR repositories.
```bash
terraform apply
```

Save the outputs; you will need the ALB DNS name.

### 3) Build and Push Images to ECR
Set the outputs (example using Terraform output values):
```bash
export AWS_REGION=us-east-1
export APP_REPO=$(terraform output -raw app_ecr_repository)
export FRONTEND_REPO=$(terraform output -raw frontend_ecr_repository)
export ALB_DNS=$(terraform output -raw alb_dns_name)
```

Authenticate Docker to ECR:
```bash
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin ${APP_REPO%/*}
```

Build and push the backend image:
```bash
docker build -t ${APP_REPO}:v1 .
docker push ${APP_REPO}:v1
```

Build and push the frontend image (bake the API URL):
```bash
docker build \
  --build-arg REACT_APP_API_URL=http://${ALB_DNS}:8080 \
  -t ${FRONTEND_REPO}:v1 \
  ./frontend

docker push ${FRONTEND_REPO}:v1
```

### 4) Update the Task Definition Images
Update the Terraform variables to point at the new tags, then apply:
```bash
terraform apply \
  -var app_image=${APP_REPO}:v1 \
  -var frontend_image=${FRONTEND_REPO}:v1 \
  -var cors_origins=http://${ALB_DNS}
```

If you keep using `:latest`, force a new deployment after pushing:
```bash
aws ecs update-service \
  --cluster redteamgo-cluster \
  --service redteamgo-service \
  --force-new-deployment
```

### 5) Access the Stack
- Frontend: `http://<alb_dns>`
- API: `http://<alb_dns>:8080`
- Prometheus: `http://<alb_dns>:9090`
- Grafana: `http://<alb_dns>:3002`

## Notes
- Restrict `allowed_ingress_cidrs` in `infra/aws/variables.tf` before production use.
- Grafana/Prometheus are intentionally public for the demo stack; lock them down in a real environment.
- The ECS task runs all containers together for a simple demo deployment. Split into separate services for production hardening.

## Scale Down / Up (Cost Control)
If you want to keep infrastructure but stop compute, scale the ECS service to 0 between runs:
```bash
aws ecs update-service --cluster redteamgo-cluster --service redteamgo-service --desired-count 0
```

Scale back up when you need it:
```bash
aws ecs update-service --cluster redteamgo-cluster --service redteamgo-service --desired-count 1
```

You can also set `desired_count` in Terraform and re-apply if you prefer infra-as-code for scaling.
