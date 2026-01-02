variable "region" {
  type        = string
  description = "AWS region to deploy into."
  default     = "us-east-1"
}

variable "project_name" {
  type        = string
  description = "Project name prefix for AWS resources."
  default     = "redteamgo"
}

variable "vpc_cidr" {
  type        = string
  description = "CIDR block for the VPC."
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  type        = list(string)
  description = "Public subnet CIDRs (must be at least two)."
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "allowed_ingress_cidrs" {
  type        = list(string)
  description = "CIDR blocks allowed to reach the ALB."
  default     = ["0.0.0.0/0"]
}

variable "desired_count" {
  type        = number
  description = "Number of ECS tasks to run."
  default     = 1
}

variable "task_cpu" {
  type        = number
  description = "Fargate task CPU units."
  default     = 1024
}

variable "task_memory" {
  type        = number
  description = "Fargate task memory (MiB)."
  default     = 2048
}

variable "cors_origins" {
  type        = string
  description = "CORS origins for the API service."
  default     = "http://localhost:3000"
}

variable "log_level" {
  type        = string
  description = "Application log level."
  default     = "INFO"
}

variable "app_image" {
  type        = string
  description = "Full image URI for the backend service (leave blank to use ECR repo)."
  default     = ""
}

variable "frontend_image" {
  type        = string
  description = "Full image URI for the frontend service (leave blank to use ECR repo)."
  default     = ""
}
