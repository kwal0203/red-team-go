output "alb_dns_name" {
  description = "ALB DNS name for the stack."
  value       = aws_lb.main.dns_name
}

output "frontend_url" {
  description = "Frontend URL (HTTP)."
  value       = "http://${aws_lb.main.dns_name}"
}

output "api_url" {
  description = "API URL (HTTP)."
  value       = "http://${aws_lb.main.dns_name}:8080"
}

output "prometheus_url" {
  description = "Prometheus URL (HTTP)."
  value       = "http://${aws_lb.main.dns_name}:9090"
}

output "grafana_url" {
  description = "Grafana URL (HTTP)."
  value       = "http://${aws_lb.main.dns_name}:3002"
}

output "app_ecr_repository" {
  description = "ECR repository URL for the backend service."
  value       = aws_ecr_repository.app.repository_url
}

output "frontend_ecr_repository" {
  description = "ECR repository URL for the frontend service."
  value       = aws_ecr_repository.frontend.repository_url
}
