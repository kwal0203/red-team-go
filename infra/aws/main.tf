locals {
  name_prefix   = var.project_name
  tags          = { Project = var.project_name, ManagedBy = "terraform" }
  app_image     = var.app_image != "" ? var.app_image : "${aws_ecr_repository.app.repository_url}:latest"
  frontend_image = var.frontend_image != "" ? var.frontend_image : "${aws_ecr_repository.frontend.repository_url}:latest"
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = merge(local.tags, { Name = "${local.name_prefix}-vpc" })
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = merge(local.tags, { Name = "${local.name_prefix}-igw" })
}

resource "aws_subnet" "public" {
  count                   = length(var.public_subnet_cidrs)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = var.public_subnet_cidrs[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags                    = merge(local.tags, { Name = "${local.name_prefix}-public-${count.index + 1}" })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  tags   = merge(local.tags, { Name = "${local.name_prefix}-public-rt" })
}

resource "aws_route" "public_internet" {
  route_table_id         = aws_route_table.public.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.main.id
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "alb" {
  name        = "${local.name_prefix}-alb-sg"
  description = "ALB ingress for RedTeamGO"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "Frontend"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.allowed_ingress_cidrs
  }

  ingress {
    description = "API"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = var.allowed_ingress_cidrs
  }

  ingress {
    description = "Prometheus"
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = var.allowed_ingress_cidrs
  }

  ingress {
    description = "Grafana"
    from_port   = 3002
    to_port     = 3002
    protocol    = "tcp"
    cidr_blocks = var.allowed_ingress_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, { Name = "${local.name_prefix}-alb-sg" })
}

resource "aws_security_group" "ecs" {
  name        = "${local.name_prefix}-ecs-sg"
  description = "ECS task ingress for RedTeamGO"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "API"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description     = "Frontend"
    from_port       = 3000
    to_port         = 3000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description     = "Prometheus"
    from_port       = 9090
    to_port         = 9090
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description     = "Grafana"
    from_port       = 3001
    to_port         = 3001
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(local.tags, { Name = "${local.name_prefix}-ecs-sg" })
}

resource "aws_lb" "main" {
  name               = "${local.name_prefix}-alb"
  load_balancer_type = "application"
  subnets            = aws_subnet.public[*].id
  security_groups    = [aws_security_group.alb.id]
  tags               = merge(local.tags, { Name = "${local.name_prefix}-alb" })
}

resource "aws_lb_target_group" "frontend" {
  name        = "${local.name_prefix}-frontend-tg"
  port        = 3000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.main.id

  health_check {
    path                = "/"
    matcher             = "200-399"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 5
  }
}

resource "aws_lb_target_group" "app" {
  name        = "${local.name_prefix}-app-tg"
  port        = 8000
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.main.id

  health_check {
    path                = "/health"
    matcher             = "200-399"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 5
  }
}

resource "aws_lb_target_group" "prometheus" {
  name        = "${local.name_prefix}-prom-tg"
  port        = 9090
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.main.id

  health_check {
    path                = "/-/healthy"
    matcher             = "200-399"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 5
  }
}

resource "aws_lb_target_group" "grafana" {
  name        = "${local.name_prefix}-grafana-tg"
  port        = 3001
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.main.id

  health_check {
    path                = "/api/health"
    matcher             = "200-399"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 5
  }
}

resource "aws_lb_listener" "frontend" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.frontend.arn
  }
}

resource "aws_lb_listener" "app" {
  load_balancer_arn = aws_lb.main.arn
  port              = 8080
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

resource "aws_lb_listener" "prometheus" {
  load_balancer_arn = aws_lb.main.arn
  port              = 9090
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.prometheus.arn
  }
}

resource "aws_lb_listener" "grafana" {
  load_balancer_arn = aws_lb.main.arn
  port              = 3002
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.grafana.arn
  }
}

resource "aws_ecs_cluster" "main" {
  name = "${local.name_prefix}-cluster"
  tags = merge(local.tags, { Name = "${local.name_prefix}-cluster" })
}

resource "aws_ecr_repository" "app" {
  name = "${local.name_prefix}-app"
  tags = merge(local.tags, { Name = "${local.name_prefix}-app" })
}

resource "aws_ecr_repository" "frontend" {
  name = "${local.name_prefix}-frontend"
  tags = merge(local.tags, { Name = "${local.name_prefix}-frontend" })
}

data "aws_iam_policy_document" "ecs_task_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ecs_task_execution" {
  name               = "${local.name_prefix}-ecs-exec"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
  tags               = merge(local.tags, { Name = "${local.name_prefix}-ecs-exec" })
}

resource "aws_iam_role_policy_attachment" "ecs_exec_policy" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task" {
  name               = "${local.name_prefix}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_assume.json
  tags               = merge(local.tags, { Name = "${local.name_prefix}-ecs-task" })
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${local.name_prefix}/app"
  retention_in_days = 14
  tags              = local.tags
}

resource "aws_cloudwatch_log_group" "frontend" {
  name              = "/ecs/${local.name_prefix}/frontend"
  retention_in_days = 14
  tags              = local.tags
}

resource "aws_cloudwatch_log_group" "redis" {
  name              = "/ecs/${local.name_prefix}/redis"
  retention_in_days = 14
  tags              = local.tags
}

resource "aws_cloudwatch_log_group" "prometheus" {
  name              = "/ecs/${local.name_prefix}/prometheus"
  retention_in_days = 14
  tags              = local.tags
}

resource "aws_cloudwatch_log_group" "grafana" {
  name              = "/ecs/${local.name_prefix}/grafana"
  retention_in_days = 14
  tags              = local.tags
}

resource "aws_ecs_task_definition" "main" {
  family                   = "${local.name_prefix}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn        = aws_iam_role.ecs_task_execution.arn
  task_role_arn             = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name      = "app"
      image     = local.app_image
      essential = true
      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]
      environment = [
        { name = "REDIS_URL", value = "redis://localhost:6379" },
        { name = "LOG_LEVEL", value = var.log_level },
        { name = "CORS_ORIGINS", value = var.cors_origins }
      ]
      dependsOn = [
        { containerName = "redis", condition = "START" }
      ]
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 40
      }
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.app.name
          awslogs-region        = var.region
          awslogs-stream-prefix = "ecs"
        }
      }
    },
    {
      name      = "frontend"
      image     = local.frontend_image
      essential = true
      portMappings = [
        {
          containerPort = 3000
          hostPort      = 3000
          protocol      = "tcp"
        }
      ]
      dependsOn = [
        { containerName = "app", condition = "START" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.frontend.name
          awslogs-region        = var.region
          awslogs-stream-prefix = "ecs"
        }
      }
    },
    {
      name      = "redis"
      image     = "redis:alpine"
      essential = true
      portMappings = [
        {
          containerPort = 6379
          hostPort      = 6379
          protocol      = "tcp"
        }
      ]
      command = ["redis-server", "--appendonly", "yes"]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.redis.name
          awslogs-region        = var.region
          awslogs-stream-prefix = "ecs"
        }
      }
    },
    {
      name      = "prometheus"
      image     = "prom/prometheus:latest"
      essential = false
      portMappings = [
        {
          containerPort = 9090
          hostPort      = 9090
          protocol      = "tcp"
        }
      ]
      command = [
        "sh",
        "-c",
        "cat <<'EOF' > /etc/prometheus/prometheus.yml\nglobal:\n  scrape_interval: 15s\n  evaluation_interval: 15s\n\nscrape_configs:\n  - job_name: 'redteam-app'\n    static_configs:\n      - targets: ['localhost:8000']\n    metrics_path: '/metrics'\n    scrape_interval: 5s\n\n  - job_name: 'prometheus'\n    static_configs:\n      - targets: ['localhost:9090']\nEOF\n/bin/prometheus --config.file=/etc/prometheus/prometheus.yml --storage.tsdb.path=/prometheus"
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.prometheus.name
          awslogs-region        = var.region
          awslogs-stream-prefix = "ecs"
        }
      }
    },
    {
      name      = "grafana"
      image     = "grafana/grafana:latest"
      essential = false
      portMappings = [
        {
          containerPort = 3001
          hostPort      = 3001
          protocol      = "tcp"
        }
      ]
      environment = [
        { name = "GF_SERVER_HTTP_PORT", value = "3001" },
        { name = "GF_METRICS_ENABLED", value = "true" },
        { name = "GF_METRICS_BASIC_AUTH_ENABLED", value = "false" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.grafana.name
          awslogs-region        = var.region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "main" {
  name            = "${local.name_prefix}-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.main.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = aws_subnet.public[*].id
    security_groups = [aws_security_group.ecs.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.frontend.arn
    container_name   = "frontend"
    container_port   = 3000
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "app"
    container_port   = 8000
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.prometheus.arn
    container_name   = "prometheus"
    container_port   = 9090
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.grafana.arn
    container_name   = "grafana"
    container_port   = 3001
  }

  depends_on = [
    aws_lb_listener.frontend,
    aws_lb_listener.app,
    aws_lb_listener.prometheus,
    aws_lb_listener.grafana
  ]

  tags = merge(local.tags, { Name = "${local.name_prefix}-service" })
}
