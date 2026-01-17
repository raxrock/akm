# Deployment Guide

## Overview

This document describes the deployment process for AcmeCorp Platform services. All deployments are managed through GitHub Actions and ArgoCD.

## Environments

| Environment | Purpose | URL | Owner |
|-------------|---------|-----|-------|
| Development | Daily development | dev.acmecorp.internal | All engineers |
| Staging | Pre-production testing | staging.acmecorp.internal | QA Team |
| Production | Live traffic | api.acmecorp.com | Platform Team |

## CI/CD Pipeline

### Pipeline Stages

1. **Build**
   - Run linting (ESLint, Flake8)
   - Run unit tests
   - Build Docker image
   - Push to ECR

2. **Test**
   - Deploy to ephemeral environment
   - Run integration tests
   - Run E2E tests
   - Security scanning (Snyk)

3. **Deploy to Staging**
   - Automatic on merge to `main`
   - Smoke tests
   - Performance tests

4. **Deploy to Production**
   - Manual approval required
   - Canary deployment (10% traffic)
   - Full rollout after 30 minutes

### GitHub Actions Workflow

```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and push
        run: |
          docker build -t $ECR_REPO:$GITHUB_SHA .
          docker push $ECR_REPO:$GITHUB_SHA
```

## Service-Specific Instructions

### User Service

**Dependencies:**
- PostgreSQL 14
- Redis 7

**Environment Variables:**
```
DATABASE_URL=postgresql://user:pass@host:5432/users
REDIS_URL=redis://host:6379
JWT_SECRET=<secret>
```

**Health Check:** `GET /health`

**Deployment Owner:** Maria Garcia

### Order Service

**Dependencies:**
- MongoDB 6
- RabbitMQ 3.11
- User Service
- Inventory Service

**Environment Variables:**
```
MONGODB_URI=mongodb://host:27017/orders
RABBITMQ_URL=amqp://host:5672
USER_SERVICE_URL=http://user-service:8000
INVENTORY_SERVICE_URL=http://inventory-service:8000
```

**Health Check:** `GET /health`

**Deployment Owner:** Alex Johnson

### Inventory Service

**Dependencies:**
- MySQL 8
- Redis 7
- WMS API

**Environment Variables:**
```
MYSQL_HOST=host
MYSQL_DATABASE=inventory
WMS_API_URL=https://wms.vendor.com/api
WMS_API_KEY=<key>
```

**Health Check:** `GET /health`

**Deployment Owner:** David Lee

### API Gateway (Kong)

**Configuration:**
- Rate limiting: 100 req/min per user
- Authentication: JWT validation
- Logging: CloudWatch

**Deployment:** Managed via Kong declarative config

**Deployment Owner:** John Smith

## Rollback Procedures

### Automatic Rollback
ArgoCD automatically rolls back if:
- Health checks fail for 5 minutes
- Error rate exceeds 5%
- P99 latency exceeds 2 seconds

### Manual Rollback

```bash
# Via ArgoCD CLI
argocd app rollback <app-name> <revision>

# Via kubectl
kubectl rollout undo deployment/<deployment-name>
```

## Database Migrations

### PostgreSQL (User Service)
```bash
# Run migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### MongoDB (Order Service)
Migrations managed via `migrate-mongo`:
```bash
migrate-mongo up
migrate-mongo down
```

## Monitoring

### Dashboards
- Grafana: https://grafana.acmecorp.internal
- Service dashboards for each microservice
- SLO dashboard for platform health

### Alerts
- PagerDuty integration for P1/P2 issues
- Slack #alerts channel for P3/P4

### Key Metrics
- Request rate
- Error rate
- Latency (P50, P95, P99)
- CPU/Memory utilization

## Emergency Contacts

| Role | Name | Contact |
|------|------|---------|
| Platform On-Call | Rotating | PagerDuty |
| Platform Lead | Sarah Chen | sarah.chen@acmecorp.com |
| Backend Lead | Alex Johnson | alex.johnson@acmecorp.com |
| Security | Jennifer Walsh | security@acmecorp.com |

## Deployment Checklist

- [ ] All tests passing
- [ ] Code reviewed and approved
- [ ] Database migrations tested on staging
- [ ] Feature flags configured
- [ ] Monitoring alerts configured
- [ ] Rollback plan documented
- [ ] Stakeholders notified
