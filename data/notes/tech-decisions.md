# Technical Decision Log

This document records significant technical decisions made for the AcmeCorp Platform.

---

## TDR-001: Database Selection for Order Service

**Date:** January 15, 2024
**Decision Maker:** Alex Johnson
**Status:** Approved

### Context
The Order Service needs a database for storing order data. Requirements include:
- High write throughput (1000+ orders/minute during peak)
- Flexible schema for order items
- Good aggregation capabilities for analytics

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| PostgreSQL | ACID compliance, familiar | Schema rigidity, join complexity |
| MongoDB | Flexible schema, good writes | Eventual consistency concerns |
| DynamoDB | Serverless, auto-scaling | Query limitations, vendor lock-in |

### Decision
**MongoDB** was selected for the Order Service.

### Rationale
- Order documents map naturally to MongoDB's document model
- Flexible schema allows easy addition of new fields
- Aggregation pipeline meets analytics requirements
- Team has MongoDB experience from previous projects

### Consequences
- Need MongoDB expertise on team
- Must handle eventual consistency in code
- Backup and recovery procedures differ from PostgreSQL

---

## TDR-002: Message Queue Selection

**Date:** January 20, 2024
**Decision Maker:** Sarah Chen
**Status:** Approved

### Context
Need async messaging between microservices for:
- Order events (created, updated, cancelled)
- Inventory updates
- Notification triggers

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| RabbitMQ | Mature, flexible routing | Operational complexity |
| Apache Kafka | High throughput, replay | Overkill for current scale |
| AWS SQS | Managed, simple | Limited routing, AWS-only |

### Decision
**RabbitMQ** was selected as the message broker.

### Rationale
- Flexible routing with exchanges and queues
- Good fit for current message volume
- Can migrate to Kafka later if needed
- Team familiar with AMQP protocol

### Consequences
- Need to manage RabbitMQ cluster
- Implement retry and dead-letter queues
- Monitor queue depth and consumer lag

---

## TDR-003: Authentication Strategy

**Date:** February 1, 2024
**Decision Maker:** Maria Garcia
**Status:** Approved

### Context
Need unified authentication across all services and client applications.

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Session-based | Simple, secure | Scaling challenges, stateful |
| JWT | Stateless, scalable | Token revocation complexity |
| OAuth 2.0 + JWT | Industry standard, flexible | Implementation complexity |

### Decision
**OAuth 2.0 with JWT tokens** was selected.

### Rationale
- Industry standard for modern applications
- Supports SSO for enterprise customers
- JWT enables stateless validation
- Refresh tokens provide secure long sessions

### Consequences
- Implement token refresh mechanism
- Handle token revocation for logout
- Secure token storage in clients

---

## TDR-004: API Gateway Selection

**Date:** February 10, 2024
**Decision Maker:** John Smith
**Status:** Approved

### Context
Need API Gateway for:
- Request routing
- Rate limiting
- Authentication
- Logging and monitoring

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Kong | Feature-rich, plugins | Operational complexity |
| AWS API Gateway | Managed, integrated | Vendor lock-in, cost |
| Nginx | Simple, performant | Limited built-in features |
| Envoy | Modern, Kubernetes-native | Learning curve |

### Decision
**Kong** was selected as the API Gateway.

### Rationale
- Rich plugin ecosystem
- Good balance of features and complexity
- Active community and documentation
- Can run on-premise or managed

### Consequences
- Need Kong expertise
- Plugin development for custom needs
- Database required for Kong cluster

---

## TDR-005: Kubernetes Migration

**Date:** March 1, 2024
**Decision Maker:** Sarah Chen
**Status:** In Progress

### Context
Current deployment on EC2 instances is:
- Manual and error-prone
- Difficult to scale
- Inconsistent across environments

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| AWS EKS | Managed, integrated | Cost, AWS-specific |
| Self-managed K8s | Full control | Operational burden |
| AWS ECS | Simpler, AWS-native | Less flexible |

### Decision
**AWS EKS** was selected for container orchestration.

### Rationale
- Managed control plane reduces ops burden
- Good integration with AWS services
- Industry standard skills
- Easier hiring for K8s experience

### Consequences
- Team needs Kubernetes training
- Migration project required
- New monitoring and logging setup

### Timeline
- Q2 2024: Migration complete
- Owner: John Smith

---

## TDR-006: Event Sourcing for Order Service

**Date:** March 20, 2024
**Decision Maker:** Alex Johnson
**Status:** Proposed (POC in Progress)

### Context
Current order updates are:
- Direct database mutations
- No audit trail
- Difficult to debug issues

### Proposal
Implement event sourcing for Order Service:
- All changes as immutable events
- Current state derived from event stream
- Full audit trail and replay capability

### Options Considered

| Option | Pros | Cons |
|--------|------|------|
| Keep current | Simple, working | No audit, debug issues |
| Event sourcing | Full history, debugging | Complexity, learning curve |
| Audit log only | Simple audit | No replay capability |

### Recommendation
**Event sourcing** with Apache Kafka as event store.

### Next Steps
1. POC by April 30
2. Team training
3. Gradual migration
4. Full implementation Q3

---

*Maintained by: Sarah Chen*
*Last updated: March 2024*
