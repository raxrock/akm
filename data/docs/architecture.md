# System Architecture Overview

## Introduction

The AcmeCorp Platform is a distributed microservices architecture designed by Sarah Chen and her team. The system was initially proposed in Q1 2024 and has been in production since March 2024.

## Core Components

### API Gateway
The API Gateway is built using Kong and handles all incoming requests. It was implemented by John Smith and provides rate limiting, authentication, and request routing.

### User Service
The User Service manages user authentication and profiles. It uses PostgreSQL as its primary database and Redis for session caching. Lead developer: Maria Garcia.

Key features:
- OAuth 2.0 authentication
- JWT token management
- User profile CRUD operations

### Order Service
The Order Service handles all e-commerce transactions. It depends on the User Service for authentication and the Inventory Service for stock validation. Developed by the Backend Team led by Alex Johnson.

Technologies used:
- Python with FastAPI
- MongoDB for order storage
- RabbitMQ for event messaging

### Inventory Service
The Inventory Service tracks product stock levels across multiple warehouses. It integrates with the Warehouse Management System (WMS) via REST API. Owner: David Lee.

### Notification Service
The Notification Service sends emails, SMS, and push notifications. It uses AWS SES for emails and Twilio for SMS. Maintained by the Platform Team.

## Data Flow

1. Client requests arrive at the API Gateway
2. Gateway authenticates via User Service
3. Requests are routed to appropriate microservice
4. Services communicate via RabbitMQ message queue
5. Responses are aggregated and returned to client

## Dependencies

- User Service → PostgreSQL, Redis
- Order Service → MongoDB, RabbitMQ, User Service, Inventory Service
- Inventory Service → MySQL, WMS API
- Notification Service → AWS SES, Twilio, RabbitMQ

## Team Structure

- Platform Team: Sarah Chen (Lead), John Smith, Emily Brown
- Backend Team: Alex Johnson (Lead), Maria Garcia, David Lee
- Frontend Team: Michael Wang (Lead), Lisa Anderson

## Related Documents

- See `api-design.md` for API specifications
- See `deployment.md` for deployment procedures
- See `security.md` for security guidelines
