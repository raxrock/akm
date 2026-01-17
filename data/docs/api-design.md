# API Design Specification

## Overview

This document describes the REST API design for AcmeCorp Platform. All APIs follow OpenAPI 3.0 specification and are documented in Swagger.

## Authentication

All endpoints require JWT authentication except for `/auth/login` and `/auth/register`.

### Login Endpoint
```
POST /api/v1/auth/login
```
Request body:
- email: string (required)
- password: string (required)

Returns JWT access token and refresh token.

## User Endpoints

### Get User Profile
```
GET /api/v1/users/{userId}
```
Returns user profile information. Requires `user:read` scope.

### Update User Profile
```
PUT /api/v1/users/{userId}
```
Updates user profile. Requires `user:write` scope.

## Order Endpoints

### Create Order
```
POST /api/v1/orders
```
Creates a new order. Validates inventory via Inventory Service before processing.

Request body:
- items: array of {productId, quantity}
- shippingAddress: object
- paymentMethod: string

### Get Order Status
```
GET /api/v1/orders/{orderId}
```
Returns order details and current status.

### Cancel Order
```
DELETE /api/v1/orders/{orderId}
```
Cancels an order if not yet shipped. Triggers inventory restoration.

## Inventory Endpoints

### Check Stock
```
GET /api/v1/inventory/{productId}
```
Returns current stock level for a product.

### Reserve Stock
```
POST /api/v1/inventory/reserve
```
Reserves inventory for an order. Called by Order Service.

## Rate Limits

| Endpoint Type | Rate Limit |
|--------------|------------|
| Authentication | 10 req/min |
| Read operations | 100 req/min |
| Write operations | 50 req/min |

## Error Codes

- 400: Bad Request - Invalid input
- 401: Unauthorized - Missing or invalid token
- 403: Forbidden - Insufficient permissions
- 404: Not Found - Resource doesn't exist
- 429: Too Many Requests - Rate limit exceeded
- 500: Internal Server Error

## Versioning

APIs are versioned using URL path versioning (`/api/v1/`, `/api/v2/`). Breaking changes require new version.

## Contact

API questions: api-support@acmecorp.com
Maintained by: Platform Team (Sarah Chen)
