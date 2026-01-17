# Security Guidelines

## Overview

This document outlines security practices for the AcmeCorp Platform. Security review conducted by Jennifer Walsh, Chief Security Officer.

## Authentication & Authorization

### Password Requirements
- Minimum 12 characters
- Must include uppercase, lowercase, numbers, and symbols
- Passwords hashed using bcrypt with cost factor 12
- Password history: last 10 passwords cannot be reused

### Multi-Factor Authentication
MFA is required for:
- Admin accounts
- API key generation
- Production deployments

Supported MFA methods:
- TOTP (Google Authenticator, Authy)
- SMS verification
- Hardware keys (YubiKey)

### Session Management
- JWT tokens expire after 15 minutes
- Refresh tokens expire after 7 days
- Sessions invalidated on password change

## Data Protection

### Encryption at Rest
- Database encryption using AES-256
- File storage encrypted via AWS S3 SSE
- Encryption keys managed by AWS KMS

### Encryption in Transit
- TLS 1.3 required for all connections
- Certificate pinning for mobile apps
- Internal services use mTLS

### PII Handling
Personal Identifiable Information is classified as sensitive:
- Names, addresses, phone numbers
- Email addresses
- Payment information

PII must be:
- Encrypted at rest
- Masked in logs
- Retained for maximum 2 years

## Vulnerability Management

### Security Scanning
- SAST: SonarQube scans on every PR
- DAST: Weekly OWASP ZAP scans
- Dependency scanning: Snyk daily scans

### Incident Response
Contact security team: security@acmecorp.com
Emergency: Page Jennifer Walsh via PagerDuty

Severity levels:
- P1 (Critical): Data breach, system compromise
- P2 (High): Vulnerability in production
- P3 (Medium): Security misconfiguration
- P4 (Low): Best practice deviation

## Compliance

The platform complies with:
- SOC 2 Type II
- GDPR
- PCI DSS (for payment processing)

Annual audits conducted by Deloitte.

## Access Control

### Role-Based Access
| Role | Permissions |
|------|------------|
| Admin | Full system access |
| Developer | Code deployment, logs access |
| Support | Read-only customer data |
| Analyst | Anonymized analytics data |

### Network Security
- Production network isolated in private VPC
- Bastion host required for SSH access
- All access logged to CloudTrail

## Related Documents

- Incident Response Plan: `incident-response.md`
- Data Classification: `data-classification.md`
- Architecture: `architecture.md`
