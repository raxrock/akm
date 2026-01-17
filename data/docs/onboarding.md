# Engineering Onboarding Guide

## Welcome to AcmeCorp!

Welcome to the AcmeCorp Engineering team! This guide will help you get set up and productive in your first few weeks.

## Week 1: Setup & Orientation

### Day 1: Administrative Setup
- [ ] Get laptop from IT (contact: it-support@acmecorp.com)
- [ ] Set up email and Slack
- [ ] Complete HR paperwork
- [ ] Meet your manager and mentor

### Day 2: Development Environment

**Required Tools:**
1. **IDE:** VS Code (recommended) or PyCharm
2. **Docker Desktop:** For local development
3. **Git:** Version control
4. **AWS CLI:** For cloud resources
5. **kubectl:** Kubernetes management

**Setup Script:**
```bash
# Clone the setup repository
git clone git@github.com:acmecorp/dev-setup.git
cd dev-setup
./setup.sh
```

### Day 3: Codebase Overview

**Repository Structure:**
- `acmecorp/user-service` - User authentication (Maria Garcia)
- `acmecorp/order-service` - Order processing (Alex Johnson)
- `acmecorp/inventory-service` - Stock management (David Lee)
- `acmecorp/api-gateway` - Kong configuration (John Smith)
- `acmecorp/frontend` - React web app (Michael Wang)

**Key Documents:**
- Architecture: `/docs/architecture.md`
- API Design: `/docs/api-design.md`
- Security: `/docs/security.md`

### Day 4-5: Shadow Sessions

Schedule shadow sessions with:
- Your mentor (assigned team member)
- Platform team engineer
- Backend team engineer
- Frontend team engineer

## Week 2: Deep Dive

### Team-Specific Onboarding

**Platform Team (Lead: Sarah Chen)**
- Infrastructure overview with John Smith
- Monitoring and alerting with Emily Brown
- CI/CD pipeline walkthrough

**Backend Team (Lead: Alex Johnson)**
- Microservices architecture
- Database design patterns
- API development standards

**Frontend Team (Lead: Michael Wang)**
- Component architecture
- State management
- Design system usage

### Access Requests

Request access to:
- [ ] GitHub organization
- [ ] AWS console (read-only initially)
- [ ] Grafana dashboards
- [ ] PagerDuty (if on-call rotation)
- [ ] Confluence wiki

**Access Request Form:** https://acmecorp.atlassian.net/access

## Week 3: First Contribution

### Good First Issues

Each repository has issues labeled `good-first-issue`. Start with:
1. Documentation improvements
2. Test coverage additions
3. Small bug fixes

### Code Review Process

1. Create feature branch from `main`
2. Make changes, write tests
3. Open PR with description
4. Request review from mentor
5. Address feedback
6. Merge after approval

### PR Checklist
- [ ] Tests passing
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No security vulnerabilities
- [ ] Reviewed by at least one team member

## Week 4: Full Participation

### Sprint Participation
- Attend daily standups (10:00 AM)
- Join sprint planning (Mondays)
- Participate in retrospectives (Fridays)

### On-Call Training
If joining on-call rotation:
1. Complete incident response training
2. Shadow on-call engineer for one rotation
3. Be secondary on-call before primary

## Key Contacts

| Role | Name | Slack |
|------|------|-------|
| Engineering Director | Sarah Chen | @sarah.chen |
| Backend Lead | Alex Johnson | @alex.johnson |
| Frontend Lead | Michael Wang | @michael.wang |
| HR Partner | Amy Wilson | @amy.wilson |
| IT Support | Help Desk | #it-help |

## Communication Channels

### Slack Channels
- `#engineering` - General engineering discussions
- `#backend` - Backend team
- `#frontend` - Frontend team
- `#platform` - Platform/DevOps
- `#incidents` - Active incidents
- `#random` - Fun stuff

### Meetings
- **All-Hands:** Monthly, first Thursday
- **Engineering Sync:** Weekly, Wednesdays 2 PM
- **Team Standup:** Daily, 10 AM

## Learning Resources

### Internal
- Engineering Wiki: https://wiki.acmecorp.internal
- Tech Blog: https://tech.acmecorp.com
- Video recordings: Google Drive > Engineering > Presentations

### External Training
- Udemy Business license available
- Conference budget: $2,000/year
- Book reimbursement: $500/year

## FAQs

**Q: How do I get help?**
A: Ask in Slack! Start with your mentor, then team channel.

**Q: What if I break something?**
A: Don't panic! All changes are reversible. Ask for help.

**Q: How do I suggest improvements?**
A: Open a discussion in #engineering or bring to retro.

## Feedback

After your first month, your manager will schedule a 30-60-90 day check-in. We value your fresh perspective - please share feedback on the onboarding process!

---

*Last updated: March 2024*
*Maintained by: Sarah Chen*
