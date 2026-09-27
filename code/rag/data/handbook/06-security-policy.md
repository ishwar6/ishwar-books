# Information Security Policy

*Version 3.1, April 2026. Owner: Security team (security@lumora.example). Applies to all
employees and contractors.*

## Identity and access

- **Multi-factor authentication (MFA) is mandatory** for every account. Single sign-on is
  through **Okta**; hardware keys are required for engineers with production access.
- **Privileged account passwords rotate every 90 days.** Ordinary user passwords do not
  expire but must be at least 14 characters.
- Access is reviewed **quarterly**. Managers confirm that each report still needs the
  access they have; unconfirmed access is removed.
- Production access is granted just-in-time for a maximum of **8 hours** through the
  access portal, with a ticket reference.

## Devices

- Company laptops must have full-disk encryption (FileVault on macOS, BitLocker on
  Windows) and the endpoint agent installed. Devices without the agent cannot join Okta.
- Screens lock after **5 minutes** of inactivity.
- Personal devices may access email and Slack only through the managed mobile app.

## Data classification

| Level | Examples | Handling |
|---|---|---|
| **Public** | marketing site, published spec sheets | no restrictions |
| **Internal** | handbook, runbooks, roadmaps | employees and contractors only |
| **Confidential** | customer contracts, robot telemetry, source code | need-to-know, encrypted at rest |
| **Restricted** | payroll, personal data, security findings, keys | named individuals, logged access |

Customer robot telemetry is **Confidential**: it must never be copied to personal devices
or pasted into external tools, including public AI assistants.

## Secrets

- Secrets live in **HashiCorp Vault**. Never in code, tickets, Slack or spreadsheets.
- API keys for customers are shown once at creation and can be rotated from the Beacon
  admin console.

## Incident reporting

- Report suspected security incidents (lost laptop, phishing click, exposed key) to
  **security@lumora.example within 1 hour** of noticing. Reporting quickly is never punished.
- The security team acknowledges within 30 minutes during business hours and within
  2 hours otherwise.

## Vendors

- Any vendor that will process Confidential data, or that costs more than **$10,000 per
  year**, needs a vendor security review before signature.

## Training

- Security awareness training is completed at onboarding and **annually** thereafter.
- A phishing simulation runs every quarter; repeat clicks trigger a short refresher, not
  discipline.

## AI tools

Employees may use approved AI assistants (listed in the IT catalog) for Internal data.
Confidential or Restricted data may only be used with tools that have passed vendor
security review and run under a company account.

---
<!-- nav -->

[← Handbook index](../../README.md)
