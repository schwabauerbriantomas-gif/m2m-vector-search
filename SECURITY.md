# Security Policy

## Supported Versions

The following versions receive active security updates:

| Version  | Supported          |
| -------- | ------------------ |
| ≥ 2.0.0  | :white_check_mark: |
| 1.x.x    | :x:                |
| < 1.0.0  | :x:                |

> **Current stable release**: `2.0.0`

## Security Scope

M2M is a **local-first, offline vector database**. The primary attack surface is limited to:

- The optional REST API (`m2m.api.edge_api`) exposed via FastAPI/uvicorn.
- The WAL and SQLite persistence layer (`m2m.storage`).
- Deserialization of pickled index files (`.pkl` shards).

### REST API Security

- By default the API binds to `127.0.0.1` (localhost). **Never expose it to the public internet without authentication middleware.**
- All collection endpoints accept arbitrary JSON payloads — validate and sanitize inputs before deploying in multi-tenant environments.
- No authentication is built in. Use reverse proxies (nginx, Traefik) or API gateways with token validation for production deployments.

### Persistence Security

- WAL files (`.wal`) and SQLite databases contain raw vector data and metadata. Restrict file-system permissions appropriately.
- Pickle-based index shards (`.pkl`) are **not safe to load from untrusted sources**. Only load indexes you created yourself.
- Backup archives (`.tar.gz`) should be encrypted at rest when they contain sensitive embeddings.

## Reporting a Vulnerability

If you discover a security vulnerability in M2M, please **do not open a public GitHub issue**.

Instead:

1. **Email the maintainers directly** with a subject line: `[SECURITY] M2M Vulnerability Report`.
2. Include: affected version, description of the vulnerability, reproduction steps, and potential impact.
3. We will acknowledge receipt within **48 hours** and provide an estimated resolution timeline.

Please do **not** discuss potential vulnerabilities publicly until an official patch or advisory has been released.

## Disclosure Policy

- We follow a **90-day coordinated disclosure** policy.
- After a fix is released, we will publish a security advisory on GitHub.
- Critical vulnerabilities may result in expedited patches outside the normal release cycle.
