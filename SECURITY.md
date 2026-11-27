# Security Policy

## Supported versions

| Version | Supported | Notes |
|---|---|---|
| `2.4.0` | :white_check_mark: | Published on PyPI on 2026-11-27. Receives security fixes. |
| `2.3.1` and earlier | :x: | No backports. Several findings listed below are only fixed in `2.4.0`. |

Report the version from `python -c "import m2m; print(m2m.__version__)"`.

## What M2M is

A local-first vector database written in Python/NumPy. The attack surface is:

- the optional HTTP API (`m2m.api.edge_api`, `m2m.api.coordinator_api`, served by uvicorn),
- the files under a `storage_path` (NumPy shards, SQLite metadata, WAL, signed index),
- deserialization of index files (`pickle` behind a restricted unpickler).

There is **no TLS, no user management and no encryption at rest**. Put M2M behind
your own reverse proxy, authenticator and disk encryption.

## Network exposure

- `python -m m2m` (and the `m2m` console script) listens on **`127.0.0.1:8080`** by
  default. `M2M_HOST` and `M2M_PORT` change that (`docs/CONFIGURATION.md`).
- Binding to a non-loopback host **without** `M2M_API_KEY` logs a warning at start-up:
  anyone who can reach the port can read and write the database.
- The Dockerfiles (`Dockerfile`, `deploy/Dockerfile`) set `ENV M2M_HOST=0.0.0.0` so the
  service is reachable inside the container network. **Set `M2M_API_KEY` before
  publishing the port** (`-p 8080:8080`).
- `deploy/docker-compose.yml` publishes the coordinator on `127.0.0.1:8080` and the
  edges on `127.0.0.1:8081-8083`, and fails if a key is missing.

- The root `docker-compose.yml` publishes the edge on `127.0.0.1:8080` only and passes
  `M2M_API_KEY` and `M2M_RATE_LIMIT_PER_MIN` to the container when they are set [P15;
  `tests/test_audit_closure.py`]. It was not validated with `docker compose` (not available
  on the machine that produced it): run `docker compose config` before relying on it.

## Authentication

Authentication is **opt-in**. Without `M2M_API_KEY` the edge API is open by design
(local development, Quick Start).

With `M2M_API_KEY` set, clients send `Authorization: Bearer <key>` or `X-API-Key: <key>`.
Keys are compared with `hmac.compare_digest` (constant time).

| Edge API endpoint | Key when `M2M_API_KEY` is set |
|---|---|
| `GET /health`, `GET /v1/health` | never required |
| `GET /v1/stats` | required (`401`) |
| `POST /v1/collections`, `DELETE /v1/collections/{name}` | required (`401`) |
| `POST /v1/collections/{name}/vectors` (insert) | required (`401`) |
| `POST /v1/collections/{name}/search`, `/query`, `/energy`, `/explore`, `GET .../suggest` | required (`401`) |
| `POST /v1/admin/checkpoint`, `POST /v1/admin/backup` | **always**: `401` with a wrong key, `503` when no key is configured (fail closed) |
| `POST /ingest`, `POST /search` (legacy) | required (`401`) |
| `GET /v1/collections`, `GET /v1/collections/{name}`, `GET .../stats` | required (`401`) |
| `GET`/`PUT`/`DELETE /v1/collections/{name}/vectors/{vector_id}` | required (`401`, answered before the collection lookup) |

Every endpoint validates the key when one is configured; only the two `/health` paths stay
open [P15; `tests/test_audit_closure.py`]. The vector `GET`/`PUT`/`DELETE` endpoints answer
`401` before they look the collection up, so a wrong key does not reveal whether a
collection exists.

The coordinator API (`/register`, `/heartbeat`, `/route`, `/search`) checks
`M2M_COORDINATOR_API_KEY` (same headers) on every request. The variable is read once,
when the module is imported.

On a fan-out the coordinator sends `M2M_EDGE_API_KEY` to each edge; set it to the edges'
`M2M_API_KEY`, otherwise every edge answers `401` and `POST /search` returns empty results.
The coordinator's own key is never forwarded to the edges. `GET /health` on the
coordinator answers without a key [P15; `tests/test_audit_closure.py`].

## Request limits

- **Body size**: requests larger than `M2M_MAX_BODY_BYTES` (default 32 MiB) are
  rejected with `413` before the body is read.
- **Vectors per insert**: at most `100_000` vectors per request (`422` above that,
  checked before the arrays are validated; `tests/test_audit_api_storage.py`).
- **Validation**: vectors with `NaN`, `Inf` or norm below `1e-6` are rejected with
  `400`; the same for unknown metadata-filter operators, duplicate ids in a batch
  and non-integer `k`.
- **Rate limiting (opt-in)**: `M2M_RATE_LIMIT_PER_MIN` sets a per-client-IP limit over a
  sliding 60 s window; absent, empty, `0` or negative means off, which is the default.
  Above the limit the edge answers `429` with a `Retry-After` header, and rejected
  requests are not counted. With the limit off, `429` is never returned [P15;
  `tests/test_audit_closure.py`].
- **CORS**: origins come from `M2M_CORS_ORIGINS` (comma-separated). The default `*`
  is open; with the wildcard, credentials are not allowed. The coordinator API has
  no CORS middleware.

## Files on disk

- A `storage_path` holds raw vectors (`.npy`, loaded with `allow_pickle=False`),
  a SQLite database with metadata and documents, and the WAL. Nothing is encrypted:
  restrict filesystem permissions and encrypt the disk if the embeddings are
  sensitive. Backup archives (`.tar.gz`) have the same content.
- **Signed storage (opt-in)**: with `M2M_HMAC_SECRET` set, `save()`/checkpoint
  writes `index/manifest.idx`, an HMAC-SHA256 signature over the storage that is
  verified when the storage is reopened or loaded with `load(path)`. A mismatch
  raises `IndexTamperedError`. A storage written **without** a manifest is loaded
  with a warning (so existing databases keep working). Without the secret nothing is
  signed and nothing is verified.
- **Signed index**: `M2MPersistence.save_index()` / `load_index()` require
  `M2M_HMAC_SECRET` (`RuntimeError` otherwise) and verify the signature over the
  whole payload **before** deserializing it through `_RestrictedUnpickler`, a
  whitelist-based unpickler that rejects everything that is not a NumPy array or a
  built-in. Only load index files you created, and keep the secret out of the
  storage directory: the signature proves integrity, not origin, once the secret
  leaks.
- **Path confinement**: `storage_path` and `backup_path` reject `..`. The paths used
  to save/load the signed index and the shards are confined to the storage directory:
  `..`, absolute paths and symlinks resolving outside it raise `ValueError`.
- `Collection.storage_path` is supplied by the client in `CreateCollectionRequest`:
  in a multi-tenant deployment, validate it or restrict it to a designated root.
- `M2MPersistence.restore(backup_path)` validates its source **before** removing
  anything: `..` raises `ValueError`, a missing source `FileNotFoundError`, a file
  `NotADirectoryError`, and a source equal to, inside or above the storage (symlinks
  resolved) `ValueError`. It is not exposed over HTTP [P15; `tests/test_audit_closure.py`].

## Container and Kubernetes hardening

- Images run as the non-root user `m2m` and install only the `api` extra.
- `deploy/k8s/deployment.yaml`: `runAsNonRoot`, `runAsUser/runAsGroup 1000`,
  `seccompProfile: RuntimeDefault`, `allowPrivilegeEscalation: false`,
  `capabilities.drop: [ALL]`, resource requests, liveness/readiness probes.
  The coordinator runs with a single replica because its edge registry lives in
  memory.
- API keys come from a Kubernetes `Secret` (`M2M_API_KEY`, `M2M_COORDINATOR_API_KEY`).
- The manifests were not validated with `docker` or `kubectl` on the machine that
  produced them; run `docker compose config` and `kubectl apply --dry-run=server`
  before relying on them.

## Reporting a vulnerability

Please **do not open a public GitHub issue**.

1. Use GitHub's private vulnerability reporting on the repository, or email the
   maintainer (address in `pyproject.toml`, `[project.authors]`) with the subject
   `[SECURITY] M2M Vulnerability Report`.
2. Include the affected version (`m2m.__version__`), a description, reproduction
   steps and the potential impact.
3. You will get an acknowledgement within **48 hours** and an estimated timeline.

Please do not discuss a potential vulnerability publicly until a fix or an advisory
has been released.

## Disclosure policy

- Coordinated disclosure, **90 days**.
- After a fix is released a GitHub security advisory is published.
- Critical vulnerabilities may be fixed outside the normal release cycle.
