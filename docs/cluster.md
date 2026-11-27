# M2M cluster deployment

A coordinator routes queries to edge nodes, each running the edge API with its own
database, and merges the results. The code is in `src/m2m/cluster/` (`client.py`,
`edge_node.py`, `router.py`, `balancer.py`, `sharding.py`, `aggregator.py`,
`health.py`, `protocol.py`, `sync.py`); the HTTP side is
`m2m.api.coordinator_api`.

## Local test with Docker Compose

Prerequisites: Docker and the Compose plugin. Two keys are **required**; Compose
fails if either is missing.

```bash
cp deploy/.env.example deploy/.env          # then edit both values
# or: python -c "import secrets; print(secrets.token_urlsafe(32))"
docker compose -f deploy/docker-compose.yml up --build -d
```

| Service | Image / command | Published on the host | Key |
|---|---|---|---|
| coordinator | `uvicorn m2m.api.coordinator_api:app --port 8000` | `127.0.0.1:8080` | `M2M_COORDINATOR_API_KEY` |
| edge-1 .. edge-3 | `uvicorn m2m.api.edge_api:app --port 8000` | `127.0.0.1:8081`, `8082`, `8083` | `M2M_API_KEY` |

Inside the Compose network the edges register with the coordinator on port
`8000`; the host ports are bound to `127.0.0.1` only. The images run as a
non-root user.

<!-- P10b: verificar P15 -->
Known limitation in 2.4.0: the coordinator does **not** forward a key when it
fans a search out to the edges. With `M2M_API_KEY` set on the edges, each edge
answers `401` to the coordinator and `POST /search` on the coordinator returns
empty results. Until this is fixed, either keep the edges' published ports on
localhost and rely on that, or accept that the coordinator's search is not
authenticated end to end. The coordinator has no `GET /health` endpoint yet; the
Compose healthcheck opens a TCP connection to port 8000 instead.

## Coordinator API

| Method and path | Purpose |
|---|---|
| `POST /register` | An edge announces itself (id, URL, capacity). |
| `POST /heartbeat` | An edge refreshes its registration and load. |
| `POST /route` | Returns the edges a query would be routed to. |
| `POST /search` | Fans the query out and merges the results. |

All four require `M2M_COORDINATOR_API_KEY` (`Authorization: Bearer` or `X-API-Key`)
when the variable is set. It is read when the module is imported, so it must be in
the environment before the process starts.

The coordinator keeps its edge registry in memory: run it as a **single replica**.
Scaling out means an external registry, which 2.4.0 does not have.

## Kubernetes

Manifests: `deploy/k8s/deployment.yaml` (one coordinator, three edges) and
`deploy/k8s/service.yaml`.

```bash
docker build -t m2m-vector-search:latest -f deploy/Dockerfile .
kubectl create secret generic m2m-keys \
  --from-literal=M2M_API_KEY=<edge key> \
  --from-literal=M2M_COORDINATOR_API_KEY=<coordinator key>
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl get pods
```

The deployments set `runAsNonRoot` (uid/gid 1000), `seccompProfile: RuntimeDefault`,
`allowPrivilegeEscalation: false`, drop all capabilities, declare resource
requests and run `tcpSocket` probes on port 8000. The manifests were not
validated with `kubectl` on the machine that produced them: run
`kubectl apply --dry-run=server` first.

## Monitoring

<!-- P10b: verificar P15 -->
There is **no `/metrics` endpoint** in 2.4.0. The edge nodes keep counters (active
queries, query latency) in `m2m.cluster.edge_node` and report them in their
heartbeats, but nothing exports them over HTTP. `monitoring/prometheus.yml` is
kept in the repository for when an exporter exists; pointing Prometheus at the
pods will return `404` today. Use `GET /health` on the edges and the
`tcpSocket` probes for now.
