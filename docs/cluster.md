# M2M Cluster Deployment Guide

This guide walks you through deploying the M2M Vector Search Cluster using Docker or Kubernetes.

## 1. Local Testing with Docker Compose
The easiest way to test the full Swarm capability is using the provided `docker-compose.yml`.

### Prerequisites
- Docker & Docker Compose installed

### Steps
1. Navigate to the project root.
2. Run `docker-compose -f deploy/docker-compose.yml up --build -d`
3. This creates a virtual network with 1 Coordinator (port 8080) and 3 Edge Nodes (ports 8081-8083).
4. Coordinator will be available at `http://localhost:8080`
5. Edges will be available at `http://localhost:8081`, `8082`, `8083`.

## 2. Kubernetes Deployment
For production deployments, Kubernetes manifests are available in `deploy/k8s/`.

### Steps
1. Ensure your Kubernetes cluster is running (e.g., Minikube, EKS, GKE).
2. Build your docker image or ensure `m2m-vector-search:latest` is accessible.
3. Apply the deployments: `kubectl apply -f deploy/k8s/deployment.yaml`
4. Apply the services: `kubectl apply -f deploy/k8s/service.yaml`
5. Wait for pods to become ready: `kubectl get pods`

## 3. Monitoring via Prometheus
The cluster natively exposes `/metrics` API endpoints tracking `active_queries` and `query_latency_ms`. 
Refer to `monitoring/prometheus.yml` to configure your Prometheus server to autonomously scrape telemetry directly from the Coordinator and Edge pods globally.
