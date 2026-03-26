# Azure Services Setup Guide

## What to create

### 1. Azure Database for PostgreSQL (Flexible Server)

| Setting | Value |
|---------|-------|
| Region | East US 2 |
| Version | PostgreSQL 16 |
| SKU | Burstable B2s (dev) / General Purpose D4s_v3 (prod) |
| Storage | 32 GB |
| Extensions | `postgis`, `pg_trgm` |
| Database name | `real_estate` |
| Firewall | Allow Azure services access |

After creating, run `postgresql.sql` to create tables and indexes.

**Info I need back:**
- Host: `xxx.postgres.database.azure.com`
- Port: `5432`
- Database: `real_estate`
- Username
- Password

---

### 2. Azure AI Search

| Setting | Value |
|---------|-------|
| Region | East US 2 |
| SKU | Basic ($75/mo) for dev, Standard S1 for prod |
| Replicas | 1 |
| Partitions | 1 |

After creating, use `azure-ai-search-index.json` to create the index.
You can do this in Azure Portal → AI Search → Indexes → Add Index (JSON).

**Info I need back:**
- Endpoint: `https://xxx.search.windows.net`
- Admin API Key
- Index name: `property-features`

---

### 3. Azure Container Apps (for deployment)

| Setting | Value |
|---------|-------|
| Region | East US 2 |
| Environment | Create new Container Apps Environment |
| Runtime | Python 3.11 |
| Ingress | Enabled, external, port 8000 |
| Min instances | 1 |
| Max instances | 10 |

**Info I need back:**
- Container Apps URL
- Container Registry login server, username, password

---

## Already have (no action needed)

- Azure Anthropic API (Opus + Haiku endpoints in East US 2)

---

## Important: Everything must be in East US 2 region

The Anthropic API endpoints are in East US 2. All other services
must be in the same region for lowest latency.
