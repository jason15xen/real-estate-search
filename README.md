# Real Estate Search Engine

AI-powered real estate search engine using a multi-phase deterministic pipeline.
Returns **only** properties matching **all** criteria — accurate results, not just similar ones.

## Tech Stack

- **FastAPI** + **Uvicorn** (async web server)
- **PostgreSQL 16** + **PostGIS 3.4** (geospatial queries)
- **Azure OpenAI GPT-5.1** (query parsing, vision analysis, geocoding)
- **asyncpg** (async PostgreSQL driver)
- **Docker Compose** (deployment)

---

## Quick Start

### 1. Setup Environment

```bash
cp .env.example .env
# Fill in your Azure OpenAI credentials in .env
```

### 2. Run with Docker Compose

```bash
docker-compose up --build
```

- App: `http://localhost:8888`
- PostgreSQL: `localhost:5433`
- Schema auto-initializes from `schema/postgresql.sql`

### 3. Run Locally (without Docker)

```bash
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8888
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AZURE_OPENAI_API_KEY` | *(required)* | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | *(required)* | Azure cognitive services endpoint |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-5.1` | Deployment name |
| `AZURE_OPENAI_API_VERSION` | `2025-04-01-preview` | API version |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_USER` | `admin` | PostgreSQL user |
| `POSTGRES_PASSWORD` | `admin123` | PostgreSQL password |
| `POSTGRES_DB` | `real_estate` | Database name |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## API Endpoints

### `POST /search`

Natural language property search with optional map bounds and structured filters.

**Request:**
```json
{
  "query": "Family home near good schools",
  "bounds": {
    "north": 47.7,
    "south": 47.5,
    "east": -122.2,
    "west": -122.4
  },
  "filters": {
    "price_min": 500000,
    "price_max": 550000,
    "beds_min": 1,
    "baths_min": 2,
    "property_types": ["SINGLE_FAMILY", "CONDO"],
    "financing": ["cash", "fha"],
    "sqft_min": 255,
    "sqft_max": 555,
    "year_from": 1990,
    "year_to": 2010
  },
  "debug": true
}
```

- `query` — natural language search (required). LLM extracts features, proximity, location, and any hard filters not specified in `filters`.
- `bounds` — optional. Map viewport (north/south/east/west as floats).
- `filters` — optional. Explicit hard filters that **override** any matching field the LLM might extract from `query`. Per-field override:
  - `price_min` / `price_max` (USD)
  - `beds_min`, `baths_min` (min only)
  - `property_types[]` — array of `SINGLE_FAMILY` / `CONDO` / `TOWNHOUSE` / `MANUFACTURED` / `MULTI_FAMILY` (case-insensitive). OR semantics.
  - `financing[]` — e.g., `["cash", "fha", "va_loan"]`. OR semantics. Sourced from Zillow's `listingTerms`.
  - `sqft_min` / `sqft_max`
  - `year_from` / `year_to` (year_built bounds)
- `debug` — when `true`, the response includes `parsed_query`, `stats`, `filter_steps` for inspection.

**Response:**
```json
{
  "query": "...",
  "zillowProperties": ["guid1", "guid2"]
}
```

### `GET /health`

Health check with property count.

### `POST /process`

Upload Zillow JSON file to start background image analysis.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "processing",
  "progress": "0/27 properties"
}
```

### `GET /job/{job_id}`

Poll background job status.

### `POST /saveprocesseddata`

Ingest processed data from `src/processed/data.json` into PostgreSQL and rebuild feature registry.

---

## Workflow

### Data Ingestion

```
Zillow JSON file
    │
    ▼
POST /process
    │  Sends each property photo to GPT-5.1 Vision API
    │  Extracts: RoomType + Features per photo
    │  Runs in background (5 concurrent image analyses)
    │
    ▼
src/processed/data.json  (enriched with RoomType & Features)
    │
    ▼
POST /saveprocesseddata
    │  Converts Zillow data → PostgreSQL schema
    │  Upserts properties, rooms, room_instances, schools
    │  Rebuilds in-memory feature registry
    │
    ▼
PostgreSQL (ready for search)
```

### Search Pipeline (4 Phases)

```
User Query: "3 bedrooms, wood flooring, no pool, near good schools"
  (+ optional map bounds: north/south/east/west)
    │
    ▼
Phase 1 — Query Parser (LLM)
    │  Decomposes natural language → structured criteria
    │  Maps synonyms to exact DB feature names
    │  "wood flooring" → "hardwood floors"
    │  "hearth" → "fireplace"
    │
    ▼
Phase 2 — Hard Filters + Map Bounds (PostgreSQL indexed queries)
    │  WHERE bedroom_count = 3
    │  AND has_pool = FALSE
    │  AND geom && ST_MakeEnvelope(...)   ← if bounds provided
    │  Fast elimination using B-tree + GIST spatial indexes
    │
    ▼
Phase 3 — Proximity Filters (PostGIS spatial queries)
    │  "near good schools" → schools with rating >= 7 within 5 miles
    │  Fast path: fuzzy match school names in property_schools table
    │  Fallback: LLM geocoding + ST_DWithin
    │
    ▼
Phase 4 — Feature Matching
    │  Matches features in room_instances
    │  Uses synonym expansion (reconstructed_queries)
    │  Falls back to property description text search
    │  Negated features use set subtraction
    │
    ▼
Results: Property GUIDs matching ALL criteria
```

---

## Supported Query Types

| Type | Examples |
|------|----------|
| **Room Count** | `2 bedrooms`, `at least 3 bathrooms` |
| **Features** | `hardwood floors`, `granite countertops`, `fireplace` |
| **Negation** | `no pool`, `without stone tile` |
| **Room-Specific Features** | `kitchen with white cabinets`, `bedroom with carpet` |
| **Price** | `under $500k`, `between $300k and $600k` |
| **Area** | `at least 2000 sqft` |
| **Location** | `in Seattle`, `in Washington state` |
| **Proximity** | `near Oak Park Elementary`, `near good schools` |
| **Property Type** | `single family`, `condo`, `townhouse` |
| **Property Attributes** | `with pool`, `waterfront`, `built after 2000`, `single story` |
| **Rent** | `rent under $2000/mo` |
| **Map Bounds** | *(sent via `bounds` field — filters to a lat/lng rectangle, e.g. for map viewport)* |

All criteria are combined with **AND** logic — properties must match every condition.

### Map Bounds Filter

The `bounds` field on `/search` filters properties to a rectangular region using PostGIS. Designed for map-based UX (Google Maps / Leaflet / Mapbox) where you want results constrained to the current viewport.

```json
{
  "query": "3 bedroom",
  "bounds": {
    "north": 47.7,
    "south": 47.5,
    "east": -122.2,
    "west": -122.4
  }
}
```

- Applied in **Phase 2** alongside hard filters — single SQL query, uses the GIST index on `geom`
- Combines with `location` criteria (e.g., query `"in Seattle"` + bounds) via AND
- Invalid bounds (non-numeric) are gracefully ignored
- Omitting `bounds` preserves the original search behavior (backwards compatible)

---

## Database Schema

### Tables

- **`properties`** — Main property data (address, price, area, room counts, geom, attributes)
- **`rooms`** — Room types per property with counts
- **`room_instances`** — Individual room features extracted from photos (`features` array + `features_text`)
- **`property_schools`** — Nearby schools with ratings and distance

### Key Indexes

- Room counts, price, area, location — B-tree indexes for fast hard filtering
- `geom` — GIST index for PostGIS spatial queries
- `features` — GIN array index for feature lookups
- `features_text` — GIN trigram index for fuzzy text matching
- `school_name` — GIN trigram index for fuzzy school matching

---

## Project Structure

```
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env / .env.example
├── schema/
│   └── postgresql.sql              # DB schema with PostGIS + indexes
├── config/
│   └── settings.py                 # Pydantic settings (env vars)
├── src/
│   ├── main.py                     # FastAPI app entry point
│   ├── llm_client.py               # Azure OpenAI client factory
│   ├── data/
│   │   ├── database.py             # asyncpg connection pool
│   │   ├── feature_registry.py     # In-memory feature/room type registry
│   │   └── ingest.py               # Data ingestion (mockup.json)
│   ├── models/
│   │   └── search.py               # Pydantic search criteria models
│   ├── search/
│   │   ├── query_parser.py         # LLM query → structured criteria
│   │   ├── filter_engine.py        # Hard filters (SQL WHERE)
│   │   ├── geo_search.py           # Proximity / geospatial filters
│   │   └── orchestrator.py         # 4-phase search pipeline
│   ├── img_analyzer/
│   │   ├── router.py               # /process, /job, /saveprocesseddata
│   │   ├── analyzer.py             # Vision API image analysis
│   │   ├── db_ingest.py            # Zillow data → PostgreSQL
│   │   ├── job_manager.py          # Background job tracking
│   │   ├── models.py               # Pydantic data models
│   │   └── prompt/
│   │       ├── prompt.txt          # Vision API system prompt
│   │       └── feature.txt         # Known features list
│   └── processed/                  # Generated processed data
├── data.json                       # Sample property data
└── mockup.json                     # Test data
```

---

## Design Decisions

1. **Deterministic Search** — Uses PostgreSQL indexed queries + PostGIS, not vector search. Every result matches all criteria.
2. **Feature Registry** — Built at startup from DB. LLM maps user input to exact feature names, eliminating fuzzy matching errors.
3. **Denormalized Room Counts** — Stored in `properties` table for fast filtering without JOINs.
4. **Two-Tier Geospatial** — Fast path checks school table; fallback uses LLM geocoding + PostGIS.
5. **Background Image Processing** — `/process` returns job ID immediately, client polls for completion.
6. **Upsert Pattern** — Data ingestion merges by GUID, allowing incremental updates.
