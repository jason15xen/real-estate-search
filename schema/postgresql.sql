-- ============================================================
-- Real Estate Search — PostgreSQL + PostGIS Schema
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;   -- pgvector: feature-embedding similarity search

-- ============================================================
-- TABLES
-- ============================================================

-- Main property table (one row per property)
CREATE TABLE properties (
    id              SERIAL PRIMARY KEY,
    guid            TEXT NOT NULL UNIQUE, -- original UUID from source data

    -- Address
    name            TEXT NOT NULL,
    street          TEXT,
    district        TEXT,
    city            TEXT NOT NULL,
    state           TEXT NOT NULL,
    postal_code     TEXT,
    country         TEXT NOT NULL,

    -- Location (PostGIS geography point for spatial queries)
    geom            GEOGRAPHY(Point, 4326) NOT NULL,

    -- Numeric fields
    area_sqft       INTEGER NOT NULL,
    price_usd       INTEGER NOT NULL,

    -- Denormalized room counts (for fast filtering without JOINs)
    bedroom_count   INTEGER NOT NULL DEFAULT 0,
    bathroom_count  INTEGER NOT NULL DEFAULT 0,
    kitchen_count   INTEGER NOT NULL DEFAULT 0,
    living_room_count INTEGER NOT NULL DEFAULT 0,
    dining_room_count INTEGER NOT NULL DEFAULT 0,
    garage_count    INTEGER NOT NULL DEFAULT 0,

    -- Property attributes (from Zillow structured data)
    home_type       TEXT,                -- SINGLE_FAMILY, CONDO, TOWNHOUSE, MANUFACTURED, MULTI_FAMILY
    rent_estimate   INTEGER,             -- monthly rent (rentZestimate)
    year_built      INTEGER,
    lot_size_sqft   INTEGER DEFAULT 0,
    stories         INTEGER,
    has_pool        BOOLEAN NOT NULL DEFAULT FALSE,
    -- TRUE = the property has a COVERED pool (water roofed/screened/caged),
    -- derived from images by the ingest pipeline. "Uncovered pool" is not stored
    -- — it is derived at search time as (has a pool) AND NOT has_covered_pool.
    has_covered_pool BOOLEAN NOT NULL DEFAULT FALSE,
    has_waterfront  BOOLEAN NOT NULL DEFAULT FALSE,
    description     TEXT,                -- property description for text search fallback
    financing       TEXT[] NOT NULL DEFAULT '{}',  -- normalized from Zillow listingTerms

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Rooms table (one row per room type per property)
CREATE TABLE rooms (
    id              SERIAL PRIMARY KEY,
    property_id     INTEGER NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
    room_type       TEXT NOT NULL,
    count           INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Room instances table (one row per physical room)
CREATE TABLE room_instances (
    id              SERIAL PRIMARY KEY,
    room_id         INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    property_id     INTEGER NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
    room_type       TEXT NOT NULL,
    instance_index  INTEGER NOT NULL,
    features        TEXT[] NOT NULL,
    features_text   TEXT NOT NULL,
    color           TEXT,                -- single dominant color from 13-color palette, NULL if not extracted yet
    photo_url       TEXT,                -- canonical (highest-res JPEG) URL of the photo this room was derived from
                                          -- NULL for rows created before partial-update support; treated as "unknown origin"
                                          -- so any new POST /process for that property triggers a full re-analyze.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- One instance per (property, room_type, index). The worker computes
    -- instance_index carefully; this makes a double-insert a hard error
    -- rather than silent duplicate rooms. (Fresh-DB constraint — dedupe any
    -- legacy duplicates before applying to an existing database.)
    CONSTRAINT uq_room_instance UNIQUE (property_id, room_type, instance_index)
);

-- Raw ingest staging table — receives every incoming property record via
-- POST /process. A background worker continuously picks up rows whose status
-- is not 'processed' and syncs them into the primary tables above.
--
-- The "what changed" diff for the partial-image path is NOT stored here.
-- It's derivable any time from (data.originalPhotos URLs) ⊖ (room_instances.photo_url
-- for this property), so the worker recomputes it at processing time — that
-- makes the worker idempotent under racy /process traffic.
CREATE TABLE raw_properties (
    id           TEXT PRIMARY KEY,
    data         JSONB NOT NULL,
    status       TEXT NOT NULL CHECK (status IN (
                     'unprocessed',
                     'image_only_processed',
                     'partial_image_only_processed',
                     'processed'
                 )),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_raw_properties_status ON raw_properties(status);


-- Nearby schools per property (from Zillow data)
CREATE TABLE property_schools (
    id              SERIAL PRIMARY KEY,
    property_id     INTEGER NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
    school_name     TEXT NOT NULL,
    rating          INTEGER,
    grades          TEXT,
    distance_miles  NUMERIC(5,2) NOT NULL,
    link            TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Feature embeddings — one row per distinct feature string seen in
-- room_instances.features. Populated by the embedding-sync pipeline at app
-- startup and on NOTIFY feature_change. Used by /search Phase 4: the user's
-- raw feature phrase is embedded and cosine-searched here to retrieve the
-- top-K semantically-nearest DB features as candidates (then a second LLM
-- call filters them for precision).
CREATE TABLE feature_embeddings (
    feature      TEXT PRIMARY KEY,
    embedding    vector(1536) NOT NULL,   -- text-embedding-3-small dimension
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- HNSW index for fast approximate cosine-similarity search.
CREATE INDEX idx_feature_embeddings_hnsw
    ON feature_embeddings USING hnsw (embedding vector_cosine_ops);

-- ============================================================
-- INDEXES
-- ============================================================

-- Property filters (Phase 2: Hard Filters)
CREATE INDEX idx_properties_bedroom_count ON properties(bedroom_count);
CREATE INDEX idx_properties_bathroom_count ON properties(bathroom_count);
CREATE INDEX idx_properties_price ON properties(price_usd);
CREATE INDEX idx_properties_area ON properties(area_sqft);
CREATE INDEX idx_properties_city ON properties(city);
CREATE INDEX idx_properties_state ON properties(state);
CREATE INDEX idx_properties_country ON properties(country);
CREATE INDEX idx_properties_district ON properties(district);

-- Property attribute filters
CREATE INDEX idx_properties_home_type ON properties(home_type);
CREATE INDEX idx_properties_rent ON properties(rent_estimate);
CREATE INDEX idx_properties_year_built ON properties(year_built);
CREATE INDEX idx_properties_lot_size ON properties(lot_size_sqft);
CREATE INDEX idx_properties_has_pool ON properties(has_pool) WHERE has_pool = TRUE;
CREATE INDEX idx_properties_has_covered_pool ON properties(has_covered_pool) WHERE has_covered_pool;
CREATE INDEX idx_properties_has_waterfront ON properties(has_waterfront) WHERE has_waterfront = TRUE;
CREATE INDEX idx_properties_description ON properties USING GIN(description gin_trgm_ops);
CREATE INDEX idx_properties_financing_gin ON properties USING GIN(financing);

-- Spatial index (Phase 3: Proximity)
CREATE INDEX idx_properties_geom ON properties USING GIST(geom);

-- Room instance lookups (Phase 4: Feature Matching)
CREATE INDEX idx_room_instances_property_id ON room_instances(property_id);
CREATE INDEX idx_room_instances_room_type ON room_instances(room_type);
CREATE INDEX idx_room_instances_prop_room ON room_instances(property_id, room_type);
CREATE INDEX idx_room_instances_features_gin ON room_instances USING GIN(features);
CREATE INDEX idx_room_instances_features_text_trgm ON room_instances USING GIN(features_text gin_trgm_ops);
CREATE INDEX idx_room_instances_color ON room_instances(color);
CREATE INDEX idx_room_instances_room_color ON room_instances(room_type, color);

-- School lookups
CREATE INDEX idx_property_schools_property_id ON property_schools(property_id);
CREATE INDEX idx_property_schools_name ON property_schools USING GIN(school_name gin_trgm_ops);
CREATE INDEX idx_property_schools_distance ON property_schools(distance_miles);
