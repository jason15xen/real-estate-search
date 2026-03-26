-- ============================================================
-- Real Estate Search — PostgreSQL + PostGIS Schema
-- ============================================================
-- Prerequisites:
--   1. Create database: CREATE DATABASE real_estate;
--   2. Connect to it:   \c real_estate
--   3. Enable extensions below
-- ============================================================

-- Extensions
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================
-- TABLES
-- ============================================================

-- Main property table (one row per property)
CREATE TABLE properties (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,

    -- Address
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

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Rooms table (one row per room type per property)
CREATE TABLE rooms (
    id              SERIAL PRIMARY KEY,
    property_id     INTEGER NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
    room_type       TEXT NOT NULL,       -- 'Bedroom', 'Bathroom', 'Kitchen', etc.
    count           INTEGER NOT NULL,    -- number of this room type
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Room instances table (one row per physical room)
CREATE TABLE room_instances (
    id              SERIAL PRIMARY KEY,
    room_id         INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    property_id     INTEGER NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
    room_type       TEXT NOT NULL,       -- denormalized for faster queries
    instance_index  INTEGER NOT NULL,    -- 0, 1, 2... within the room type
    features        TEXT[] NOT NULL,     -- array of feature strings
    features_text   TEXT NOT NULL,       -- concatenated features for text search
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- INDEXES
-- ============================================================

-- Property filters (used in Phase 2: Hard Filters)
CREATE INDEX idx_properties_bedroom_count ON properties(bedroom_count);
CREATE INDEX idx_properties_bathroom_count ON properties(bathroom_count);
CREATE INDEX idx_properties_price ON properties(price_usd);
CREATE INDEX idx_properties_area ON properties(area_sqft);
CREATE INDEX idx_properties_city ON properties(city);
CREATE INDEX idx_properties_state ON properties(state);
CREATE INDEX idx_properties_country ON properties(country);
CREATE INDEX idx_properties_district ON properties(district);

-- Spatial index (used in Phase 3: Proximity Filters)
CREATE INDEX idx_properties_geom ON properties USING GIST(geom);

-- Room instance lookups
CREATE INDEX idx_room_instances_property_id ON room_instances(property_id);
CREATE INDEX idx_room_instances_room_type ON room_instances(room_type);

-- Text search on features
CREATE INDEX idx_room_instances_features_gin ON room_instances USING GIN(features);
CREATE INDEX idx_room_instances_features_text_trgm ON room_instances USING GIN(features_text gin_trgm_ops);

-- ============================================================
-- EXAMPLE QUERIES (for testing)
-- ============================================================

-- Find properties with exactly 3 bedrooms, under $500K
-- SELECT * FROM properties
-- WHERE bedroom_count = 3 AND price_usd <= 500000;

-- Find properties within 5 miles (8046 meters) of a point
-- SELECT * FROM properties
-- WHERE ST_DWithin(geom, ST_MakePoint(-97.725, 30.220)::geography, 8046);

-- Find properties with 3 bedrooms + within 5 miles + under $500K
-- SELECT * FROM properties
-- WHERE bedroom_count = 3
--   AND price_usd <= 500000
--   AND ST_DWithin(geom, ST_MakePoint(-97.725, 30.220)::geography, 8046);

-- Find room instances that have 'fireplace' in a Bedroom
-- SELECT ri.*, p.name
-- FROM room_instances ri
-- JOIN properties p ON p.id = ri.property_id
-- WHERE ri.room_type = 'Bedroom'
--   AND 'fireplace' = ANY(ri.features);
