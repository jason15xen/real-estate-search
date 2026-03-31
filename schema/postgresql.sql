-- ============================================================
-- Real Estate Search — PostgreSQL + PostGIS Schema
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
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

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

-- Spatial index (Phase 3: Proximity)
CREATE INDEX idx_properties_geom ON properties USING GIST(geom);

-- Room instance lookups (Phase 4: Feature Matching)
CREATE INDEX idx_room_instances_property_id ON room_instances(property_id);
CREATE INDEX idx_room_instances_room_type ON room_instances(room_type);
CREATE INDEX idx_room_instances_prop_room ON room_instances(property_id, room_type);
CREATE INDEX idx_room_instances_features_gin ON room_instances USING GIN(features);
CREATE INDEX idx_room_instances_features_text_trgm ON room_instances USING GIN(features_text gin_trgm_ops);

-- School lookups
CREATE INDEX idx_property_schools_property_id ON property_schools(property_id);
CREATE INDEX idx_property_schools_name ON property_schools USING GIN(school_name gin_trgm_ops);
CREATE INDEX idx_property_schools_distance ON property_schools(distance_miles);
