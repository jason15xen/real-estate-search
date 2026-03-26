-- ============================================================================
-- Real Estate Search — PostgreSQL + PostGIS Schema
-- ============================================================================
-- Purpose: Hard filters (room counts, price, area, location) + proximity queries
-- Requires: CREATE EXTENSION postgis;
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS postgis;

-- ============================================================================
-- 1. PROPERTIES — Main property table with denormalized room counts
-- ============================================================================
-- Room counts are denormalized (stored directly) to avoid JOINs during search.
-- This is critical for performance: a single indexed WHERE clause instead of
-- joining + grouping a rooms table on every query.

CREATE TABLE properties (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(255) NOT NULL,

    -- Address
    street          VARCHAR(255),
    district        VARCHAR(100),
    city            VARCHAR(100) NOT NULL,
    state           VARCHAR(100) NOT NULL,
    postal_code     VARCHAR(20),
    country         VARCHAR(100) NOT NULL,

    -- PostGIS geography column (lat/lng as a spatial point)
    -- SRID 4326 = WGS84 (standard GPS coordinates)
    geom            GEOGRAPHY(Point, 4326) NOT NULL,

    -- Property details
    area_sqft       INTEGER NOT NULL,
    price_usd       INTEGER NOT NULL,

    -- Denormalized room counts (avoids JOINs during search)
    bedroom_count   SMALLINT NOT NULL DEFAULT 0,
    bathroom_count  SMALLINT NOT NULL DEFAULT 0,
    kitchen_count   SMALLINT NOT NULL DEFAULT 0,
    living_room_count SMALLINT NOT NULL DEFAULT 0,
    dining_room_count SMALLINT NOT NULL DEFAULT 0,
    garage_count    SMALLINT NOT NULL DEFAULT 0,

    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================================
-- 2. ROOMS — Individual room records
-- ============================================================================
-- Each room instance is a separate row (e.g., a property with 3 bedrooms
-- has 3 rows in this table). This supports per-room feature queries.

CREATE TABLE rooms (
    id              SERIAL PRIMARY KEY,
    property_id     INTEGER NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
    room_type       VARCHAR(50) NOT NULL,    -- 'Bedroom', 'Bathroom', 'Kitchen', etc.
    instance_index  SMALLINT NOT NULL,       -- 0, 1, 2... (which instance of this room type)

    UNIQUE(property_id, room_type, instance_index)
);


-- ============================================================================
-- 3. ROOM_FEATURES — Features for each room instance
-- ============================================================================
-- Stored as separate rows (not arrays) for indexed text search.
-- In real data, features are free text — this allows flexible querying.

CREATE TABLE room_features (
    id              SERIAL PRIMARY KEY,
    room_id         INTEGER NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    property_id     INTEGER NOT NULL REFERENCES properties(id) ON DELETE CASCADE,
    room_type       VARCHAR(50) NOT NULL,    -- Denormalized from rooms table for fast filtering
    feature         TEXT NOT NULL             -- Free text: "accent wall", "modern fireplace", etc.
);


-- ============================================================================
-- 4. INDEXES — Critical for search performance
-- ============================================================================

-- Property-level filters (used in Phase 2: Hard Filters)
CREATE INDEX idx_properties_bedroom_count ON properties(bedroom_count);
CREATE INDEX idx_properties_bathroom_count ON properties(bathroom_count);
CREATE INDEX idx_properties_price ON properties(price_usd);
CREATE INDEX idx_properties_area ON properties(area_sqft);
CREATE INDEX idx_properties_city ON properties(city);
CREATE INDEX idx_properties_state ON properties(state);
CREATE INDEX idx_properties_country ON properties(country);
CREATE INDEX idx_properties_district ON properties(district);

-- Composite index for common multi-column queries
-- e.g., "3 bedrooms under $500K in Texas"
CREATE INDEX idx_properties_bed_price_state
    ON properties(bedroom_count, price_usd, state);

-- PostGIS spatial index (used in Phase 3: Proximity Filters)
-- Makes ST_DWithin queries fast on millions of rows
CREATE INDEX idx_properties_geom ON properties USING GIST(geom);

-- Room features indexes (used for pre-filtering before vector search)
CREATE INDEX idx_room_features_property ON room_features(property_id);
CREATE INDEX idx_room_features_room_type ON room_features(room_type);
CREATE INDEX idx_room_features_feature ON room_features USING GIN(to_tsvector('english', feature));

-- Rooms index
CREATE INDEX idx_rooms_property ON rooms(property_id);
CREATE INDEX idx_rooms_type ON rooms(room_type);


-- ============================================================================
-- 5. EXAMPLE QUERIES (matching our search pipeline)
-- ============================================================================

-- Example 1: "3 bedrooms with accent walls under $500K"
-- Phase 2 (hard filter) + basic feature check
/*
SELECT DISTINCT p.*
FROM properties p
JOIN room_features rf ON rf.property_id = p.id
WHERE p.bedroom_count = 3
  AND p.price_usd <= 500000
  AND rf.room_type = 'Bedroom'
  AND rf.feature ILIKE '%accent wall%';
*/

-- Example 2: "within 5 miles of a school at coordinates (-71.09, 42.36)"
-- Phase 3 (proximity filter)
/*
SELECT p.*
FROM properties p
WHERE ST_DWithin(
    p.geom,
    ST_MakePoint(-71.09, 42.36)::geography,
    8046  -- 5 miles in meters
)
ORDER BY ST_Distance(p.geom, ST_MakePoint(-71.09, 42.36)::geography);
*/

-- Example 3: Combined — "3 bedrooms, fireplace, under $500K, within 5 miles of MIT"
/*
SELECT DISTINCT p.*
FROM properties p
JOIN room_features rf ON rf.property_id = p.id
WHERE p.bedroom_count = 3
  AND p.price_usd <= 500000
  AND rf.room_type = 'Bedroom'
  AND rf.feature ILIKE '%fireplace%'
  AND ST_DWithin(p.geom, ST_MakePoint(-71.09, 42.36)::geography, 8046);
*/
