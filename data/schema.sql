-- TJI Police Data Intelligence Database Schema (Normalized Design)
-- This schema follows database normalization best practices to eliminate repeating groups
-- Source data from CSV files will be transformed during loading via ETL script

-- ============================================================================
-- MASTER TABLES (Entities that can be reused across incidents)
-- ============================================================================

-- Officers table: stores unique officer demographics
CREATE TABLE IF NOT EXISTS officers (
    officer_id SERIAL PRIMARY KEY,
    age INTEGER,
    race TEXT,
    gender TEXT,
    name_first TEXT,
    name_last TEXT,
    -- Composite unique constraint to prevent exact duplicates
    -- Note: This may still create duplicates if data quality is poor
    UNIQUE NULLS NOT DISTINCT (name_first, name_last, age, race, gender)
);

-- Civilians table: stores unique civilian demographics
CREATE TABLE IF NOT EXISTS civilians (
    civilian_id SERIAL PRIMARY KEY,
    age INTEGER,
    race TEXT,
    gender TEXT,
    name_first TEXT,
    name_last TEXT,
    name_full TEXT,
    -- Composite unique constraint to prevent exact duplicates
    UNIQUE NULLS NOT DISTINCT (name_first, name_last, age, race, gender)
);

-- Agencies table: stores unique law enforcement agencies
CREATE TABLE IF NOT EXISTS agencies (
    agency_id SERIAL PRIMARY KEY,
    name TEXT,
    city TEXT,
    county TEXT,
    zip TEXT,
    -- Agencies identified by name and location
    UNIQUE NULLS NOT DISTINCT (name, city, county)
);

-- ============================================================================
-- INCIDENT TABLES (Core incident data - one row per incident)
-- ============================================================================

-- Incidents where police shot civilians
CREATE TABLE IF NOT EXISTS incidents_civilians_shot (
    incident_id SERIAL PRIMARY KEY,
    ois_report_no TEXT UNIQUE,
    date_ag_received DATE,
    date_incident DATE,
    time_incident TIME,

    -- Location
    incident_address TEXT,
    incident_city TEXT,
    incident_county TEXT,
    incident_zip TEXT,

    -- Incident context
    incident_result_of TEXT,
    incident_call_other TEXT,

    -- Weapon information
    weapon_reported_by_media TEXT,
    weapon_reported_by_media_category TEXT,
    deadly_weapon BOOLEAN,

    -- Metadata
    num_officers_recorded INTEGER,
    multiple_officers_involved BOOLEAN,
    officer_on_duty BOOLEAN,
    num_reports_filed INTEGER,
    num_rows_about_this_incident INTEGER,

    -- Narratives
    cdr_narrative TEXT,
    custodial_death_report BOOLEAN,
    lea_narrative_published TEXT,
    lea_narrative_shorter TEXT
);

-- Incidents where civilians shot officers
CREATE TABLE IF NOT EXISTS incidents_officers_shot (
    incident_id SERIAL PRIMARY KEY,
    ois_report_no TEXT UNIQUE,
    date_ag_received DATE,
    date_incident TIMESTAMP,

    -- Location
    incident_address TEXT,
    incident_city TEXT,
    incident_county TEXT,
    incident_zip TEXT,

    -- Metadata
    num_civilians_recorded INTEGER,
    civilian_harm TEXT,  -- NONE, DEATH, or INJURY
    civilian_suicide BOOLEAN
);

-- ============================================================================
-- JUNCTION TABLES (Many-to-many relationships)
-- ============================================================================

-- Officers involved in shooting civilians (shooters)
CREATE TABLE IF NOT EXISTS incident_civilians_shot_officers_involved (
    id SERIAL PRIMARY KEY,
    incident_id INTEGER NOT NULL REFERENCES incidents_civilians_shot(incident_id) ON DELETE CASCADE,
    officer_id INTEGER NOT NULL REFERENCES officers(officer_id) ON DELETE CASCADE,
    officer_sequence INTEGER,  -- Original position (1, 2, 3...) from CSV
    caused_injury BOOLEAN,  -- Only recorded for some officers in original data
    UNIQUE (incident_id, officer_id, officer_sequence)
);

-- Civilians who were shot by police (victims)
CREATE TABLE IF NOT EXISTS incident_civilians_shot_victims (
    id SERIAL PRIMARY KEY,
    incident_id INTEGER NOT NULL REFERENCES incidents_civilians_shot(incident_id) ON DELETE CASCADE,
    civilian_id INTEGER NOT NULL REFERENCES civilians(civilian_id) ON DELETE CASCADE,
    civilian_died BOOLEAN,
    UNIQUE (incident_id, civilian_id)
);

-- Officers who were shot by civilians (victims)
CREATE TABLE IF NOT EXISTS incident_officers_shot_victims (
    id SERIAL PRIMARY KEY,
    incident_id INTEGER NOT NULL REFERENCES incidents_officers_shot(incident_id) ON DELETE CASCADE,
    officer_id INTEGER NOT NULL REFERENCES officers(officer_id) ON DELETE CASCADE,
    officer_harm TEXT,  -- INJURY or DEATH
    UNIQUE (incident_id, officer_id)
);

-- Civilians who shot officers (shooters)
CREATE TABLE IF NOT EXISTS incident_officers_shot_shooters (
    id SERIAL PRIMARY KEY,
    incident_id INTEGER NOT NULL REFERENCES incidents_officers_shot(incident_id) ON DELETE CASCADE,
    civilian_id INTEGER NOT NULL REFERENCES civilians(civilian_id) ON DELETE CASCADE,
    civilian_sequence INTEGER,  -- Original position (1, 2, 3) from CSV
    UNIQUE (incident_id, civilian_id, civilian_sequence)
);

-- Links agencies to civilians_shot incidents
CREATE TABLE IF NOT EXISTS incident_civilians_shot_agencies (
    id SERIAL PRIMARY KEY,
    incident_id INTEGER NOT NULL REFERENCES incidents_civilians_shot(incident_id) ON DELETE CASCADE,
    agency_id INTEGER NOT NULL REFERENCES agencies(agency_id) ON DELETE CASCADE,
    agency_sequence INTEGER,  -- Original position (1-11) from CSV
    report_date DATE,
    person_filling_out_name TEXT,
    person_filling_out_email TEXT,
    UNIQUE (incident_id, agency_id, agency_sequence)
);

-- Links agencies to officers_shot incidents
CREATE TABLE IF NOT EXISTS incident_officers_shot_agencies (
    id SERIAL PRIMARY KEY,
    incident_id INTEGER NOT NULL REFERENCES incidents_officers_shot(incident_id) ON DELETE CASCADE,
    agency_id INTEGER NOT NULL REFERENCES agencies(agency_id) ON DELETE CASCADE,
    agency_sequence INTEGER,  -- Original position (1-2) from CSV
    report_date DATE,
    person_filling_out_name TEXT,
    person_filling_out_email TEXT,
    UNIQUE (incident_id, agency_id, agency_sequence)
);

-- ============================================================================
-- SUPPORTING TABLES (One-to-many relationships)
-- ============================================================================

-- Media coverage for civilians_shot incidents
CREATE TABLE IF NOT EXISTS media_coverage_civilians_shot (
    id SERIAL PRIMARY KEY,
    incident_id INTEGER NOT NULL REFERENCES incidents_civilians_shot(incident_id) ON DELETE CASCADE,
    media_url TEXT,
    coverage_sequence INTEGER,  -- Original position (1-4) from CSV
    UNIQUE (incident_id, coverage_sequence)
);

-- Media coverage for officers_shot incidents
CREATE TABLE IF NOT EXISTS media_coverage_officers_shot (
    id SERIAL PRIMARY KEY,
    incident_id INTEGER NOT NULL REFERENCES incidents_officers_shot(incident_id) ON DELETE CASCADE,
    media_url TEXT,
    coverage_sequence INTEGER,  -- Original position (1-3) from CSV
    UNIQUE (incident_id, coverage_sequence)
);

-- ============================================================================
-- INDEXES (Optimized for common query patterns)
-- ============================================================================

-- Incident lookups by location and date
CREATE INDEX IF NOT EXISTS idx_civ_shot_city_date ON incidents_civilians_shot(incident_city, date_incident);
CREATE INDEX IF NOT EXISTS idx_civ_shot_date ON incidents_civilians_shot(date_incident);
CREATE INDEX IF NOT EXISTS idx_civ_shot_county ON incidents_civilians_shot(incident_county);
CREATE INDEX IF NOT EXISTS idx_off_shot_city_date ON incidents_officers_shot(incident_city, date_incident);
CREATE INDEX IF NOT EXISTS idx_off_shot_date ON incidents_officers_shot(date_incident);
CREATE INDEX IF NOT EXISTS idx_off_shot_county ON incidents_officers_shot(incident_county);

-- Person lookups
CREATE INDEX IF NOT EXISTS idx_officers_name ON officers(name_first, name_last);
CREATE INDEX IF NOT EXISTS idx_officers_race ON officers(race);
CREATE INDEX IF NOT EXISTS idx_civilians_name ON civilians(name_first, name_last);
CREATE INDEX IF NOT EXISTS idx_civilians_race ON civilians(race);

-- Agency lookups
CREATE INDEX IF NOT EXISTS idx_agencies_name ON agencies(name);
CREATE INDEX IF NOT EXISTS idx_agencies_city ON agencies(city);

-- Junction table lookups (for joins)
CREATE INDEX IF NOT EXISTS idx_civ_shot_off_involved_incident ON incident_civilians_shot_officers_involved(incident_id);
CREATE INDEX IF NOT EXISTS idx_civ_shot_off_involved_officer ON incident_civilians_shot_officers_involved(officer_id);
CREATE INDEX IF NOT EXISTS idx_civ_shot_victims_incident ON incident_civilians_shot_victims(incident_id);
CREATE INDEX IF NOT EXISTS idx_civ_shot_victims_civilian ON incident_civilians_shot_victims(civilian_id);
CREATE INDEX IF NOT EXISTS idx_off_shot_victims_incident ON incident_officers_shot_victims(incident_id);
CREATE INDEX IF NOT EXISTS idx_off_shot_victims_officer ON incident_officers_shot_victims(officer_id);
CREATE INDEX IF NOT EXISTS idx_off_shot_shooters_incident ON incident_officers_shot_shooters(incident_id);
CREATE INDEX IF NOT EXISTS idx_off_shot_shooters_civilian ON incident_officers_shot_shooters(civilian_id);
