# TJI Police Data - Setup Instructions

This directory contains the Texas Justice Initiative police shooting data and setup scripts.

## Files and Structure

```
data/
├── tji_civilians-shot.csv        # Police shooting civilians (2,255 records, denormalized)
├── tji_officers-shot.csv         # Civilians shooting police (282 records, denormalized)
├── schema.sql                    # Normalized PostgreSQL database schema
├── load_data.py                  # Main ETL orchestrator (141 lines)
├── test_queries.md               # Test cases for enrichment pipeline
└── etl/                          # Modular ETL package
    ├── cleaners.py               # Data cleaning functions (5 functions)
    ├── entity_managers.py        # Entity deduplication (officers, civilians, agencies)
    ├── loaders.py                # Dataset-specific ETL workflows
    └── config.py                 # Database configuration
```

The ETL code is organized into reusable modules:
- **cleaners.py**: Pure functions for data type conversion (boolean, date, text, etc.)
- **entity_managers.py**: PostgreSQL deduplication logic using INSERT...ON CONFLICT
- **loaders.py**: Complete CSV-to-database transformation pipelines
- **config.py**: Centralized database configuration with environment variable support

## Quick Start

### 1. Create Database

```bash
createdb tji_police_data
```

### 2. Run Tests (Optional)

```bash
# From project root
pytest tests/ -v

# With coverage report
pytest tests/ --cov=data.etl --cov-report=term-missing
```

All ETL modules have unit tests (94% coverage). Tests use mocked databases, so they don't require PostgreSQL running.

### 3. Load Schema and Data

```bash
# From project root
python data/load_data.py

# Or as a module
python -m data.load_data
```

This script will:

- Connect to PostgreSQL database `tji_police_data`
- Create normalized tables from `schema.sql`
- Transform wide-format CSV data into normalized form
- Deduplicate officers, civilians, and agencies
- Print summary statistics

### 4. Verify Data Loaded (Optional)

While unit tests validate the ETL logic with mocked data, this step verifies that data was actually loaded into your PostgreSQL database:

```bash
psql tji_police_data
```

Then run:

```sql
-- Check row counts
SELECT COUNT(*) FROM incidents_civilians_shot;  -- Should be ~1,674
SELECT COUNT(*) FROM incidents_officers_shot;   -- Should be ~282
SELECT COUNT(*) FROM officers;                  -- Deduplicated officers
SELECT COUNT(*) FROM civilians;                 -- Deduplicated civilians
SELECT COUNT(*) FROM agencies;                  -- Deduplicated agencies

-- Sample query: Austin incidents in 2016 (normalized schema with JOINs)
SELECT
    i.date_incident,
    c.name_first,
    c.name_last,
    c.age,
    v.civilian_died,
    i.weapon_reported_by_media
FROM incidents_civilians_shot i
LEFT JOIN incident_civilians_shot_victims v ON i.incident_id = v.incident_id
LEFT JOIN civilians c ON v.civilian_id = c.civilian_id
WHERE i.incident_city = 'AUSTIN'
  AND EXTRACT(YEAR FROM i.date_incident) = 2016
ORDER BY i.date_incident;

-- Check for missing data (enrichment targets)
SELECT
    COUNT(*) as total_records,
    COUNT(weapon_reported_by_media) as has_weapon,
    COUNT(*) - COUNT(weapon_reported_by_media) as missing_weapon,
    ROUND(100.0 * (COUNT(*) - COUNT(weapon_reported_by_media)) / COUNT(*), 1) as pct_missing
FROM incidents_civilians_shot;

-- Officer demographics across all incidents (shows power of normalization)
SELECT
    o.race,
    o.gender,
    COUNT(DISTINCT oi.incident_id) as num_incidents
FROM officers o
JOIN incident_civilians_shot_officers_involved oi ON o.officer_id = oi.officer_id
GROUP BY o.race, o.gender
ORDER BY num_incidents DESC;
```

## Using ETL Modules in Your Code

The refactored ETL package makes data processing functions easily reusable:

```python
# Import cleaning functions
from data.etl.cleaners import clean_text, clean_boolean, clean_date

# Import entity managers (for deduplication)
from data.etl.entity_managers import get_or_create_officer, get_or_create_civilian

# Import loaders (for complete ETL workflows)
from data.etl.loaders import load_civilians_shot, load_officers_shot

# Import main orchestrator
from data import main

# Example: Use cleaning functions in validation agents
raw_value = "  true  "
cleaned = clean_boolean(raw_value)  # Returns: True
```

All modules are fully tested (94% coverage) and ready for use by enrichment agents.

## Database Schema (Normalized Design)

### Why Normalization?

The original CSV files have a **denormalized structure** with repeating columns:

- `officer_age_1`, `officer_age_2`, ..., `officer_age_11` (up to 11 officers per incident)
- `agency_name_1`, `agency_name_2`, ..., `agency_name_11` (up to 11 agencies)
- `civilian_name_first_1`, `civilian_name_first_2`, `civilian_name_first_3` (up to 3 civilians)

It's understandable to have a single flat file for simplicity, but this design leads to several issues:

- **Hard limits**: Cannot store incidents with >11 officers
- **Wasted space**: Most incidents don't have 11 officers → many NULL columns
- **Complex queries**: Finding all incidents for Officer X requires checking 11 columns
- **Poor analytics**: Cannot easily aggregate by officer demographics across incidents
- **Maintenance challenge**: Adding a new officer attribute requires 11 new columns

The normalized schema fixes these issues by separating entities (officers, civilians, agencies) from relationships (who was involved in which incident).

### Master Tables (Deduplicated Entities)

- `officers`: Unique officer demographics (age, race, gender, name), reusable across all incidents
- `civilians`: Unique civilian demographics (age, race, gender, name), reusable across all incidents
- `agencies`: Unique law enforcement agencies (name, city, county, zip), reusable across all incidents

### Incident Tables (One per Incident Type)

- `incidents_civilians_shot` (Police → Civilian Shootings): Core incident data include date, location, weapon, narratives
- `incidents_officers_shot` (Civilian → Officer Shootings): Core incident data include date, location, harm level

### Junction Tables (Relationships)

- `incident_civilians_shot_officers_involved`: Links officers (shooters) to civilian shooting incidents (many-to-many)
- `incident_civilians_shot_victims`: Links civilians (victims) to civilian shooting incidents, stores outcome (civilian_died as boolean)
- `incident_officers_shot_victims`: Links officers (victims) to officer shooting incidents, stores outcome (officer_harm  as INJURY or DEATH)
- `incident_officers_shot_shooters`: Links civilians (shooters) to officer shooting incidents (many-to-many)
- `incident_civilians_shot_agencies` & `incident_officers_shot_agencies`: Links agencies to incidents with report metadata

### Supporting Tables

- `media_coverage_civilians_shot` & `media_coverage_officers_shot`: Media links for each incident (one-to-many)

## Test Cases

See `test_queries.md` for enrichment pipeline test cases:

1. **Single Record Enrichment** - Enrich missing weapon info for one incident
2. **Batch Enrichment** - Process 10 records missing weapon data
3. **Conflict Resolution** - Handle disagreeing news sources
4. **No Results Found** - Gracefully handle incidents with no news coverage

## Troubleshooting

### Database connection failed

```bash
# Check if PostgreSQL is running
brew services list

# Start if needed
brew services start postgresql@15
```

### Permission denied

The config auto-detects your username (`$USER`). To override, use environment variables:

```bash
export POSTGRES_USER=your_username
export POSTGRES_PASSWORD=your_password
python data/load_data.py
```

Or edit `data/etl/config.py` directly if needed.

## Next Steps

After loading the data:

1. **Explore the normalized schema** - Run sample queries in `psql` to understand the data structure
2. **Identify enrichment targets** - Query for records with missing critical fields
3. **Implement enrichment agents** - Build the web search + extraction pipeline (see `test_queries.md`)
4. **Test with real data** - Start with a small batch (10 records) to validate the pipeline
