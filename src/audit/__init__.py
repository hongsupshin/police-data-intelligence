"""Discrepancy audit: flag mismatches between official records and news coverage.

Inverts the enrichment pipeline's premise: instead of treating the TJI/OAG
database as ground truth and filling missing fields, the audit runs the
pipeline over incidents whose records are complete and flags fields where
news coverage contradicts the official record. The official record is the
audit object, not the reference.

Flags are ADVISE-only. A flagged mismatch has three possible causes —
government error, news error, extraction error — and no free ground truth
distinguishes them, so every reported flag must be human-verified before it
is treated as an error in the official record.
"""
