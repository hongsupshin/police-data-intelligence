"""Data package for Texas Justice Initiative police data ETL.

This package provides tools for loading and preprocessing police shooting
incident data into a PostgreSQL database.
"""

from data.etl.loaders import load_civilians_shot, load_officers_shot
from data.load_data import main

__all__ = ["load_civilians_shot", "load_officers_shot", "main"]
