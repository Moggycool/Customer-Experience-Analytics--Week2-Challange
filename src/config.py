"""
config.py
Centralized configuration module for the Google Play Review Scraper project.

This module stores:
- App IDs for target Ethiopian banks.
- Folder paths for saving raw, processed, and metadata files.
- Scraper configuration parameters.
- General project settings.

All other modules import from here, ensuring a single source of truth.

Usage
-----
from src.config import APP_IDS, BANK_NAMES, SCRAPING_CONFIG, DATA_PATHS
"""

from pathlib import Path

# ============================================================
# Project Root & Directory Structure
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"  # matches preprocessing.py

# Ensure directories exist
for folder in [RAW_DIR, PREPROCESSED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ============================================================
# Data Paths Dictionary
# ============================================================

DATA_PATHS = {
    "raw_reviews": str(RAW_DIR / "google_play_raw_reviews.csv"),
    "processed_reviews": str(PREPROCESSED_DIR / "google_play_processed_reviews.csv"),
    "app_info": str(RAW_DIR / "google_play_app_info.csv"),
}

# ============================================================
# Bank App Mapping
# ============================================================

APP_IDS = {
    "CBE": "com.combanketh.mobilebanking",
    "BOA": "com.boa.boaMobileBanking",
    "DASHEN": "com.dashen.dashensuperapp",
}

BANK_NAMES = {
    "CBE": "Commercial Bank of Ethiopia",
    "BOA": "Bank of Abyssinia",
    "DASHEN": "Dashen Bank",
}

# ============================================================
# Scraping Configuration
# ============================================================

SCRAPING_CONFIG = {
    "reviews_per_bank": 600,
    "lang": "en",
    "country_fallback": ["et", "us", "gb"],
    "max_retries": 3,
    "scraping_delay": 2,
}

# ============================================================
# General Settings
# ============================================================

GENERAL_SETTINGS = {
    "enable_logging": True,
    "log_level": "INFO",
    "timestamp_format": "%Y-%m-%d %H:%M:%S",
}
# Pipeline configuration
PIPELINE_CONFIG = {
    "enable_validation": True,
    "enable_feature_generation": True,
    "enable_export": True,
    "enable_reporting": True,
    "export_formats": ["csv", "json", "parquet"],
    "min_review_length": 3,
    "max_review_length": 10000,
}

# ============================================================
# Helper: Pretty Print Config
# ============================================================


def print_config_summary() -> None:
    """Print an overview of the project's configuration for debugging."""
    print("\n========== CONFIG SUMMARY ==========")
    print("Project Root:", PROJECT_ROOT)
    print("Raw Data Path:", DATA_PATHS["raw_reviews"])
    print("Processed Data Path:", DATA_PATHS["processed_reviews"])
    print("App Info Path:", DATA_PATHS["app_info"])
    print("\nApp IDs:", APP_IDS)
    print("\nScraping Config:", SCRAPING_CONFIG)
    print("====================================\n")
