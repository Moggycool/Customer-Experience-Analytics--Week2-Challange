"""
Configuration file for Bank Reviews Analysis Project
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google Play Store App IDs
APP_IDS = {
    "CBE": os.getenv("CBE_APP_ID", "combanketh.mobilebanking"),
    "BOA": os.getenv("BOA_APP_ID", "boa.boaMobileBanking"),
    "DB": os.getenv("DASHEN_APP_ID", "com.dashen.dashensuperapp"),
}

BANK_NAMES = {
    "CBE": "Commercial Bank of Ethiopia",
    "BOA": "Bank of Abyssinia",
    "DB": "Dashen Bank",
}

# Scraping Configuration
SCRAPING_CONFIG = {
    "reviews_per_bank": int(os.getenv("REVIEWS_PER_BANK", "400")),
    "lang": os.getenv("LANG", "en"),
    "country_fallback": os.getenv("COUNTRY_FALLBACK", "et,us,gb").split(","),
    "max_retries": int(os.getenv("MAX_RETRIES", "3")),
    "scraping_delay": float(os.getenv("SCRAPING_DELAY", "2")),
}

# File Paths
DATA_PATHS = {
    'raw': 'data/raw',
    'processed': 'data/preprocessed',
    'raw_reviews': 'data/raw/reviews_raw.csv',
    'processed_reviews': 'data/preprocessed/reviews_processed.csv',
    'sentiment_results': 'data/preprocessed/reviews_with_sentiment.csv',
    'final_results': 'data/preprocessed/reviews_final.csv',
    'app_info': 'data/raw/app_info.csv'  # <-- new entry
}
