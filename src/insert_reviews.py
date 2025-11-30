"""
insert_reviews.py

Inserts cleaned Google Play Store reviews into PostgreSQL.

Reads:
- data/preprocessed/google_play_processed_reviews.csv
- data/preprocessed/sentiment_preprocessed.csv
- data/raw/google_play_app_info.csv

Writes:
- Inserts into reviews(review_id, bank_id, review_text, rating,
  review_date, sentiment_label, sentiment_score, source)
"""
import os
import sys
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd
from psycopg2.extras import execute_batch


# ------------------------------------------------------------
# Project path setup
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from src.connection import get_connection, release_connection  # pylint: disable=import-error

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------
BASE_DIR = PARENT_DIR
REVIEWS_FILE = os.path.join(BASE_DIR, "data", "preprocessed", "google_play_processed_reviews.csv")
SENTIMENT_FILE = os.path.join(BASE_DIR, "data", "preprocessed", "sentiment_preprocessed.csv")
APP_INFO_FILE = os.path.join(BASE_DIR, "data", "raw", "google_play_app_info.csv")

# ------------------------------------------------------------
# Logging helper
# ------------------------------------------------------------
def log(msg):
    """ Logging Function"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# ------------------------------------------------------------
# SQLAlchemy engine for reading bank_id mapping
# ------------------------------------------------------------
ENGINE = create_engine("postgresql+psycopg2://moggy:MoGGy8080@localhost:5432/bank_reviews")

# ------------------------------------------------------------
# Load CSV files
# ------------------------------------------------------------
def load_csv(file_path, required_cols):
    """Load a CSV safely with UTF-8 and check required columns."""
    log(f"Loading CSV: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except FileNotFoundError:
        log(f"ERROR: File not found → {file_path}")
        sys.exit(1)
    except Exception as e:
        log(f"ERROR: Failed to read CSV → {e}")
        sys.exit(1)

    # Trim whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    

    # Validate columns
    if not required_cols.issubset(df.columns.str.lower()):
        log(f"ERROR: CSV must contain columns: {required_cols}")
        sys.exit(1)

    return df

# ------------------------------------------------------------
# Merge reviews, sentiment, and bank info
# ------------------------------------------------------------
def load_and_merge():
    """A function to load and merge data from the three CSV files located in data folder"""
    df_reviews = load_csv(REVIEWS_FILE, {"review_text", "rating", "review_date", "bank_code"})
    df_sentiment = load_csv(SENTIMENT_FILE, {"sentiment_label", "sentiment_score"})
    df_appinfo = load_csv(APP_INFO_FILE, {"bank_code", "bank_name"}).drop_duplicates(subset=["bank_code"])

    log("Merging review and sentiment data...")
    sentiment_cols = [c for c in df_sentiment.columns if c not in df_reviews.columns]
    df_merged = pd.concat([df_reviews.reset_index(drop=True),
                           df_sentiment[sentiment_cols].reset_index(drop=True)], axis=1)

    # Merge bank_name
    df_merged = df_merged.drop(columns=['bank_name'], errors='ignore')
    df_merged = df_merged.merge(df_appinfo[['bank_code', 'bank_name']], on='bank_code', how='left')

    # Lowercase column names except bank_name
    df_merged.columns = [c.lower() if c != 'bank_name' else c for c in df_merged.columns]

    # Remove duplicate review_ids
    if 'review_id' in df_merged.columns:
        df_merged = df_merged.drop_duplicates(subset=['review_id'])

    # Map bank_name → bank_id from database
    log("Fetching bank_id mapping from database...")
    banks_df = pd.read_sql("SELECT bank_id, bank_name FROM banks;", ENGINE)
    banks_df.columns = banks_df.columns.str.lower()
    banks_df = banks_df.drop_duplicates(subset=['bank_name'])

    df_merged = df_merged.merge(banks_df, on='bank_name', how='left')
    df_merged = df_merged.dropna(subset=['bank_id'])

    df_merged['source'] = 'Google Play'
    log(f"Prepared {len(df_merged)} unique reviews for insertion.")
    return df_merged

# ------------------------------------------------------------
# Insert reviews into PostgreSQL
# ------------------------------------------------------------
def insert_reviews(df):
    """A function to insert the merged data in to reviews table"""
    log("Inserting reviews into database...")
    conn = None
    cur = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Avoid duplicate review_ids
        cur.execute("SELECT review_id FROM reviews;")
        existing_ids = set(row[0] for row in cur.fetchall())
        df_to_insert = df[~df['review_id'].isin(existing_ids)]

        if df_to_insert.empty:
            log("No new reviews to insert.")
            return

        INSERT_QUERY = """
            INSERT INTO reviews (
                review_id, bank_id, review_text, rating, review_date,
                sentiment_label, sentiment_score, source
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        records = [
            (
                row['review_id'],
                row['bank_id'],
                row['review_text'],
                float(row['rating']),
                row['review_date'],
                row['sentiment_label'],
                float(row['sentiment_score']),
                row['source']
            )
            for _, row in df_to_insert.iterrows()
        ]

        execute_batch(cur, INSERT_QUERY, records, page_size=50)
        conn.commit()
        log(f"Inserted {len(records)} new reviews successfully!")
    except Exception as e:
        log(f"ERROR: Failed to insert reviews → {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            release_connection(conn)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    df_to_insert = load_and_merge()
    insert_reviews(df_to_insert)
