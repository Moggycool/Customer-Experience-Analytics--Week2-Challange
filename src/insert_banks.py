"""
insert_banks.py
Inserts bank names and app titles into the `banks` table.

Reads:
- data/raw/google_play_app_info.csv

Writes:
- Inserts into banks(bank_name, app_name)
"""
import os
import sys
from datetime import datetime
import pandas as pd
from psycopg2.extras import execute_batch


# ------------------------------------------------------------
# Ensure project root is in Python path BEFORE local imports
# ------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from src.connection import get_connection, release_connection  # pylint: disable=import-error

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------
BASE_DIR = PARENT_DIR
APP_INFO_FILE = os.path.join(BASE_DIR, "data", "raw", "google_play_app_info.csv")


# ------------------------------------------------------------
# Logging helper
# ------------------------------------------------------------
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


# ------------------------------------------------------------
# Load CSV file
# ------------------------------------------------------------
def load_app_info():
    """Load and clean bank/app info CSV"""
    log("Loading bank/app info file...")

    try:
        df = pd.read_csv(APP_INFO_FILE, encoding="utf-8-sig")
    except FileNotFoundError:
        log(f"ERROR: File not found → {APP_INFO_FILE}")
        sys.exit(1)
    except Exception as e:
        log(f"ERROR: Failed to read CSV → {e}")
        sys.exit(1)

    # Ensure lowercase headers
    df.columns = df.columns.str.lower()

    # Validate required columns
    required = {"bank_name", "title"}
    if not required.issubset(df.columns):
        log(f"ERROR: CSV must contain columns: {required}")
        sys.exit(1)

    # Trim whitespace and drop duplicates
    df["bank_name"] = df["bank_name"].str.strip()
    df["title"] = df["title"].str.strip()
    df_unique = df.drop_duplicates(subset=["bank_name"])

    log(f"Loaded {len(df_unique)} unique banks.")
    return df_unique


# ------------------------------------------------------------
# Insert into PostgreSQL
# ------------------------------------------------------------
INSERT_QUERY = """
    INSERT INTO banks (bank_name, app_name)
    VALUES (%s, %s)
    ON CONFLICT (bank_name) DO NOTHING;
"""


def insert_banks(df):
    """Insert bank/app info into the database"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        records = [(row["bank_name"], row["title"]) for _, row in df.iterrows()]

        log(f"Inserting {len(records)} banks...")
        execute_batch(cur, INSERT_QUERY, records, page_size=50)
        conn.commit()
        log("Bank insert completed successfully!")
    except Exception as e:
        log(f"ERROR: Failed to insert banks → {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            release_connection(conn)


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    df = load_app_info()
    insert_banks(df)
