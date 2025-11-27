"""
preprocessing.py
Review Cleaning & Preprocessing Module

This module loads raw scraped Google Play reviews, applies cleaning
transformations, handles missing values (including rating), computes
review length, and saves a clean processed dataset for downstream analysis.
"""

import os
import re
from datetime import datetime
import pandas as pd
from src.config import DATA_PATHS


class ReviewPreprocessor:
    """Preprocess raw Google Play Store reviews."""

    def __init__(self):
        self.raw_path = DATA_PATHS["raw_reviews"]
        self.output_path = DATA_PATHS["processed_reviews"]
        self.df = pd.DataFrame()  # initialize DataFrame

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    # ----------------------------------------------------------------------
    # TEXT CLEANING HELPERS
    # ----------------------------------------------------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean review text: remove URLs, emoji, normalize spacing."""
        if not isinstance(text, str):
            return ""

        text = text.strip().lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove emoji
        text = re.sub(
            r"["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F700-\U0001F77F"
            u"\U0001F780-\U0001F7FF"
            u"\U0001F800-\U0001F8FF"
            u"\U0001F900-\U0001F9FF"
            u"\U0001FA00-\U0001FA6F"
            u"\U0001FA70-\U0001FAFF"
            "]+",
            "",
            text,
        )

        # Remove special characters except punctuation
        text = re.sub(r"[^a-zA-Z0-9.,!? ]", " ", text)

        # Normalize spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def fix_date(date_value):
        """Convert date field to YYYY-MM-DD or return today's date."""
        if not isinstance(date_value, str):
            return datetime.today().strftime("%Y-%m-%d")
        try:
            return pd.to_datetime(date_value).strftime("%Y-%m-%d")
        except Exception:
            return datetime.today().strftime("%Y-%m-%d")

    # ----------------------------------------------------------------------
    # MISSING VALUE HANDLING
    # ----------------------------------------------------------------------
    @staticmethod
    def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate defaults and report per bank."""
        banks = df['bank_name'].unique()
        print("⚠️ Checking and handling missing values per bank...\n")
        for bank in banks:
            bank_df = df[df['bank_name'] == bank]
            total = len(bank_df)
            for col in ['reply_content', 'review_version', 'rating']:
                missing_count = bank_df[col].isna().sum()
                missing_pct = missing_count / total * 100
                print(f"Bank: {bank} ({total} reviews)")
                print(
                    f"  - {col}: {missing_count} missing ({missing_pct:.2f}%)")
                # Fill missing values
                if col == 'reply_content':
                    df.loc[df['bank_name'] == bank,
                           col] = bank_df[col].fillna("")
                elif col == 'review_version':
                    df.loc[df['bank_name'] == bank,
                           col] = bank_df[col].fillna("N/A")
                elif col == 'rating':
                    median_rating = bank_df['rating'].median()
                    df.loc[df['bank_name'] == bank,
                           col] = bank_df[col].fillna(median_rating)
        print()
        return df

    # ----------------------------------------------------------------------
    # MAIN PROCESSING LOGIC
    # ----------------------------------------------------------------------
    def process(self) -> bool:
        """Load, clean, handle missing, add text length, and save processed review data."""
        if not os.path.exists(self.raw_path):
            print(f"❌ Raw reviews file not found: {self.raw_path}")
            return False

        print(f"📥 Loading raw reviews from {self.raw_path}...")
        df = pd.read_csv(self.raw_path)

        if df.empty:
            print("❌ ERROR: Raw reviews file is empty.")
            return False

        # Ensure required fields exist
        required_cols = [
            "review_id",
            "review_text",
            "rating",
            "review_date",
            "bank_code",
            "bank_name",
            "source",
            "reply_content",
            "review_version"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"❌ Missing expected columns: {missing}")
            return False

        # Handle missing values
        df = self.handle_missing(df)

        # Clean text
        print("🧹 Cleaning review text...")
        df["clean_text"] = df["review_text"].astype(str).apply(self.clean_text)

        # Filter: remove empty or too-short reviews
        df = df[df["clean_text"].str.len() > 5]

        # Compute text length
        df["text_length"] = df["clean_text"].str.len()

        # Fix dates
        print("📅 Normalizing dates...")
        df["review_date"] = df["review_date"].apply(self.fix_date)

        # Remove duplicates
        before = len(df)
        df = df.drop_duplicates(subset=["review_id", "clean_text"])
        after = len(df)
        print(f"🧽 Removed {before - after} duplicate reviews.")

        # Sort by date
        df = df.sort_values(by="review_date").reset_index(drop=True)

        # Save processed file
        df.to_csv(self.output_path, index=False)
        print(f"📁 Processed dataset saved → {self.output_path}")

        self.df = df  # store processed DataFrame
        return True


# ----------------------------------------------------------------------
# MODULE ENTRY POINT
# ----------------------------------------------------------------------
def main():
    """Run preprocessing as a standalone script."""
    processor = ReviewPreprocessor()
    success = processor.process()  # returns True/False

    if success:
        print("\n✅ Preprocessing finished successfully!")
        return processor.df
    else:
        print("\n❌ Preprocessing failed.")
        return None


if __name__ == "__main__":
    main()
