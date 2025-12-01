"""
preprocessing.py

Enhanced Google Play Review Preprocessing Module
-------------------------------------------------

This module loads raw scraped Google Play Store reviews and applies
an advanced preprocessing pipeline while preserving dataset completeness.
It cleans text, normalizes dates, imputes missing ratings, safely handles
duplicates, and preserves very short or empty reviews. Additionally, it
ensures a target number of preprocessed rows per bank.

Key Features:
-------------
1. Clean review text:
   - Lowercase
   - Remove URLs and special characters
   - Normalize whitespace
   - Preserve very short text

2. Normalize dates:
   - Convert to ISO format (YYYY-MM-DD)
   - Replace invalid dates with dataset median date

3. Impute missing ratings:
   - Bank-level median, fallback to global median

4. Remove duplicates:
   - Only among real-text reviews
   - Placeholder or very short reviews are preserved

5. Bank-level sampling:
   - Ensures a target number of rows per bank (e.g., 400)
   - Preserves randomness with a fixed seed

6. Detailed summary:
   - Tracks total rows, duplicates, invalid dates, missing ratings,
     very short/empty reviews per bank, and final dataset size

Usage:
------
from preprocessing import ReviewPreprocessor

processor = ReviewPreprocessor(target_per_bank=400)
clean_df = processor.process()
"""

import os
import re
from datetime import datetime
import pandas as pd
from src.config import DATA_PATHS


class ReviewPreprocessor:
    """
    ReviewPreprocessor

    Cleans and preprocesses raw Google Play Store reviews, preserves very
    short or empty text, imputes missing ratings, removes duplicates among
    real-text reviews, and ensures a target number of preprocessed rows per bank.

    Attributes
    ----------
    raw_path : str
        Path to raw reviews CSV.
    output_path : str
        Path to save processed CSV.
    target_per_bank : int
        Target number of preprocessed rows per bank.
    stats : dict
        Tracks counts for cleaning operations, replacements, duplicates,
        very short/empty text, and final rows.
    """

    def __init__(self, target_per_bank: int = 400):
        """Initialize paths, target rows, and statistics dictionary."""
        self.raw_path = DATA_PATHS["raw_reviews"]
        self.output_path = DATA_PATHS["processed_reviews"]
        self.target_per_bank = target_per_bank
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        self.stats = {
            "initial_rows": 0,
            "invalid_dates_fixed": 0,
            "missing_ratings_imputed": 0,
            "duplicates_real_text": 0,
            "final_rows": 0,
            "very_short_empty_per_bank": {}
        }

    @staticmethod
    def clean_text(text):
        """Clean and normalize review text (preserves very short text)."""
        if not isinstance(text, str):
            return ""
        text = text.strip().lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"[^\w\s.,!?]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def normalize_date(self, date_val, median_date):
        """Convert date to ISO format, replace invalid dates with median date."""
        try:
            return pd.to_datetime(date_val).strftime("%Y-%m-%d")
        except:
            self.stats["invalid_dates_fixed"] += 1
            return median_date

    def impute_missing_ratings(self, df):
        """Impute missing ratings using bank-level median, fallback to global median."""
        bank_medians = df.groupby("bank")["rating"].median()

        def impute(row):
            if pd.isna(row["rating"]):
                self.stats["missing_ratings_imputed"] += 1
                return bank_medians.get(row["bank"], df["rating"].median())
            return row["rating"]

        df["rating"] = df.apply(impute, axis=1)
        return df

    def process(self):
        """Run full preprocessing pipeline with bank-level sampling and detailed summary."""
        if not os.path.exists(self.raw_path):
            print(f"❌ File missing: {self.raw_path}")
            return None

        df = pd.read_csv(self.raw_path)
        self.stats["initial_rows"] = len(df)

        # Clean text
        df["clean_text"] = df["review"].astype(str).apply(self.clean_text)

        # Normalize dates
        try:
            median_date = df["date"].dropna().pipe(pd.to_datetime).median().strftime("%Y-%m-%d")
        except:
            median_date = datetime.today().strftime("%Y-%m-%d")
        df["date"] = df["date"].apply(lambda x: self.normalize_date(x, median_date))

        # Impute missing ratings
        df = self.impute_missing_ratings(df)

        # Remove duplicates among real-text reviews
        df_no_text = df[df["clean_text"].str.len() <= 5]  # very short/empty
        df_with_text = df[df["clean_text"].str.len() > 5]

        before_real_text = len(df_with_text)
        df_with_text = df_with_text.drop_duplicates(subset=["review_id", "clean_text"])
        self.stats["duplicates_real_text"] = before_real_text - len(df_with_text)

        # Track very short/empty per bank
        for bank in df_no_text["bank"].unique():
            self.stats["very_short_empty_per_bank"][bank] = len(df_no_text[df_no_text["bank"] == bank])

        # Combine back
        df = pd.concat([df_with_text, df_no_text], ignore_index=True)

        # Bank-level sampling to ensure target rows per bank
        df_list = []
        for bank in df["bank"].unique():
            bank_df = df[df["bank"] == bank]
            if len(bank_df) > self.target_per_bank:
                bank_df = bank_df.sample(n=self.target_per_bank, random_state=42)
            df_list.append(bank_df)
        df = pd.concat(df_list, ignore_index=True)

        # Final columns
        df = df[["review", "clean_text", "rating", "date", "bank", "source"]]
        df.to_csv(self.output_path, index=False)
        self.stats["final_rows"] = len(df)

        self.print_summary()
        return df

    def print_summary(self):
        """Print detailed preprocessing statistics per bank."""
        print("\n" + "=" * 60)
        print("🧽 PREPROCESSING SUMMARY (VERY SHORT TEXT PRESERVED)")
        print("=" * 60)
        print(f"Total input rows: {self.stats['initial_rows']}")
        print(f"Invalid dates fixed: {self.stats['invalid_dates_fixed']}")
        print(f"Missing ratings imputed: {self.stats['missing_ratings_imputed']}")
        print(f"Duplicates removed (real text only): {self.stats['duplicates_real_text']}")
        print("Very short/empty reviews per bank:")
        for bank, count in self.stats["very_short_empty_per_bank"].items():
            print(f"  - {bank}: {count}")
        print(f"Target rows per bank: {self.target_per_bank}")
        print(f"Final dataset size: {self.stats['final_rows']}")
        print(f"📁 Saved to: {self.output_path}")
        print("=" * 60)


def main():
    """Run preprocessing as a standalone script."""
    processor = ReviewPreprocessor(target_per_bank=400)
    return processor.process()


if __name__ == "__main__":
    main()
