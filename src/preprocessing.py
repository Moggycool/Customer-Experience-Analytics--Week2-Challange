"""
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
        self.df = pd.DataFrame()

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    # ----------------------------------------------------------------------
    # TEXT CLEANING HELPERS
    # ----------------------------------------------------------------------
    @staticmethod
    def is_english_only(text: str) -> bool:
        """
        Check if text contains ONLY English characters, digits, and basic punctuation.
        Returns True if text is purely English, False otherwise.
        """
        if not isinstance(text, str) or not text.strip():
            return False

        # Pattern for English-only text (allows basic punctuation and spaces)
        english_only_pattern = re.compile(r'^[A-Za-z0-9\s.,!?;:\'"()\-]+$')

        return bool(english_only_pattern.fullmatch(text.strip()))

    @staticmethod
    def contains_non_english(text: str) -> bool:
        """Check if text contains non-English characters."""
        if not isinstance(text, str):
            return False

        # Pattern to detect non-English alphabets and emojis
        # This includes: Cyrillic, Arabic, Hebrew, Chinese, Japanese, Korean, etc.
        non_english_pattern = re.compile(
            # Arabic, Chinese
            r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u4E00-\u9FFF\u3400-\u4DBF'
            r'\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF'  # Japanese, Korean
            r'\u0400-\u04FF\u0500-\u052F'  # Cyrillic
            r'\u0590-\u05FF\uFB1D-\uFB4F'  # Hebrew
            r'\u0900-\u097F\u0980-\u09FF'  # Devanagari, Bengali
            r'\u0E00-\u0E7F'  # Thai
            r'\u0C00-\u0C7F'  # Telugu
            r'\u0B00-\u0B7F'  # Oriya
            r'\u0A00-\u0A7F'  # Gurmukhi
            r'\u1F600-\u1F64F\u1F300-\u1F5FF\u1F680-\u1F6FF\u1F1E0-\u1F1FF'  # Emojis
            r'\u2600-\u26FF\u2700-\u27BF]'  # Miscellaneous symbols
        )

        return bool(non_english_pattern.search(text))

    @staticmethod
    def extract_english_from_mixed(text: str) -> str:
        """
        Extract English text from mixed language content.
        Removes non-English words, emojis, and foreign alphabets while keeping English text.

        Examples:
        - "Hello مرحبا world" → "Hello world"
        - "Good app! Emogi" → "Good app!"
        - "Nice app شكرا" → "Nice app"
        - "银行bank应用app" → "bank app"
        """
        if not isinstance(text, str):
            return ""

        text = text.strip()
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # First, extract English words (with numbers and basic punctuation)
        # This pattern captures:
        # - Words starting with English letters (allowing apostrophes in contractions)
        # - Numbers
        # - Basic punctuation at word boundaries
        english_word_pattern = r'\b(?:[A-Za-z][A-Za-z\']*|[0-9]+)[.,!?;:\'\"()\-]*\b'

        # Find all English words and number sequences
        english_words = re.findall(english_word_pattern, text)

        # Join the extracted words
        cleaned_text = ' '.join(english_words)

        # Remove any leftover non-English characters that might have been captured
        cleaned_text = re.sub(
            r'[^A-Za-z0-9\s.,!?;:\'\"()\-]', ' ', cleaned_text)

        # Normalize spacing and clean up
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # Lowercase for consistency
        cleaned_text = cleaned_text.lower()

        return cleaned_text

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean review text:
        - Remove URLs
        - For English-only text: keep as is (just normalize)
        - For mixed text: extract only English content
        - Normalize spacing
        """
        if not isinstance(text, str):
            return ""

        text = text.strip()
        if not text:
            return ""

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Check if text is already English-only
        if ReviewPreprocessor.is_english_only(text):
            # Just normalize spacing and lowercase
            text = re.sub(r'\s+', ' ', text).strip().lower()
            return text
        else:
            # Extract English from mixed content
            return ReviewPreprocessor.extract_english_from_mixed(text)

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
        print("[INFO] Checking and handling missing values per bank...\n")

        for bank in banks:
            bank_df = df[df['bank_name'] == bank]
            total = len(bank_df)

            for col in ['reply_content', 'review_version', 'rating']:
                missing_count = bank_df[col].isna().sum()
                missing_pct = (missing_count / total * 100) if total else 0

                print(f"Bank: {bank} ({total} reviews)")
                print(
                    f"  - {col}: {missing_count} missing ({missing_pct:.2f}%)")

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
        """Process reviews: clean, filter, handle missing, compute length, save dataset."""
        if not os.path.exists(self.raw_path):
            print(f"[INFO] Raw reviews file not found: {self.raw_path}")
            return False

        print(f"[INFO] Loading raw reviews from {self.raw_path}...")
        df = pd.read_csv(self.raw_path)

        if df.empty:
            print("[INFO] ERROR: Raw reviews file is empty.")
            return False

        required_cols = [
            "review_id", "review_text", "rating", "review_date",
            "bank_code", "bank_name", "source",
            "reply_content", "review_version"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"[INFO] Missing expected columns: {missing}")
            return False

        # Handle missing values
        df = self.handle_missing(df)

        # ------------------------------------------------------------------
        # 1. REMOVE ROWS WITH DUPLICATE USER_NAME (if column exists)
        # ------------------------------------------------------------------
        if 'user_name' in df.columns:
            before_dup = len(df)
            df = df.drop_duplicates(subset=['user_name'], keep='first')
            after_dup = len(df)
            print(
                f"[INFO] Removed {before_dup - after_dup} rows with duplicate user_name.")

        # ------------------------------------------------------------------
        # 2. UPDATE review_text COLUMN: Clean ALL reviews (English-only AND mixed)
        # ------------------------------------------------------------------
        print("[INFO] Cleaning all review_text content...")
        before_clean = len(df)

        # Store original text for reference
        df['original_text'] = df['review_text'].copy()

        # Clean ALL review_text entries
        df['review_text'] = df['review_text'].astype(
            str).apply(self.clean_text)

        # ------------------------------------------------------------------
        # 3. SEPARATE HANDLING FOR ENGLISH-ONLY vs MIXED CONTENT
        # ------------------------------------------------------------------
        print("[INFO] Classifying reviews by language content...")

        # Create a copy of original text for checking
        original_texts = df['original_text'].astype(str)

        # Classify reviews
        df['is_english_only'] = original_texts.apply(self.is_english_only)
        df['has_mixed_content'] = original_texts.apply(
            self.contains_non_english)

        # Count statistics
        english_only_count = df['is_english_only'].sum()
        mixed_content_count = df['has_mixed_content'].sum()
        cleaned_english_count = len(df[df['review_text'].str.strip() != ''])

        print(f"[INFO] Language Statistics:")
        print(f"  - English-only reviews: {english_only_count}")
        print(f"  - Mixed-content reviews: {mixed_content_count}")
        print(
            f"  - Reviews with English content after cleaning: {cleaned_english_count}")

        # Remove temporary columns
        df = df.drop(columns=['is_english_only',
                     'has_mixed_content'], errors='ignore')

        # ------------------------------------------------------------------
        # REMOVE EMPTY / IRRELEVANT REVIEWS
        # ------------------------------------------------------------------
        print("[INFO] Removing empty or irrelevant reviews...")
        before_filter = len(df)

        # Remove empty cleaned text
        df = df[df["review_text"].str.strip() != ""]

        # Remove extremely short / meaningless text (e.g., ".", "??", "ok")
        df = df[df["review_text"].str.replace(
            r"[.,!?;:'\"()\- ]", "", regex=True).str.len() > 2]

        # Minimum meaningful length
        df = df[df["review_text"].str.len() > 3]

        after_filter = len(df)
        print(f"  Removed {before_filter - after_filter} empty/short reviews.")

        # ------------------------------------------------------------------
        # Create clean_text column (same as review_text after cleaning)
        # ------------------------------------------------------------------
        df["clean_text"] = df["review_text"]

        # ------------------------------------------------------------------
        # Compute length based on cleaned text
        df["text_length"] = df["clean_text"].str.len()

        # Fix dates
        print("[INFO] Normalizing dates...")
        df["review_date"] = df["review_date"].apply(self.fix_date)

        # Remove duplicates based on cleaned content
        before = len(df)
        df = df.drop_duplicates(subset=["review_id", "clean_text"])
        after = len(df)
        print(f"[INFO] Removed {before - after} duplicate reviews.")

        # Sort by date and reset index
        df = df.sort_values(by="review_date").reset_index(drop=True)

        # Drop the original_text column if no longer needed
        df = df.drop(columns=['original_text'], errors='ignore')

        # Save processed output
        df.to_csv(self.output_path, index=False)
        print(f"[INFO] Processed dataset saved -> {self.output_path}")

        print("\n[INFO] Cleaning Summary:")
        print(f"   - Total reviews processed: {before_clean}")
        print(f"   - Reviews kept after cleaning: {len(df)}")
        print(f"   - Reviews removed: {before_clean - len(df)}")

        self.df = df
        return True


def main():
    """"Main module"""
    processor = ReviewPreprocessor()
    success = processor.process()

    if success:
        print("\n[INFO] Preprocessing finished successfully!")
        print(f"[INFO] Final dataset shape: {processor.df.shape}")

        # Show sample of cleaned reviews
        print("\n[INFO] Sample of cleaned reviews:")
        sample_df = processor.df[['review_text',
                                  'clean_text', 'text_length']].head(5)
        for idx, row in sample_df.iterrows():
            print(f"  Review {idx+1}:")
            print(f"    Length: {row['text_length']} chars")
            print(f"    Text: {row['review_text'][:100]}..." if len(
                row['review_text']) > 100 else f"    Text: {row['review_text']}")
            print()

        return processor.df
    else:
        print("\n[INFO] Preprocessing failed.")
        return None


if __name__ == "__main__":
    main()
