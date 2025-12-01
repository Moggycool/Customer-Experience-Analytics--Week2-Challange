"""
scraping.py

Google Play Store Review Scraper Module (Enhanced for Preprocessing Pipeline)
-----------------------------------------------------------------------------

This module collects app metadata and user reviews from Google Play Store
for multiple banks, scraping more reviews than the target to account for
duplicates, empty, or very short reviews. It ensures sufficient data
for the preprocessing pipeline to produce a consistent number of clean rows.

Key Features:
-------------
1. Scrapes app metadata (title, score, installs, ratings, etc.)
2. Scrapes reviews with retries and country fallbacks
3. Targets more reviews than needed (e.g., 600 per bank)
4. Saves raw reviews as CSV for preprocessing
"""

import os
import time
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from google_play_scraper import app, reviews, Sort
from src.config import APP_IDS, BANK_NAMES, SCRAPING_CONFIG, DATA_PATHS


class PlayStoreScraper:
    """Scraper for app metadata and reviews from Google Play."""

    def __init__(self):
        self.app_ids = APP_IDS
        self.bank_names = BANK_NAMES
        self.reviews_per_bank = SCRAPING_CONFIG.get("reviews_per_bank", 600)
        self.lang = SCRAPING_CONFIG.get("lang", "en")
        self.country_fallback = SCRAPING_CONFIG.get("country_fallback", ["et", "us", "gb"])
        self.max_retries = SCRAPING_CONFIG.get("max_retries", 3)
        self.scraping_delay = SCRAPING_CONFIG.get("scraping_delay", 2)

        # Ensure output directories exist
        for path in [DATA_PATHS["raw_reviews"], DATA_PATHS["processed_reviews"], DATA_PATHS["app_info"]]:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def get_app_info(self, app_id):
        """Fetch app metadata, trying multiple countries as fallback."""
        for country in self.country_fallback:
            try:
                result = app(app_id, lang=self.lang, country=country)
                return {
                    "app_id": app_id,
                    "title": result.get("title", "N/A"),
                    "score": result.get("score", 0),
                    "ratings": result.get("ratings", 0),
                    "reviews": result.get("reviews", 0),
                    "installs": result.get("installs", "N/A"),
                    "country": country,
                }
            except Exception as e:
                print(f"⚠️ App {app_id} not found in {country}: {e}")
        print(f"❌ App {app_id} not found in any fallback country.")
        return None

    def scrape_reviews(self, app_id, count):
        """Scrape reviews for a given app with retry and fallback mechanism."""
        for attempt in range(self.max_retries):
            for country in self.country_fallback:
                try:
                    result, _ = reviews(
                        app_id,
                        lang=self.lang,
                        country=country,
                        sort=Sort.NEWEST,
                        count=count,
                        filter_score_with=None,
                    )
                    if result:
                        print(f"✅ Scraped {len(result)} reviews for {app_id} in {country}")
                        return result
                except Exception as e:
                    print(f"⚠️ Attempt {attempt+1} failed for {app_id} in {country}: {e}")
                    time.sleep(4)
        print(f"❌ Failed to scrape reviews for {app_id} after {self.max_retries} attempts")
        return []

    def process_reviews(self, reviews_data, bank_code):
        """Convert raw JSON reviews into structured dictionaries."""
        processed = []
        for review in reviews_data:
            processed.append({
                "review_id": review.get("reviewId", ""),
                "review": review.get("content", ""),
                "rating": review.get("score", 0),
                "date": review.get("at", datetime.now()).strftime("%Y-%m-%d"),
                "user_name": review.get("userName", "Anonymous"),
                "thumbs_up": review.get("thumbsUpCount", 0),
                "reply_content": review.get("replyContent", ""),
                "bank": self.bank_names.get(bank_code, "Unknown"),
                "source": "Google Play",
                "review_version": review.get("reviewCreatedVersion", "N/A"),
            })
        return processed

    def scrape_all_banks(self):
        """Scrape app info and reviews for all banks, save raw CSVs."""
        all_reviews = []
        app_info_list = []

        print("\n" + "="*60)
        print("🚀 Starting Google Play Store Review Scraper")
        print("="*60)

        # Fetch app info
        for bank_code, app_id in self.app_ids.items():
            info = self.get_app_info(app_id)
            if info:
                info["bank_code"] = bank_code
                info["bank_name"] = self.bank_names.get(bank_code, "Unknown")
                app_info_list.append(info)
            time.sleep(self.scraping_delay)

        if app_info_list:
            app_info_df = pd.DataFrame(app_info_list)
            app_info_df.to_csv(DATA_PATHS["app_info"], index=False)
            print(f"📁 Saved app info → {DATA_PATHS['app_info']}")

        # Scrape reviews
        for bank_code, app_id in tqdm(self.app_ids.items(), desc="Banks"):
            reviews_data = self.scrape_reviews(app_id, self.reviews_per_bank)
            if reviews_data:
                processed = self.process_reviews(reviews_data, bank_code)
                all_reviews.extend(processed)
                print(f"Collected {len(processed)} reviews for {self.bank_names[bank_code]}")
            else:
                print(f"⚠️ No reviews collected for {self.bank_names[bank_code]}")
            time.sleep(self.scraping_delay)

        # Save raw reviews CSV
        if all_reviews:
            df = pd.DataFrame(all_reviews)
            df.to_csv(DATA_PATHS["raw_reviews"], index=False)
            print("\n" + "="*60)
            print("✅ Scraping Complete!")
            print("="*60)
            print(f"Total reviews collected: {len(df)}")
            for bank_code in self.bank_names.keys():
                count = len(df[df["bank"] == self.bank_names[bank_code]])
                print(f"  {self.bank_names[bank_code]}: {count} reviews")
            print(f"📁 Raw reviews saved → {DATA_PATHS['raw_reviews']}")
            return df
        else:
            print("❌ No reviews were collected.")
            return pd.DataFrame()

def main():
    """Run scraper end-to-end."""
    scraper = PlayStoreScraper()
    df = scraper.scrape_all_banks()
    return df

if __name__ == "__main__":
    main()
