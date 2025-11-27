""" Pipeline Module """
import os
import logging
import pandas as pd
from src.sentiment_analysis import SentimentAnalyzer
from src.thematic_analysis import ThematicAnalyzer

# --------------------------
# Configure logging
# --------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class ReviewPipeline:
    """
    Runs preprocessing, sentiment analysis, thematic analysis, and saves results.
    """

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

    def run_pipeline(self):
        """Runs the full pipeline"""
        try:
            # Step 1: Load preprocessed reviews
            logging.info(f"Loading data from {self.input_path}...")
            df = pd.read_csv(self.input_path)
            logging.info(f"Loaded {len(df)} rows.")

            # Step 2: Sentiment Analysis
            logging.info("Starting sentiment analysis...")
            sentiment_analyzer = SentimentAnalyzer(
                df, text_column="clean_text")
            df = sentiment_analyzer.analyze_sentiment()
            logging.info("Sentiment analysis completed.")

            # Step 3: Thematic Analysis
            logging.info("Starting thematic analysis...")
            thematic_analyzer = ThematicAnalyzer(df, text_column="clean_text")
            df = thematic_analyzer.extract_keywords(top_n=10)
            df = thematic_analyzer.assign_themes()
            logging.info("Thematic analysis completed.")

            # Step 4: Save results
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            df.to_csv(self.output_path, index=False)
            logging.info(
                f"Pipeline complete. Output saved to {self.output_path}")
            print(f"Pipeline complete. Output saved to {self.output_path}")

            return df

        except FileNotFoundError as e:
            logging.error(f"Input file not found: {e}")
            raise
        except pd.errors.ParserError as e:
            logging.error(f"Error parsing CSV: {e}")
            raise
        except Exception as e:
            logging.exception(f"Unexpected error occurred: {e}")
            raise


# Example usage:
if __name__ == "__main__":
    pipeline = ReviewPipeline(
        input_path="data/preprocessed/google_play_processed_reviews.csv",
        output_path="data/sentiment_preprocessed.csv"
    )
    pipeline.run_pipeline()
