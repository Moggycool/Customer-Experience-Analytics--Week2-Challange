""" Pipeline Module"""
import os
import pandas as pd
from src.sentiment_analysis import SentimentAnalyzer
from src.thematic_analysis import ThematicAnalyzer


class ReviewPipeline:
    """
    Runs preprocessing, sentiment analysis, thematic analysis and saves results.
    """

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path

    def run_pipeline(self):
        "runs the pipeline"
        # Step 1: Load preprocessed reviews
        df = pd.read_csv(self.input_path)

        # Step 2: Sentiment Analysis
        sentiment_analyzer = SentimentAnalyzer(df)
        df = sentiment_analyzer.analyze_sentiment()

        # Step 3: Thematic Analysis
        thematic_analyzer = ThematicAnalyzer(df)
        df = thematic_analyzer.extract_keywords(top_n=10)
        df = thematic_analyzer.assign_themes()

        # Step 4: Save results
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        df.to_csv(self.output_path, index=False)
        print(f"Pipeline complete. Output saved to {self.output_path}")
        return df


# Example usage:
if __name__ == "__main__":
    pipeline = ReviewPipeline(
        input_path="data/preprocessed/google_play_processed_reviews.csv",
        output_path="data/sentiment_preprocessed.csv"
    )
    pipeline.run_pipeline()
