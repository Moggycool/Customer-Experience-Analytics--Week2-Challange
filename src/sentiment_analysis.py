"""Sentiment Analysis Module"""
import pandas as pd
from textblob import TextBlob  # simple sentiment analysis alternative
# Optional: for transformer-based sentiment, import transformers


class SentimentAnalyzer:
    """
    Computes sentiment for each review.
    """

    def __init__(self, data: pd.DataFrame, text_column: str = "clean_text"):
        self.data = data.copy()
        self.text_column = text_column

    def analyze_sentiment(self):
        """
        Adds 'sentiment_label' and 'sentiment_score' columns to the DataFrame.
        """
        def get_sentiment(text):
            if not isinstance(text, str) or len(text.strip()) == 0:
                return 0.0, "neutral"
            score = TextBlob(text).sentiment.polarity
            if score > 0.1:
                label = "positive"
            elif score < -0.1:
                label = "negative"
            else:
                label = "neutral"
            return score, label

        self.data[["sentiment_score", "sentiment_label"]] = self.data[self.text_column].apply(
            lambda x: pd.Series(get_sentiment(x))
        )
        return self.data
