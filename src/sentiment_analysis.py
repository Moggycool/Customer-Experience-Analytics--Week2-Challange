"""
Sentiment Analysis Module with distilBERT and TextBlob comparison

This module provides sentiment analysis functionality for bank reviews.
It computes sentiment scores using distilBERT model and provides
aggregation by bank and rating.
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from transformers import pipeline
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import json
from datetime import datetime
import warnings
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


@dataclass
class SentimentMetrics:
    """Data class to store sentiment analysis metrics."""
    total_reviews: int = 0
    positive_count: int = 0
    neutral_count: int = 0
    negative_count: int = 0
    positive_percentage: float = 0.0
    neutral_percentage: float = 0.0
    negative_percentage: float = 0.0
    avg_sentiment_score: float = 0.0
    avg_positive_score: float = 0.0
    avg_negative_score: float = 0.0
    sentiment_std: float = 0.0
    confidence_threshold: float = 0.5

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            'total_reviews': self.total_reviews,
            'positive_reviews': self.positive_count,
            'neutral_reviews': self.neutral_count,
            'negative_reviews': self.negative_count,
            'positive_percentage': round(self.positive_percentage, 2),
            'neutral_percentage': round(self.neutral_percentage, 2),
            'negative_percentage': round(self.negative_percentage, 2),
            'avg_sentiment_score': round(self.avg_sentiment_score, 4),
            'avg_positive_score': round(self.avg_positive_score, 4),
            'avg_negative_score': round(self.avg_negative_score, 4),
            'sentiment_std': round(self.sentiment_std, 4),
            'confidence_threshold': self.confidence_threshold
        }


class SentimentAnalyzer:
    """
    Computes sentiment scores and labels using distilBERT and/or TextBlob.

    Attributes
    ----------
    data : pd.DataFrame
        DataFrame containing reviews.
    text_column : str
        Name of the column with text to analyze.
    method : str
        Sentiment analysis method: 'distilbert', 'textblob', or 'both'
    device : int
        Device for distilBERT model: -1 for CPU, 0 for GPU
    batch_size : int
        Batch size for distilBERT inference
    """

    def __init__(self,
                 data: pd.DataFrame,
                 text_column: str = "clean_text",
                 method: str = "distilbert",
                 device: int = -1,  # -1 for CPU, 0 for GPU
                 batch_size: int = 32,
                 confidence_threshold: float = 0.5):
        """
        Initialize the SentimentAnalyzer.

        Args:
            data: DataFrame containing reviews
            text_column: Column name containing text to analyze
            method: Sentiment analysis method ('distilbert', 'textblob', or 'both')
            device: Device for distilBERT (-1 for CPU, 0 for GPU)
            batch_size: Batch size for distilBERT inference
            confidence_threshold: Confidence threshold for distilBERT predictions
        """
        self.data = data.copy()
        self.text_column = text_column
        self.method = method.lower()
        self.device = device
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold

        # Initialize models based on method
        self.distilbert_pipeline = None
        self.textblob_available = True

        if self.method in ['distilbert', 'both']:
            try:
                logger.info("Loading distilBERT model...")
                self.distilbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=device,
                    batch_size=batch_size
                )
                logger.info("distilBERT model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load distilBERT model: {e}")
                if self.method == 'distilbert':
                    logger.warning("Falling back to TextBlob")
                    self.method = 'textblob'
                self.distilbert_pipeline = None

        # Validate input
        self._validate_input()

        logger.info(f"SentimentAnalyzer initialized:")
        logger.info(f"  - Method: {self.method}")
        logger.info(f"  - Text column: {self.text_column}")
        logger.info(f"  - Device: {'CPU' if device == -1 else 'GPU'}")
        logger.info(f"  - Confidence threshold: {confidence_threshold}")
        logger.info(f"  - Total reviews: {len(self.data)}")

    def _validate_input(self):
        """Validate input data and parameters."""
        if self.data.empty:
            raise ValueError("Input data is empty")

        if self.text_column not in self.data.columns:
            raise ValueError(
                f"Text column '{self.text_column}' not found in data")

        # Check if text column contains valid strings
        if not all(isinstance(x, str) for x in self.data[self.text_column].dropna().head()):
            logger.warning(
                "Text column contains non-string values, converting to string")
            self.data[self.text_column] = self.data[self.text_column].astype(
                str)

        # Validate method
        valid_methods = ['distilbert', 'textblob', 'both']
        if self.method not in valid_methods:
            raise ValueError(
                f"Method must be one of {valid_methods}, got {self.method}")

    def analyze_sentiment(self,
                          return_metrics: bool = False,
                          include_rating_aggregation: bool = True) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
        """
        Analyze sentiment for all reviews using specified method.

        Args:
            return_metrics: If True, return metrics along with data
            include_rating_aggregation: If True, include rating-based sentiment aggregation

        Returns:
            DataFrame with sentiment columns, optionally with metrics
        """
        logger.info("Starting sentiment analysis...")

        # Apply sentiment analysis based on method
        if self.method == 'distilbert':
            self._analyze_with_distilbert()
        elif self.method == 'textblob':
            self._analyze_with_textblob()
        elif self.method == 'both':
            self._analyze_with_both()

        # Calculate aggregated metrics
        overall_metrics = self._calculate_overall_metrics()

        if include_rating_aggregation:
            rating_metrics = self._aggregate_by_rating()
            bank_rating_metrics = self._aggregate_by_bank_and_rating()

            aggregated_metrics = {
                'overall': overall_metrics.to_dict(),
                'by_rating': rating_metrics,
                'by_bank_and_rating': bank_rating_metrics
            }
        else:
            aggregated_metrics = {
                'overall': overall_metrics.to_dict()
            }

        logger.info("Sentiment analysis completed")

        if return_metrics:
            return self.data, aggregated_metrics
        return self.data

    def _analyze_with_distilbert(self):
        """Analyze sentiment using distilBERT model."""
        logger.info("Analyzing sentiment with distilBERT...")

        def get_distilbert_sentiment(text: str) -> Tuple[str, float]:
            """Get sentiment label and score from distilBERT."""
            if not isinstance(text, str) or not text.strip():
                return "neutral", 0.0

            try:
                # Truncate text if too long for model
                max_length = 512
                if len(text) > max_length:
                    text = text[:max_length]

                result = self.distilbert_pipeline(text)[0]
                label = result['label'].lower()
                score = result['score']

                # Convert to our label format
                if label == 'positive':
                    return "positive", score
                elif label == 'negative':
                    return "negative", score
                else:
                    return "neutral", score
            except Exception as e:
                logger.warning(f"Error with distilBERT analysis: {e}")
                return "neutral", 0.0

        # Apply distilBERT analysis in batches for efficiency
        texts = self.data[self.text_column].tolist()
        results = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = [get_distilbert_sentiment(
                text) for text in batch_texts]
            results.extend(batch_results)

            if (i // self.batch_size) % 10 == 0:
                logger.info(
                    f"Processed {i + len(batch_texts)}/{len(texts)} reviews")

        # Extract labels and scores
        labels, scores = zip(*results)

        # Add to dataframe
        self.data['sentiment_label_distilbert'] = list(labels)
        self.data['sentiment_score_distilbert'] = list(scores)

        # Convert distilBERT scores to -1 to 1 scale
        def convert_score(label, score):
            if label == 'positive':
                return score
            elif label == 'negative':
                return -score
            else:
                return 0.0

        self.data['sentiment_score'] = self.data.apply(
            lambda row: convert_score(row['sentiment_label_distilbert'],
                                      row['sentiment_score_distilbert']),
            axis=1
        )
        self.data['sentiment_label'] = self.data['sentiment_label_distilbert']

        logger.info("distilBERT analysis completed")

    def _analyze_with_textblob(self):
        """Analyze sentiment using TextBlob."""
        logger.info("Analyzing sentiment with TextBlob...")

        def get_textblob_sentiment(text: str) -> Tuple[float, str]:
            """Get sentiment score and label from TextBlob."""
            if not isinstance(text, str) or not text.strip():
                return 0.0, "neutral"

            try:
                score = TextBlob(text).sentiment.polarity

                # Apply confidence threshold
                if score > self.confidence_threshold:
                    label = "positive"
                elif score < -self.confidence_threshold:
                    label = "negative"
                else:
                    label = "neutral"

                return score, label
            except Exception as e:
                logger.warning(f"Error with TextBlob analysis: {e}")
                return 0.0, "neutral"

        # Apply TextBlob analysis
        results = self.data[self.text_column].apply(get_textblob_sentiment)

        # Extract scores and labels
        self.data['sentiment_score_textblob'] = [r[0] for r in results]
        self.data['sentiment_label_textblob'] = [r[1] for r in results]
        self.data['sentiment_score'] = self.data['sentiment_score_textblob']
        self.data['sentiment_label'] = self.data['sentiment_label_textblob']

        logger.info("TextBlob analysis completed")

    def _analyze_with_both(self):
        """Analyze sentiment with both methods and compare."""
        logger.info("Analyzing sentiment with both distilBERT and TextBlob...")

        # Run both analyses
        self._analyze_with_distilbert()

        # Save distilBERT results
        distilbert_scores = self.data['sentiment_score'].copy()
        distilbert_labels = self.data['sentiment_label'].copy()

        # Run TextBlob analysis
        self._analyze_with_textblob()

        # Add comparison columns
        self.data['sentiment_score_distilbert'] = distilbert_scores
        self.data['sentiment_label_distilbert'] = distilbert_labels

        # Calculate agreement
        self.data['sentiment_agreement'] = (
            self.data['sentiment_label'] == self.data['sentiment_label_distilbert']
        ).astype(int)

        # Use distilBERT as primary if available
        self.data['sentiment_score'] = self.data['sentiment_score_distilbert']
        self.data['sentiment_label'] = self.data['sentiment_label_distilbert']

        # Log comparison statistics
        agreement_rate = self.data['sentiment_agreement'].mean() * 100
        logger.info(
            f"Sentiment agreement between methods: {agreement_rate:.2f}%")

    def _calculate_overall_metrics(self) -> SentimentMetrics:
        """Calculate overall sentiment metrics."""
        logger.info("Calculating overall sentiment metrics...")

        if 'sentiment_label' not in self.data.columns:
            raise ValueError(
                "Sentiment labels not found. Run analyze_sentiment() first.")

        # Get sentiment counts
        sentiment_counts = self.data['sentiment_label'].value_counts()

        metrics = SentimentMetrics(
            total_reviews=len(self.data),
            positive_count=sentiment_counts.get('positive', 0),
            neutral_count=sentiment_counts.get('neutral', 0),
            negative_count=sentiment_counts.get('negative', 0),
            confidence_threshold=self.confidence_threshold
        )

        # Calculate percentages
        if metrics.total_reviews > 0:
            metrics.positive_percentage = (
                metrics.positive_count / metrics.total_reviews) * 100
            metrics.neutral_percentage = (
                metrics.neutral_count / metrics.total_reviews) * 100
            metrics.negative_percentage = (
                metrics.negative_count / metrics.total_reviews) * 100

        # Calculate score statistics
        if 'sentiment_score' in self.data.columns:
            scores = self.data['sentiment_score'].dropna()
            if len(scores) > 0:
                metrics.avg_sentiment_score = scores.mean()
                metrics.sentiment_std = scores.std()

                # Calculate average scores by label
                positive_scores = self.data[self.data['sentiment_label']
                                            == 'positive']['sentiment_score']
                negative_scores = self.data[self.data['sentiment_label']
                                            == 'negative']['sentiment_score']

                if len(positive_scores) > 0:
                    metrics.avg_positive_score = positive_scores.mean()
                if len(negative_scores) > 0:
                    metrics.avg_negative_score = negative_scores.mean()

        return metrics

    def _aggregate_by_rating(self) -> Dict[str, Any]:
        """
        Aggregate sentiment metrics by rating.

        Returns:
            Dictionary with sentiment metrics aggregated by rating
        """
        logger.info("Aggregating sentiment by rating...")

        if 'rating' not in self.data.columns:
            logger.warning(
                "Rating column not found, skipping rating aggregation")
            return {}

        rating_aggregation = {}

        for rating in sorted(self.data['rating'].unique()):
            rating_df = self.data[self.data['rating'] == rating]

            if len(rating_df) == 0:
                continue

            sentiment_counts = rating_df['sentiment_label'].value_counts()

            rating_metrics = {
                'total_reviews': len(rating_df),
                'positive_count': sentiment_counts.get('positive', 0),
                'neutral_count': sentiment_counts.get('neutral', 0),
                'negative_count': sentiment_counts.get('negative', 0),
                'positive_percentage': (sentiment_counts.get('positive', 0) / len(rating_df)) * 100,
                'negative_percentage': (sentiment_counts.get('negative', 0) / len(rating_df)) * 100
            }

            # Calculate average sentiment score
            if 'sentiment_score' in rating_df.columns:
                rating_metrics['avg_sentiment_score'] = rating_df['sentiment_score'].mean(
                )
                rating_metrics['sentiment_std'] = rating_df['sentiment_score'].std()

            rating_aggregation[str(rating)] = rating_metrics

        return rating_aggregation

    def _aggregate_by_bank_and_rating(self) -> Dict[str, Any]:
        """
        Aggregate sentiment metrics by bank and rating.

        Returns:
            Dictionary with sentiment metrics aggregated by bank and rating
        """
        logger.info("Aggregating sentiment by bank and rating...")

        if 'bank_name' not in self.data.columns:
            logger.warning(
                "Bank name column not found, skipping bank-rating aggregation")
            return {}

        if 'rating' not in self.data.columns:
            logger.warning(
                "Rating column not found, skipping bank-rating aggregation")
            return {}

        bank_rating_aggregation = {}

        for bank in self.data['bank_name'].unique():
            bank_df = self.data[self.data['bank_name'] == bank]
            bank_rating_aggregation[bank] = {}

            for rating in sorted(bank_df['rating'].unique()):
                rating_df = bank_df[bank_df['rating'] == rating]

                if len(rating_df) == 0:
                    continue

                sentiment_counts = rating_df['sentiment_label'].value_counts()

                rating_metrics = {
                    'total_reviews': len(rating_df),
                    'positive_count': sentiment_counts.get('positive', 0),
                    'neutral_count': sentiment_counts.get('neutral', 0),
                    'negative_count': sentiment_counts.get('negative', 0),
                    'positive_percentage': (sentiment_counts.get('positive', 0) / len(rating_df)) * 100,
                    'negative_percentage': (sentiment_counts.get('negative', 0) / len(rating_df)) * 100
                }

                # Calculate average sentiment score
                if 'sentiment_score' in rating_df.columns:
                    rating_metrics['avg_sentiment_score'] = rating_df['sentiment_score'].mean(
                    )

                bank_rating_aggregation[bank][str(rating)] = rating_metrics

        return bank_rating_aggregation

    def get_sentiment_summary(self) -> pd.DataFrame:
        """
        Get comprehensive sentiment summary.

        Returns:
            DataFrame with sentiment summary statistics
        """
        if 'sentiment_label' not in self.data.columns:
            raise ValueError(
                "Sentiment labels not found. Run analyze_sentiment() first.")

        summary_data = []

        # Overall summary
        overall_metrics = self._calculate_overall_metrics()
        summary_data.append({
            'scope': 'overall',
            'category': 'all',
            'total_reviews': overall_metrics.total_reviews,
            'positive_percentage': overall_metrics.positive_percentage,
            'neutral_percentage': overall_metrics.neutral_percentage,
            'negative_percentage': overall_metrics.negative_percentage,
            'avg_sentiment_score': overall_metrics.avg_sentiment_score
        })

        # By rating summary
        rating_aggregation = self._aggregate_by_rating()
        for rating, metrics in rating_aggregation.items():
            summary_data.append({
                'scope': 'by_rating',
                'category': f'rating_{rating}',
                'total_reviews': metrics['total_reviews'],
                'positive_percentage': metrics['positive_percentage'],
                'negative_percentage': metrics['negative_percentage'],
                'avg_sentiment_score': metrics.get('avg_sentiment_score', 0)
            })

        # By bank summary (if available)
        if 'bank_name' in self.data.columns:
            for bank in self.data['bank_name'].unique():
                bank_df = self.data[self.data['bank_name'] == bank]
                sentiment_counts = bank_df['sentiment_label'].value_counts()

                summary_data.append({
                    'scope': 'by_bank',
                    'category': bank,
                    'total_reviews': len(bank_df),
                    'positive_percentage': (sentiment_counts.get('positive', 0) / len(bank_df)) * 100,
                    'negative_percentage': (sentiment_counts.get('negative', 0) / len(bank_df)) * 100,
                    'avg_sentiment_score': bank_df['sentiment_score'].mean() if 'sentiment_score' in bank_df.columns else 0
                })

        return pd.DataFrame(summary_data)

    def export_sentiment_report(self,
                                output_path: str,
                                include_detailed: bool = True):
        """
        Export comprehensive sentiment analysis report.

        Args:
            output_path: Path to save the report
            include_detailed: Whether to include detailed review-level data
        """
        logger.info(f"Exporting sentiment report to: {output_path}")

        # Calculate all metrics
        overall_metrics = self._calculate_overall_metrics()
        rating_aggregation = self._aggregate_by_rating()
        bank_rating_aggregation = self._aggregate_by_bank_and_rating()

        # Build report
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_method': self.method,
            'confidence_threshold': self.confidence_threshold,
            'summary': {
                'overall': overall_metrics.to_dict(),
                'by_rating': rating_aggregation,
                'by_bank_and_rating': bank_rating_aggregation
            },
            'method_comparison': {}
        }

        # Add method comparison if both methods were used
        if self.method == 'both' and 'sentiment_agreement' in self.data.columns:
            report['method_comparison'] = {
                'agreement_rate': float(self.data['sentiment_agreement'].mean()),
                'agreement_percentage': float(self.data['sentiment_agreement'].mean() * 100)
            }

        # Add detailed data if requested
        if include_detailed:
            detailed_data = self.data[[
                'review_id', 'bank_name', 'rating', 'clean_text',
                'sentiment_label', 'sentiment_score'
            ]].head(100).to_dict('records')  # Include first 100 reviews
            report['sample_reviews'] = detailed_data

        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            # json.dump(report, f, indent=2, ensure_ascii=False)
            json.dump(report, f, indent=2, ensure_ascii=False,
                      default=lambda o: int(o) if hasattr(o, 'item') else str(o))

        logger.info(f"Sentiment report saved to {output_path}")

        # Also save summary as CSV
        summary_df = self.get_sentiment_summary()
        summary_path = output_path.replace('.json', '_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Sentiment summary saved to {summary_path}")

    def save_analyzed_data(self, output_path: str):
        """
        Save the analyzed data with sentiment columns.

        Args:
            output_path: Path to save the CSV file
        """
        self.data.to_csv(output_path, index=False)
        logger.info(f"Analyzed data saved to {output_path}")


# Convenience functions
def compare_sentiment_methods(df: pd.DataFrame,
                              text_column: str = "clean_text",
                              sample_size: int = 1000) -> pd.DataFrame:
    """
    Compare sentiment analysis methods on a sample of data.

    Args:
        df: Input DataFrame
        text_column: Text column to analyze
        sample_size: Number of reviews to sample for comparison

    Returns:
        DataFrame with comparison results
    """
    logger.info(
        f"Comparing sentiment analysis methods on {sample_size} samples...")

    # Sample data if needed
    if len(df) > sample_size:
        sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    else:
        sample_df = df.copy()

    # Initialize analyzers
    distilbert_analyzer = SentimentAnalyzer(
        sample_df, text_column=text_column, method='distilbert'
    )
    textblob_analyzer = SentimentAnalyzer(
        sample_df, text_column=text_column, method='textblob'
    )

    # Run analyses
    distilbert_results, _ = distilbert_analyzer.analyze_sentiment(
        return_metrics=True)
    textblob_results, _ = textblob_analyzer.analyze_sentiment(
        return_metrics=True)

    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'review_id': sample_df['review_id'] if 'review_id' in sample_df.columns else sample_df.index,
        'text': sample_df[text_column],
        'distilbert_label': distilbert_results['sentiment_label'],
        'distilbert_score': distilbert_results['sentiment_score'],
        'textblob_label': textblob_results['sentiment_label'],
        'textblob_score': textblob_results['sentiment_score'],
        'agreement': distilbert_results['sentiment_label'] == textblob_results['sentiment_label']
    })

    # Calculate agreement statistics
    agreement_rate = comparison_df['agreement'].mean() * 100
    logger.info(f"Method agreement: {agreement_rate:.2f}%")

    return comparison_df


def run_sentiment_analysis_pipeline(
    input_file: str = "data/preprocessed/google_play_processed_reviews.csv",
    output_file: str = "data/preprocessed/sentiment_analyzed.csv",
    method: str = "distilbert",
    text_column: str = "clean_text",
    device: int = -1,
    batch_size: int = 32,
    confidence_threshold: float = 0.5,
    return_metrics: bool = True,
    export_report: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run the complete sentiment analysis pipeline on preprocessed data.

    This function:
    1. Loads the preprocessed reviews from data/preprocessed/google_play_processed_reviews.csv
    2. Performs sentiment analysis using the specified method
    3. Saves the results to data/preprocessed/sentiment_analyzed.csv
    4. Optionally exports a detailed report

    Args:
        input_file: Path to input CSV file (preprocessed reviews)
        output_file: Path to output CSV file (sentiment analyzed data)
        method: Sentiment analysis method ('distilbert', 'textblob', or 'both')
        text_column: Column containing text to analyze
        device: Device for distilBERT (-1 for CPU, 0 for GPU)
        batch_size: Batch size for distilBERT inference
        confidence_threshold: Confidence threshold for predictions
        return_metrics: If True, return metrics along with data
        export_report: If True, export detailed JSON report

    Returns:
        Tuple of (analyzed DataFrame, sentiment metrics)
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    # Load the preprocessed data
    logger.info(f"Loading preprocessed data from: {input_file}")
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            f"Please make sure you have run the preprocessing pipeline first."
        )

    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} reviews from {input_file}")

    # Check if required columns exist
    required_columns = [text_column, 'rating', 'bank_name']
    missing_columns = [
        col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing columns in input data: {missing_columns}")

    # Initialize and run sentiment analyzer
    logger.info(f"Initializing SentimentAnalyzer with method: {method}")
    analyzer = SentimentAnalyzer(
        df,
        text_column=text_column,
        method=method,
        device=device,
        batch_size=batch_size,
        confidence_threshold=confidence_threshold
    )

    # Perform sentiment analysis
    logger.info("Performing sentiment analysis...")
    if return_metrics:
        result_df, metrics = analyzer.analyze_sentiment(return_metrics=True)
    else:
        result_df = analyzer.analyze_sentiment(return_metrics=False)
        metrics = None

    # Save the analyzed data
    logger.info(f"Saving sentiment-analyzed data to: {output_file}")
    analyzer.save_analyzed_data(output_file)

    # Export detailed report if requested
    if export_report:
        report_file = output_file.replace('.csv', '_report.json')
        logger.info(f"Exporting detailed report to: {report_file}")
        analyzer.export_sentiment_report(report_file)

        # Also save summary statistics
        summary_df = analyzer.get_sentiment_summary()
        summary_file = output_file.replace('.csv', '_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved summary statistics to: {summary_file}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS COMPLETE")
    print("="*60)
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"Method used: {method}")
    print(f"Total reviews analyzed: {len(result_df):,}")

    if metrics and 'overall' in metrics:
        overall = metrics['overall']
        print(f"\nSentiment Distribution:")
        print(
            f"  Positive: {overall['positive_reviews']:,} ({overall['positive_percentage']}%)")
        print(
            f"  Neutral:  {overall['neutral_reviews']:,} ({overall['neutral_percentage']}%)")
        print(
            f"  Negative: {overall['negative_reviews']:,} ({overall['negative_percentage']}%)")
        print(
            f"  Average sentiment score: {overall['avg_sentiment_score']:.3f}")

    print(f"\n✅ Analysis complete! Files saved to data/preprocessed/")

    if return_metrics:
        return result_df, metrics
    return result_df


# Main execution function
def main():
    """Main function for command-line usage."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Sentiment Analysis Pipeline for Bank Reviews")
    parser.add_argument("--input", default="data/preprocessed/google_play_processed_reviews.csv",
                        help="Input CSV file path (preprocessed reviews)")
    parser.add_argument("--output", default="data/preprocessed/sentiment_analyzed.csv",
                        help="Output CSV file path (sentiment analyzed data)")
    parser.add_argument("--text-column", default="clean_text",
                        help="Text column name")
    parser.add_argument("--method", default="distilbert",
                        choices=["distilbert", "textblob", "both"],
                        help="Sentiment analysis method")
    parser.add_argument("--device", type=int, default=-1, choices=[-1, 0],
                        help="Device for distilBERT (-1 for CPU, 0 for GPU)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for distilBERT")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold")
    parser.add_argument("--no-report", action="store_false", dest="export_report",
                        help="Don't export detailed JSON report")

    args = parser.parse_args()

    try:
        # Run the complete sentiment analysis pipeline
        result_df, metrics = run_sentiment_analysis_pipeline(
            input_file=args.input,
            output_file=args.output,
            method=args.method,
            text_column=args.text_column,
            device=args.device,
            batch_size=args.batch_size,
            confidence_threshold=args.threshold,
            return_metrics=True,
            export_report=args.export_report
        )

    except Exception as e:
        logger.error(f"Error in sentiment analysis pipeline: {e}")
        raise


# Quick execution function for immediate use
def quick_sentiment_analysis():
    """
    Quick function to run sentiment analysis with default settings.
    This is useful for immediate execution in Jupyter notebooks or scripts.
    """
    print("Running quick sentiment analysis pipeline...")
    return run_sentiment_analysis_pipeline()


if __name__ == "__main__":
    main()
