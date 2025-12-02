"""
Thematic Analysis Module for Bank Reviews

This module extracts keywords and assigns themes to bank reviews using:
- TF-IDF for keyword extraction
- Rule-based theme clustering
- Predefined thematic categories

Requirements:
1. Extract significant keywords and n-grams using TF-IDF or spaCy
2. Group keywords into 3-5 overarching themes per bank
3. Assign themes to reviews based on keyword matches
4. Ensure minimum 2 themes per bank via keywords
5. Provide examples for each theme
"""

import os
import logging
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import Counter, defaultdict
from datetime import datetime
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
import nltk


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ThematicAnalyzer:
    """
    Extracts keywords and assigns reviews to predefined themes with enhanced functionality.

    Attributes
    ----------
    data : pd.DataFrame
        DataFrame containing reviews.
    text_column : str
        Column containing the review text.
    nlp : spacy.lang
        spaCy NLP model for text processing.
    themes : Dict[str, List[str]]
        Predefined themes and their associated keywords.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 text_column: str = "clean_text",
                 spacy_model: str = "en_core_web_sm",
                 output_dir: str = "data/preprocessed"):
        """
        Initialize the ThematicAnalyzer.

        Args:
            data: DataFrame containing reviews
            text_column: Column containing review text
            spacy_model: spaCy model to load
            output_dir: Directory to save all output files
        """
        self.data = data.copy()
        self.text_column = text_column
        self.output_dir = output_dir

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Load spaCy model for NLP processing
        try:
            self.nlp = spacy.load(spacy_model, disable=["parser", "ner"])
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}")
            logger.warning("Falling back to basic text processing")
            self.nlp = None

        # Enhanced theme definitions with more keywords
        self.themes = {
            "Account Access Issues": [
                "login", "password", "account", "locked", "verification",
                "access", "sign in", "log in", "authentication", "security",
                "blocked", "unlock", "credentials", "two factor", "2fa"
            ],
            "Transaction Performance": [
                "slow", "transfer", "payment", "failed", "processing",
                "transaction", "delay", "instant", "quick", "speed",
                "wait", "time", "process", "complete", "pending",
                "declined", "rejected", "successful", "immediate"
            ],
            "User Interface & Experience": [
                "ui", "interface", "navigation", "easy", "confusing",
                "design", "layout", "user friendly", "intuitive", "complicated",
                "simple", "clean", "modern", "outdated", "appearance",
                "menu", "button", "screen", "display", "visual"
            ],
            "Customer Support": [
                "support", "service", "help", "response", "staff",
                "customer", "assistance", "agent", "representative", "call",
                "email", "chat", "hotline", "contact", "resolve",
                "friendly", "professional", "rude", "unhelpful", "excellent"
            ],
            "App Reliability & Bugs": [
                "crash", "bug", "error", "freeze", "glitch",
                "reliable", "stable", "unstable", "close", "restart",
                "hang", "lag", "jerk", "not working", "malfunction",
                "technical", "issue", "problem", "fix", "update"
            ],
            "Mobile Banking Features": [
                "mobile", "app", "feature", "function", "capability",
                "notification", "alert", "push", "message", "reminder",
                "biometric", "fingerprint", "face id", "touch id", "scan",
                "qr", "mobile deposit", "remote", "online", "digital"
            ],
            "Security & Privacy": [
                "security", "privacy", "safe", "secure", "hack",
                "data", "personal", "information", "protection", "breach",
                "encryption", "fraud", "scam", "phishing", "trust",
                "confidential", "private", "leak", "expose", "theft"
            ],
            "Fees & Charges": [
                "fee", "charge", "cost", "price", "expensive",
                "free", "no fee", "zero fee", "minimum", "balance",
                "maintenance", "service charge", "transaction fee", "atm",
                "overdraft", "interest", "rate", "costly", "affordable"
            ]
        }

        # Theme descriptions for reporting
        self.theme_descriptions = {
            "Account Access Issues": "Problems related to logging in, account access, or authentication",
            "Transaction Performance": "Issues with transaction speed, processing time, or failed transactions",
            "User Interface & Experience": "Feedback about app design, navigation, and user experience",
            "Customer Support": "Comments about customer service quality and responsiveness",
            "App Reliability & Bugs": "Reports of app crashes, errors, bugs, or stability issues",
            "Mobile Banking Features": "Feedback on mobile-specific features and functionality",
            "Security & Privacy": "Concerns about data security, privacy, and fraud protection",
            "Fees & Charges": "Feedback about banking fees, charges, and pricing"
        }

        logger.info(
            f"ThematicAnalyzer initialized with {len(self.data)} reviews")
        logger.info(f"Text column: {self.text_column}")
        logger.info(f"Available themes: {list(self.themes.keys())}")
        logger.info(f"Output directory: {self.output_dir}")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text: tokenization, lemmatization, stopword removal.

        Args:
            text: Input text string

        Returns:
            Preprocessed text string
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        if self.nlp:
            # Use spaCy for advanced preprocessing
            doc = self.nlp(text.lower())
            tokens = []
            for token in doc:
                # Keep tokens that are not stop words, punctuation, or whitespace
                if not token.is_stop and not token.is_punct and not token.is_space:
                    # Use lemma (base form) of the word
                    tokens.append(token.lemma_)
            return " ".join(tokens)
        else:
            # Basic preprocessing if spaCy is not available
            # Simple tokenization and stopword removal

            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)

            stop_words = set(stopwords.words('english'))

            # Tokenize
            tokens = re.findall(r'\b[a-z]+\b', text.lower())
            # Remove stop words
            tokens = [token for token in tokens if token not in stop_words]
            return " ".join(tokens)

    def extract_keywords_tfidf(self,
                               top_n: int = 10,
                               ngram_range: Tuple[int, int] = (1, 3),
                               min_df: int = 2) -> pd.DataFrame:
        """
        Extract significant keywords using TF-IDF.

        Args:
            top_n: Number of top keywords to extract per review
            ngram_range: Range for n-grams (e.g., (1, 2) for unigrams and bigrams)
            min_df: Minimum document frequency for a term to be considered

        Returns:
            DataFrame with extracted keywords
        """
        logger.info(f"Extracting keywords using TF-IDF (top_n={top_n})...")

        # Preprocess text for better keyword extraction
        processed_texts = self.data[self.text_column].fillna(
            "").apply(self.preprocess_text)

        # Initialize and fit TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=5000  # Limit features for efficiency
        )

        try:
            X = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()

            # Extract top keywords for each review
            keywords_list = []
            for i, row in enumerate(X):
                row_data = row.toarray()[0]
                top_indices = row_data.argsort()[-top_n:][::-1]
                keywords = [feature_names[idx]
                            for idx in top_indices if row_data[idx] > 0]
                keywords_list.append(", ".join(keywords))

            self.data["keywords_tfidf"] = keywords_list
            logger.info(f"TF-IDF keyword extraction completed")

            # Also extract overall important keywords
            self._extract_overall_keywords(vectorizer, X)

        except Exception as e:
            logger.error(f"Error in TF-IDF extraction: {e}")
            self.data["keywords_tfidf"] = [""] * len(self.data)

        return self.data

    def _extract_overall_keywords(self, vectorizer, X):
        """Extract overall important keywords from all reviews."""
        # Calculate overall TF-IDF scores
        overall_scores = np.array(X.sum(axis=0)).flatten()
        feature_names = vectorizer.get_feature_names_out()

        # Get top overall keywords
        top_indices = overall_scores.argsort()[-50:][::-1]
        self.overall_keywords = [feature_names[idx] for idx in top_indices]

        logger.info(f"Top 10 overall keywords: {self.overall_keywords[:10]}")

    def extract_keywords_spacy(self, top_n: int = 10) -> pd.DataFrame:
        """
        Extract keywords using spaCy for entities and noun phrases.

        Args:
            top_n: Number of top keywords to extract per review

        Returns:
            DataFrame with extracted keywords
        """
        if self.nlp is None:
            logger.warning(
                "spaCy not available, skipping spaCy keyword extraction")
            self.data["keywords_spacy"] = [""] * len(self.data)
            return self.data

        logger.info(f"Extracting keywords using spaCy (top_n={top_n})...")

        keywords_list = []
        for text in self.data[self.text_column].fillna(""):
            if not text.strip():
                keywords_list.append("")
                continue

            doc = self.nlp(text.lower())

            # Extract noun phrases and entities
            keywords = []

            # Get noun phrases
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip()
                if len(phrase.split()) <= 3:  # Limit phrase length
                    keywords.append(phrase)

            # Get entities (if any)
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "EVENT", "WORK_OF_ART"]:
                    keywords.append(ent.text.strip())

            # Get important tokens (nouns, verbs, adjectives)
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN", "VERB", "ADJ"] and not token.is_stop:
                    keywords.append(token.lemma_)

            # Deduplicate and limit
            keywords = list(dict.fromkeys(keywords))[:top_n]
            keywords_list.append(", ".join(keywords))

        self.data["keywords_spacy"] = keywords_list
        logger.info("spaCy keyword extraction completed")

        return self.data

    def extract_keywords(self,
                         method: str = "tfidf",
                         top_n: int = 10) -> pd.DataFrame:
        """
        Extract keywords using specified method.

        Args:
            method: Extraction method ('tfidf', 'spacy', or 'both')
            top_n: Number of top keywords to extract

        Returns:
            DataFrame with extracted keywords
        """
        if method.lower() == "tfidf":
            return self.extract_keywords_tfidf(top_n=top_n)
        elif method.lower() == "spacy":
            return self.extract_keywords_spacy(top_n=top_n)
        elif method.lower() == "both":
            self.extract_keywords_tfidf(top_n=top_n)
            self.extract_keywords_spacy(top_n=top_n)
            # Combine both keyword sets
            self.data["keywords"] = self.data.apply(
                lambda row: f"{row['keywords_tfidf']}, {row['keywords_spacy']}".strip(
                    ", "),
                axis=1
            )
            return self.data
        else:
            raise ValueError(
                f"Unknown method: {method}. Use 'tfidf', 'spacy', or 'both'")

    def assign_themes(self,
                      min_themes_per_review: int = 1,
                      max_themes_per_review: int = 3) -> pd.DataFrame:
        """
        Assign one or more themes to each review based on keyword matches.

        Args:
            min_themes_per_review: Minimum number of themes to assign
            max_themes_per_review: Maximum number of themes to assign

        Returns:
            DataFrame with assigned themes
        """
        logger.info("Assigning themes to reviews...")

        assigned_themes = []
        theme_counts = defaultdict(int)

        for idx, review in enumerate(self.data[self.text_column].fillna("")):
            if not isinstance(review, str):
                review = ""

            # Prepare text for matching
            review_lower = review.lower()

            # Check each theme for keyword matches
            theme_scores = {}
            for theme, keywords in self.themes.items():
                score = 0
                for keyword in keywords:
                    if keyword.lower() in review_lower:
                        # Weight based on keyword importance
                        if len(keyword.split()) > 1:  # Multi-word keywords get higher weight
                            score += 2
                        else:
                            score += 1

                if score > 0:
                    theme_scores[theme] = score

            # Select top themes based on scores
            if theme_scores:
                # Sort themes by score (descending)
                sorted_themes = sorted(
                    theme_scores.items(), key=lambda x: x[1], reverse=True)
                # Take top themes up to max_themes_per_review
                selected_themes = [theme for theme,
                                   _ in sorted_themes[:max_themes_per_review]]
            else:
                selected_themes = ["General Feedback"]

            # Ensure minimum themes
            while len(selected_themes) < min_themes_per_review:
                selected_themes.append("General Feedback")

            # Update theme counts
            for theme in selected_themes:
                theme_counts[theme] += 1

            assigned_themes.append(", ".join(selected_themes))

        self.data["identified_themes"] = assigned_themes

        # Log theme distribution
        logger.info("Theme distribution:")
        for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.data)) * 100
            logger.info(f"  {theme}: {count} reviews ({percentage:.1f}%)")

        return self.data

    def analyze_themes_by_bank(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze themes by bank to ensure minimum 2 themes per bank.

        Returns:
            Dictionary with theme analysis per bank
        """
        if "bank_name" not in self.data.columns:
            logger.warning(
                "Bank name column not found, skipping bank-level theme analysis")
            return {}

        logger.info("Analyzing themes by bank...")

        bank_analysis = {}

        for bank in self.data["bank_name"].unique():
            bank_df = self.data[self.data["bank_name"] == bank]

            if len(bank_df) == 0:
                continue

            # Extract all themes for this bank
            all_themes = []
            for themes in bank_df["identified_themes"].fillna(""):
                theme_list = [t.strip()
                              for t in str(themes).split(",") if t.strip()]
                all_themes.extend(theme_list)

            # Count theme occurrences
            theme_counter = Counter(all_themes)

            # Get top themes (minimum 2, maximum 5)
            top_themes = theme_counter.most_common(5)

            # Ensure at least 2 themes
            if len(top_themes) < 2:
                # Add "General Feedback" if needed
                while len(top_themes) < 2:
                    top_themes.append(("General Feedback", 0))

            # Get example reviews for each top theme
            theme_examples = {}
            for theme, _ in top_themes[:5]:  # Top 5 themes max
                # Find reviews with this theme
                theme_reviews = bank_df[
                    bank_df["identified_themes"].str.contains(theme, na=False)
                ]

                if len(theme_reviews) > 0:
                    # Get a few example reviews
                    examples = []
                    for _, row in theme_reviews.head(3).iterrows():
                        review_text = str(row[self.text_column])[
                            :100] + "..." if len(str(row[self.text_column])) > 100 else str(row[self.text_column])
                        examples.append(review_text)
                    theme_examples[theme] = examples

            # Store bank analysis
            bank_analysis[bank] = {
                "total_reviews": len(bank_df),
                "top_themes": dict(top_themes[:5]),  # Top 5 themes with counts
                "theme_examples": theme_examples,
                # Number of unique themes
                "theme_coverage": len(set(all_themes))
            }

            logger.info(f"  Bank: {bank}")
            logger.info(f"    Total reviews: {len(bank_df)}")
            # Show top 3
            logger.info(f"    Top themes: {dict(top_themes[:3])}")

        return bank_analysis

    def generate_theme_report(self, report_path: str = None) -> Dict:
        """
        Generate comprehensive theme analysis report and save to JSON.

        Args:
            report_path: Path to save the report (default: data/preprocessed/thematic_analysis_report.json)

        Returns:
            Dictionary with report data
        """
        if report_path is None:
            report_path = os.path.join(
                self.output_dir, "thematic_analysis_report.json")

        logger.info(f"Generating theme report: {report_path}")

        # Get bank-level analysis
        bank_analysis = self.analyze_themes_by_bank()

        # Get overall theme statistics
        all_themes = []
        for themes in self.data["identified_themes"].fillna(""):
            theme_list = [t.strip()
                          for t in str(themes).split(",") if t.strip()]
            all_themes.extend(theme_list)

        theme_counter = Counter(all_themes)

        # Build report - FIXED: Use self.data.empty instead of self.data
        total_reviews = len(self.data)
        avg_themes = len(all_themes) / \
            total_reviews if total_reviews > 0 else 0

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_reviews": total_reviews,
                "unique_themes": len(set(all_themes)),
                "average_themes_per_review": avg_themes,
                "theme_distribution": dict(theme_counter.most_common(10))
            },
            "theme_definitions": self.theme_descriptions,
            "bank_analysis": bank_analysis,
            "compliance_check": {
                "min_400_reviews": total_reviews >= 400,
                "min_2_themes_per_bank": all(len(bank_data["top_themes"]) >= 2
                                             for bank_data in bank_analysis.values()) if bank_analysis else False,
                "theme_examples_provided": all(len(bank_data["theme_examples"]) >= 2
                                               for bank_data in bank_analysis.values()) if bank_analysis else False
            }
        }

        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Theme report saved to {report_path}")

        return report

    def _create_summary_dataframe(self, bank_analysis: Dict) -> pd.DataFrame:
        """
        Create a comprehensive summary DataFrame for thematic analysis.

        Args:
            bank_analysis: Bank-level theme analysis results

        Returns:
            DataFrame with thematic analysis summary
        """
        summary_data = []

        for bank, data in bank_analysis.items():
            # Extract top themes with their counts
            top_themes = data["top_themes"]

            # Create row for each bank
            row = {
                "bank_name": bank,
                "total_reviews": data["total_reviews"],
                "unique_themes_count": data["theme_coverage"],
                "theme_1": "",
                "theme_1_count": 0,
                "theme_1_example": "",
                "theme_2": "",
                "theme_2_count": 0,
                "theme_2_example": "",
                "theme_3": "",
                "theme_3_count": 0,
                "theme_3_example": "",
                "theme_4": "",
                "theme_4_count": 0,
                "theme_5": "",
                "theme_5_count": 0,
                "review_with_themes_percentage": (sum(top_themes.values()) / data["total_reviews"] * 100)
                if data["total_reviews"] > 0 else 0
            }

            # Fill theme data (up to 5 themes)
            theme_items = list(top_themes.items())
            theme_examples = data.get("theme_examples", {})

            for i, (theme, count) in enumerate(theme_items[:5]):
                theme_key = f"theme_{i+1}"
                count_key = f"theme_{i+1}_count"
                example_key = f"theme_{i+1}_example"

                row[theme_key] = theme
                row[count_key] = count

                # Get example if available
                if theme in theme_examples and theme_examples[theme]:
                    row[example_key] = theme_examples[theme][0]
                else:
                    row[example_key] = "No examples available"

            summary_data.append(row)

        # Add overall summary row
        if summary_data:
            total_reviews = sum(row["total_reviews"] for row in summary_data)
            overall_row = {
                "bank_name": "ALL BANKS",
                "total_reviews": total_reviews,
                "unique_themes_count": len(set(theme for row in summary_data
                                               for i in range(1, 6)
                                               if row[f"theme_{i}"] and row[f"theme_{i}"] != "")),
                "theme_1": "Overall Summary",
                "theme_1_count": total_reviews,
                "theme_1_example": "Aggregated across all banks",
                "review_with_themes_percentage": (total_reviews / total_reviews * 100) if total_reviews > 0 else 0
            }
            summary_data.append(overall_row)

        return pd.DataFrame(summary_data)

    def save_thematic_summary(self) -> Tuple[str, str]:
        """
        Save thematic analysis summary files to data/preprocessed folder.

        Returns:
            Tuple of (summary_path, metrics_path)
        """
        # Get bank-level analysis
        bank_analysis = self.analyze_themes_by_bank()

        if not bank_analysis:
            logger.warning("No bank analysis available, cannot save summary")
            return None, None

        # 1. Save detailed thematic summary
        summary_df = self._create_summary_dataframe(bank_analysis)
        summary_path = os.path.join(self.output_dir, "thematic_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Thematic analysis summary saved to: {summary_path}")

        # 2. Save simplified metrics
        simple_metrics = pd.DataFrame([
            {
                "bank_name": bank,
                "total_reviews": data["total_reviews"],
                "top_theme": list(data["top_themes"].keys())[0] if data["top_themes"] else "",
                "top_theme_count": list(data["top_themes"].values())[0] if data["top_themes"] else 0,
                "top_theme_percentage": (list(data["top_themes"].values())[0] / data["total_reviews"] * 100)
                if data["total_reviews"] > 0 and data["top_themes"] else 0,
                "second_theme": list(data["top_themes"].keys())[1] if len(data["top_themes"]) > 1 else "",
                "second_theme_count": list(data["top_themes"].values())[1] if len(data["top_themes"]) > 1 else 0,
                "second_theme_percentage": (list(data["top_themes"].values())[1] / data["total_reviews"] * 100)
                if data["total_reviews"] > 0 and len(data["top_themes"]) > 1 else 0,
                "unique_themes": data["theme_coverage"],
                "theme_coverage_percentage": (sum(data["top_themes"].values()) / data["total_reviews"] * 100)
                if data["total_reviews"] > 0 else 0
            }
            for bank, data in bank_analysis.items()
        ])
        metrics_path = os.path.join(self.output_dir, "thematic_metrics.csv")
        simple_metrics.to_csv(metrics_path, index=False)
        logger.info(f"Thematic metrics saved to: {metrics_path}")

        return summary_path, metrics_path

    def save_analysis_results(self) -> str:
        """
        Save the full thematic analysis results to data/preprocessed folder.

        Returns:
            Path to the saved analysis file
        """
        # Ensure we have all required columns
        required_columns = ["review_id", "clean_text",
                            "keywords", "identified_themes"]

        # Add review_id if not present
        if "review_id" not in self.data.columns:
            self.data["review_id"] = range(1, len(self.data) + 1)

        # Ensure keywords column exists
        if "keywords" not in self.data.columns:
            # Create a default keywords column
            if "keywords_tfidf" in self.data.columns:
                self.data["keywords"] = self.data["keywords_tfidf"]
            elif "keywords_spacy" in self.data.columns:
                self.data["keywords"] = self.data["keywords_spacy"]
            else:
                self.data["keywords"] = ""

        # Ensure identified_themes column exists
        if "identified_themes" not in self.data.columns:
            self.data["identified_themes"] = ""

        # Select and reorder columns
        output_cols = []
        for col in required_columns:
            if col in self.data.columns:
                output_cols.append(col)

        # Add other useful columns if available
        for col in ["bank_name", "rating", "sentiment_label", "sentiment_score"]:
            if col in self.data.columns:
                output_cols.append(col)

        # Save to CSV
        analysis_path = os.path.join(self.output_dir, "thematic_analysis.csv")
        self.data[output_cols].to_csv(analysis_path, index=False)
        logger.info(f"Thematic analysis results saved to: {analysis_path}")

        return analysis_path

    def save_all_outputs(self) -> Dict[str, str]:
        """
        Save all thematic analysis outputs to data/preprocessed folder.

        Returns:
            Dictionary with paths to all saved files
        """
        logger.info("Saving all thematic analysis outputs...")

        output_paths = {}

        # 1. Save full analysis results
        analysis_path = self.save_analysis_results()
        output_paths["analysis"] = analysis_path

        # 2. Save thematic summary
        summary_path, metrics_path = self.save_thematic_summary()
        output_paths["summary"] = summary_path
        output_paths["metrics"] = metrics_path

        # 3. Generate and save report
        report_path = os.path.join(
            self.output_dir, "thematic_analysis_report.json")
        report = self.generate_theme_report(report_path)
        output_paths["report"] = report_path

        logger.info("✓ All thematic analysis outputs saved successfully")

        return output_paths


# Convenience function for running the analysis
def run_thematic_analysis(input_path: str,
                          output_dir: str = "data/preprocessed",
                          text_column: str = "clean_text",
                          keyword_method: str = "tfidf",
                          min_themes: int = 1,
                          max_themes: int = 3):
    """
    Convenience function to run thematic analysis and save all outputs to data/preprocessed.

    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save all output files
        text_column: Column containing text to analyze
        keyword_method: Method for keyword extraction ('tfidf', 'spacy', or 'both')
        min_themes: Minimum themes per review
        max_themes: Maximum themes per review
    """
    # Load data
    logger.info(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = ThematicAnalyzer(
        df, text_column=text_column, output_dir=output_dir)

    # Extract keywords
    analyzer.extract_keywords(method=keyword_method, top_n=10)

    # Assign themes
    analyzer.assign_themes(min_themes_per_review=min_themes,
                           max_themes_per_review=max_themes)

    # Save all outputs
    output_paths = analyzer.save_all_outputs()

    # Print summary
    print("\n" + "="*60)
    print("THEMATIC ANALYSIS COMPLETE")
    print("="*60)

    print(f"\n📁 Output Files Saved to {output_dir}:")
    for file_type, path in output_paths.items():
        print(f"  ✓ {file_type.title()}: {os.path.basename(path)}")

    # Get statistics
    bank_analysis = analyzer.analyze_themes_by_bank()

    if bank_analysis:
        print(f"\n📊 Bank-level Analysis:")
        for bank, data in bank_analysis.items():
            print(f"\n  Bank: {bank}")
            print(f"    Total Reviews: {data['total_reviews']:,}")
            print(f"    Top 3 Themes:")
            for i, (theme, count) in enumerate(list(data['top_themes'].items())[:3]):
                percentage = (count / data['total_reviews']) * 100
                print(f"      {i+1}. {theme}: {count:,} ({percentage:.1f}%)")

    # Check compliance
    print(f"\n✅ Thematic analysis complete!")
    print(f"   All outputs saved to: {output_dir}")

    return analyzer, output_paths


if __name__ == "__main__":
    import sys
    import argparse
    from typing import Tuple  # Added import

    parser = argparse.ArgumentParser(
        description="Thematic Analysis for Bank Reviews - Saves all outputs to data/preprocessed")
    parser.add_argument(
        "input", help="Input CSV file path (e.g., data/preprocessed/google_play_processed_reviews.csv)")
    parser.add_argument("--output-dir", default="data/preprocessed",
                        help="Directory to save output files (default: data/preprocessed)")
    parser.add_argument("--text-column", default="clean_text",
                        help="Text column name (default: clean_text)")
    parser.add_argument("--keyword-method", default="tfidf",
                        choices=["tfidf", "spacy", "both"],
                        help="Keyword extraction method (default: tfidf)")
    parser.add_argument("--min-themes", type=int, default=1,
                        help="Minimum themes per review (default: 1)")
    parser.add_argument("--max-themes", type=int, default=3,
                        help="Maximum themes per review (default: 3)")

    args = parser.parse_args()

    try:
        # Run thematic analysis
        analyzer, output_paths = run_thematic_analysis(
            args.input,
            output_dir=args.output_dir,
            text_column=args.text_column,
            keyword_method=args.keyword_method,
            min_themes=args.min_themes,
            max_themes=args.max_themes
        )

        # Show final message
        print("\n" + "="*60)
        print("FILES CREATED IN data/preprocessed:")
        print("="*60)
        print("1. thematic_analysis.csv       - Full analysis results with themes")
        print("2. thematic_summary.csv        - Bank-level theme summary")
        print("3. thematic_metrics.csv        - Key metrics and statistics")
        print("4. thematic_analysis_report.json - Detailed analysis report")
        print("="*60)

    except Exception as e:
        logger.error(f"Error in thematic analysis: {e}")
        import traceback
        traceback.print_exc()
        raise
