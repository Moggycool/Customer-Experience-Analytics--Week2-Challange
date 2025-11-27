""" Thematic Analysis Module"""
from collections import defaultdict  # pylint: disable=unused-import
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy


class ThematicAnalyzer:
    """
    Extracts keywords and groups into predefined themes.
    """

    def __init__(self, data: pd.DataFrame, text_column: str = "review"):
        self.data = data.copy()
        self.text_column = text_column
        self.nlp = spacy.load("en_core_web_sm")
        self.themes = {
            "Account Access Issues": ["login", "password", "account", "locked", "verification"],
            "Transaction Performance": ["slow", "transfer", "payment", "failed", "processing"],
            "User Interface & Experience": ["UI", "interface", "navigation", "easy", "confusing"],
            "Customer Support": ["support", "service", "help", "response", "staff"],
            "Feature Requests": ["feature", "option", "request", "add", "improve"]
        }

    def extract_keywords(self, top_n: int = 10):
        """
        Optional: returns top TF-IDF keywords per review for exploratory purposes
        """
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        X = vectorizer.fit_transform(
            self.data[self.text_column].fillna("")
        )  # pylint: disable=invalid-name

        feature_names = vectorizer.get_feature_names_out()
        keywords_list = []

        for row in X:  # pylint: disable=invalid-name
            row_data = row.toarray()[0]
            top_indices = row_data.argsort()[-top_n:][::-1]
            keywords = [feature_names[i]
                        for i in top_indices if row_data[i] > 0]
            keywords_list.append(", ".join(keywords))
        self.data["keywords"] = keywords_list
        return self.data

    def assign_themes(self):
        """
        Assigns one or more themes to each review based on keyword matches.
        """
        assigned_themes = []

        for review in self.data[self.text_column]:
            review_themes = []
            if not isinstance(review, str):
                review = ""
            review_tokens = review.lower().split()
            for theme, keywords in self.themes.items():
                if any(kw.lower() in review_tokens for kw in keywords):
                    review_themes.append(theme)
            assigned_themes.append(
                ", ".join(review_themes) if review_themes else "Other")

        self.data["identified_themes"] = assigned_themes
        return self.data
