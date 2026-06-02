"""
NL2SQL — Phase 3: Feature Engineer for ML Intent Classification
Run: python ml/feature_engineer.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import INTENTS
from utils.logger import get_logger

logger = get_logger(__name__)


def _contains_word(text: str, word: str) -> bool:
    """Check if word exists as a whole word using regex boundaries."""
    pattern = rf'\b{re.escape(word)}\b'
    return re.search(pattern, text) is not None


class FeatureEngineer:
    """
    Converts raw query text into feature vectors for ML models.

    Combines TF-IDF vectorization with hand-crafted linguistic features
    to create rich feature representations for intent classification.

    Attributes:
        vectorizer: TfidfVectorizer instance for text features
        label_encoder: LabelEncoder for intent labels
        is_fitted: Flag indicating if vectorizer has been fitted
    """

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple = (1, 3)
    ):
        """
        Initialize FeatureEngineer with TF-IDF and label encoder.

        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: Range of n-grams to extract (min_n, max_n)
        """
        logger.info(f"Initializing FeatureEngineer (max_features={max_features}, ngram_range={ngram_range})")

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            analyzer="word",
            sublinear_tf=True,
            min_df=1,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.is_fitted = False
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(INTENTS)

        logger.info("FeatureEngineer initialized successfully")

    def fit(self, questions: List[str]) -> None:
        """
        Fit TF-IDF vectorizer on a corpus of questions.

        Args:
            questions: List of question strings to fit on

        Raises:
            RuntimeError: If questions list is empty
        """
        if not questions:
            raise RuntimeError("Cannot fit FeatureEngineer on empty questions list")

        logger.info(f"Fitting TF-IDF vectorizer on {len(questions)} questions")
        self.vectorizer.fit(questions)
        self.is_fitted = True

        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(f"TF-IDF vectorizer fitted. Vocabulary size: {vocab_size} tokens")

    def transform(self, questions: List[str]) -> Any:
        """
        Transform questions into TF-IDF feature matrix.

        Args:
            questions: List of question strings

        Returns:
            Sparse feature matrix (n_samples × n_features)

        Raises:
            RuntimeError: If vectorizer is not fitted
        """
        if not self.is_fitted:
            raise RuntimeError(
                "FeatureEngineer not fitted. "
                "Call fit() or fit_transform() before transform()."
            )

        logger.debug(f"Transforming {len(questions)} questions to TF-IDF features")
        return self.vectorizer.transform(questions)

    def fit_transform(self, questions: List[str]) -> Any:
        """
        Fit vectorizer and transform questions in one step.

        Args:
            questions: List of question strings

        Returns:
            Sparse feature matrix (n_samples × n_features)
        """
        logger.info(f"Fit-transforming {len(questions)} questions")
        X = self.vectorizer.fit_transform(questions)
        self.is_fitted = True
        vocab_size = self.vectorizer.vocabulary_.__len__()
        logger.info(f"Fit-transform complete. Vocabulary size: {vocab_size}")
        return X

    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """
        Convert intent label strings to integer indices.

        Args:
            labels: List of intent label strings

        Returns:
            numpy array of encoded integer labels

        Raises:
            ValueError: If any label is not in INTENTS
        """
        logger.debug(f"Encoding {len(labels)} labels")
        return self.label_encoder.transform(labels)

    def decode_labels(self, indices: List[int]) -> List[str]:
        """
        Convert integer indices back to intent label strings.

        Args:
            indices: List of integer label indices

        Returns:
            List of intent label strings
        """
        logger.debug(f"Decoding {len(indices)} label indices")
        return self.label_encoder.inverse_transform(indices).tolist()

    def build_features(
        self,
        question: str,
        nlp_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build rich feature dictionary combining TF-IDF with hand-crafted features.

        Args:
            question: Raw question string
            nlp_result: Optional output from NLPPipeline.process()

        Returns:
            Dictionary with features:
            {
                "tfidf_vector": sparse matrix (1 × max_features),
                "has_aggregate": bool,
                "has_comparison": bool,
                "has_order": bool,
                "has_join": bool,
                "has_group": bool,
                "has_limit": bool,
                "number_count": int,
                "word_count": int,
                "intent_hint": str
            }
        """
        # Get TF-IDF vector
        if self.is_fitted:
            tfidf_vector = self.vectorizer.transform([question])
        else:
            # Return empty sparse vector if not fitted
            from scipy.sparse import csr_matrix
            tfidf_vector = csr_matrix((1, 1))

        # Hand-crafted features
        question_lower = question.lower()
        words = question_lower.split()
        word_count = len(words)

        # Keyword-based binary features (using word-boundary matching)
        aggregate_keywords = {"count", "sum", "avg", "average", "max", "minimum", "min", "maximum"}
        comparison_keywords = {"greater", "less", "above", "below", "more than", "less than", "at least", "at most", "between", "equal"}
        order_keywords = {"top", "bottom", "first", "last", "sort", "order by", "ranked", "highest", "lowest", "most expensive", "cheapest"}
        join_keywords = {"and", "with", "their", "along", "including", "combined", "join"}
        group_keywords = {"group", "per", "each", "every", "grouped"}
        limit_keywords = {"limit", "only"}

        # Use word-boundary matching to avoid false positives
        has_aggregate = any(_contains_word(question_lower, kw) for kw in aggregate_keywords)
        has_comparison = any(_contains_word(question_lower, kw) for kw in comparison_keywords)
        has_order = any(_contains_word(question_lower, kw) for kw in order_keywords)
        has_join = any(_contains_word(question_lower, kw) for kw in join_keywords)
        has_group = any(_contains_word(question_lower, kw) for kw in group_keywords)
        has_limit = any(_contains_word(question_lower, kw) for kw in limit_keywords)

        # Count numbers in question
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', question)
        number_count = len(numbers)

        # Get intent hint from NLP result if available
        intent_hint = "SELECT"  # default
        if nlp_result and "sql_hints" in nlp_result:
            intent_hint = nlp_result["sql_hints"].get("intent_hint", "SELECT")

        features = {
            "tfidf_vector": tfidf_vector,
            "has_aggregate": has_aggregate,
            "has_comparison": has_comparison,
            "has_order": has_order,
            "has_join": has_join,
            "has_group": has_group,
            "has_limit": has_limit,
            "number_count": number_count,
            "word_count": word_count,
            "intent_hint": intent_hint
        }

        logger.debug(f"Built features for question: {question[:50]}...")
        return features

    def save(self, path: str) -> None:
        """
        Save fitted vectorizer and label encoder to disk.

        Args:
            path: Full file path to save to (e.g., models/saved/tfidf_vectorizer.pkl)
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            "vectorizer": self.vectorizer,
            "label_encoder": self.label_encoder,
            "is_fitted": self.is_fitted
        }, filepath, compress=3)

        logger.info(f"FeatureEngineer saved to {filepath}")

    def load(self, path: str) -> None:
        """
        Load fitted vectorizer and label encoder from disk.

        Args:
            path: Exact file path to load (e.g., models/saved/tfidf_vectorizer.pkl)

        Raises:
            FileNotFoundError: If saved file doesn't exist
        """
        filepath = Path(path)

        if not filepath.exists():
            raise FileNotFoundError(f"FeatureEngineer file not found: {filepath}")

        logger.info(f"Loading FeatureEngineer from {filepath}")
        data = joblib.load(filepath)

        self.vectorizer = data["vectorizer"]
        self.label_encoder = data["label_encoder"]
        self.is_fitted = data["is_fitted"]

        logger.info("FeatureEngineer loaded successfully")


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FeatureEngineer — Phase 3 Standalone Test")
    print("=" * 70)

    fe = FeatureEngineer(max_features=1000, ngram_range=(1, 2))

    sample_questions = [
        "Show all customers from Mumbai",
        "Count total orders by city",
        "Average salary of employees",
        "Top 5 products by price",
        "Customer names with their orders",
    ]

    print("\nFitting and transforming sample questions...")
    X = fe.fit_transform(sample_questions)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Vocabulary size: {len(fe.vectorizer.vocabulary_)}")

    print("\nHand-crafted features for each question:")
    for q in sample_questions:
        feats = fe.build_features(q)
        print(f"  {q[:45]:<45} "
              f"agg={feats['has_aggregate']} "
              f"cmp={feats['has_comparison']} "
              f"ord={feats['has_order']} "
              f"jn={feats['has_join']} "
              f"grp={feats['has_group']} "
              f"lim={feats['has_limit']}")

    print("\n" + "=" * 70)
    print("FeatureEngineer test complete")
    print("=" * 70)
