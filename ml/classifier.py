"""
NL2SQL — Phase 3: ML Intent Classifier (Inference)
Run: python ml/classifier.py
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import joblib

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    INTENTS,
    ML_MODEL_PATH,
    VECTORIZER_PATH,
    CONFIDENCE_THRESHOLD
)
from utils.logger import get_logger
from ml.feature_engineer import FeatureEngineer

logger = get_logger(__name__)


class IntentClassifier:
    """
    Inference-only classifier for intent detection.

    Loads pre-trained ML model and feature engineer. Provides
    high-level predict() method with confidence thresholds.
    Falls back to rule-based classification if model unavailable.

    Attributes:
        model: Loaded sklearn classifier
        feature_engineer: Loaded FeatureEngineer
        is_loaded: Flag indicating successful load
    """

    def __init__(
        self,
        model_path: str = ML_MODEL_PATH,
        vectorizer_path: str = VECTORIZER_PATH
    ):
        """
        Initialize classifier by loading saved model and feature engineer.

        Args:
            model_path: Path to saved classifier .pkl file
            vectorizer_path: Path to saved feature_engineer.joblib
        """
        self.model = None
        self.feature_engineer = None
        self.is_loaded = False

        self._load(model_path, vectorizer_path)

    def _load(
        self,
        model_path: str,
        vectorizer_path: str
    ) -> None:
        """
        Load saved model and feature engineer from disk.

        Args:
            model_path: Path to classifier pickle
            vectorizer_path: Path to feature engineer joblib
        """
        model_path = Path(model_path)
        vectorizer_path = Path(vectorizer_path)

        logger.info("Loading ML classifier...")

        # Check if both files exist
        if not model_path.exists():
            logger.error(f"Model file NOT found: {model_path}")
            logger.error("Run 'python ml/trainer.py' first to train the model.")
            self.is_loaded = False
            return

        if not vectorizer_path.exists():
            logger.error(f"Feature engineer file NOT found: {vectorizer_path}")
            logger.error("Run 'python ml/trainer.py' first to train the model.")
            self.is_loaded = False
            return

        try:
            # Load model
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded successfully")

            # Load feature engineer
            logger.info(f"Loading feature engineer from {vectorizer_path}")
            self.feature_engineer = FeatureEngineer()
            self.feature_engineer.load(str(vectorizer_path))
            logger.info(f"Feature engineer loaded successfully")

            # Determine model name for logging
            model_name = type(self.model).__name__
            if hasattr(self.model, 'steps'):
                model_name = self.model.steps[-1][0]
            logger.info(f"ML classifier loaded — model: {model_name}")

            self.is_loaded = True

        except Exception as e:
            logger.error(f"Failed to load model or feature engineer: {e}")
            self.is_loaded = False

    def predict(
        self,
        question: str,
        nlp_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Predict intent for a single question.

        Args:
            question: Raw user question string
            nlp_result: Optional NLP pipeline result for feature augmentation

        Returns:
            Prediction dictionary:
            {
                "intent": str,
                "confidence": float,
                "all_scores": dict,
                "needs_clarification": bool,
                "clarification_msg": str | None,
                "fallback_used": bool
            }
        """
        # Fallback if model not loaded
        if not self.is_loaded:
            logger.warning("Model not loaded, using rule-based fallback")
            return self._rule_based_fallback(question)

        try:
            # Build features
            features = self.feature_engineer.build_features(question, nlp_result)
            X = features["tfidf_vector"]

            # Get prediction and probabilities
            y_pred_encoded = self.model.predict(X)
            y_pred_proba = self.model.predict_proba(X)

            # Decode label
            predicted_label = self.feature_engineer.decode_labels([y_pred_encoded[0]])[0]
            confidence = float(np.max(y_pred_proba[0]))

            # Build all_scores dict
            all_scores = {}
            model_classes = self.feature_engineer.decode_labels(self.model.classes_.tolist())
            for idx, cls in enumerate(model_classes):
                all_scores[cls] = float(y_pred_proba[0][idx])

            # Ensure all INTENTS are in all_scores (even if zero)
            for intent in INTENTS:
                if intent not in all_scores:
                    all_scores[intent] = 0.0

            # Determine if clarification needed
            needs_clarification = confidence < CONFIDENCE_THRESHOLD
            clarification_msg = None

            if needs_clarification:
                clarification_msg = (
                    f"I'm not sure what type of query you mean "
                    f"(confidence: {confidence:.0%}). "
                    f"Could you rephrase? For example:\n"
                    f"  - To filter: 'Show customers where city is Mumbai'\n"
                    f"  - To count: 'How many orders were delivered?'\n"
                    f"  - To sort: 'Top 10 products by price'"
                )

            result = {
                "intent": predicted_label,
                "confidence": confidence,
                "all_scores": all_scores,
                "needs_clarification": needs_clarification,
                "clarification_msg": clarification_msg,
                "fallback_used": False
            }

            logger.debug(
                f"Prediction complete: intent={predicted_label}, "
                f"confidence={confidence:.3f}"
            )
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            # Fallback on error
            return self._rule_based_fallback(question)

    def _rule_based_fallback(self, question: str) -> Dict[str, Any]:
        """
        Pure rule-based intent classification from question text.

        Used when ML model unavailable or prediction fails.
        Checks keyword patterns in order of specificity.

        Args:
            question: Raw question string

        Returns:
            Prediction dictionary with fallback_used=True
        """
        logger.info("Using rule-based fallback classifier")
        question_lower = question.lower()

        # Check in order of priority
        join_keywords = {" and ", "with their", "along with", "including", "combined", "join"}
        if any(kw in question_lower for kw in join_keywords):
            intent = "SELECT_JOIN"
        elif "group" in question_lower or "per " in question_lower or "each " in question_lower:
            intent = "SELECT_GROUP"
        elif any(kw in question_lower for kw in ["count", "sum", "avg", "average", "max", "min"]):
            intent = "SELECT_AGGREGATE"
        elif any(kw in question_lower for kw in ["top ", "bottom ", "first ", "last ", "sort", "order", "highest", "lowest"]):
            intent = "SELECT_ORDER"
        elif any(kw in question_lower for kw in ["where", "filter", "with ", "above", "below", "greater", "less"]):
            intent = "SELECT_WHERE"
        elif "limit" in question_lower or "only" in question_lower:
            intent = "SELECT_LIMIT"
        else:
            intent = "SELECT"

        return {
            "intent": intent,
            "confidence": 0.60,
            "all_scores": {label: 0.0 for label in INTENTS},
            "needs_clarification": False,
            "clarification_msg": None,
            "fallback_used": True
        }

    def predict_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Predict intents for multiple questions efficiently.

        Batching improves performance by reducing repeated TF-IDF transform overhead.

        Args:
            questions: List of question strings

        Returns:
            List of prediction dictionaries (one per question)
        """
        logger.info(f"Batch predicting {len(questions)} questions")

        if not self.is_loaded:
            logger.warning("Model not loaded, using fallback for all questions")
            return [self._rule_based_fallback(q) for q in questions]

        try:
            # Build features for batch (TF-IDF transform once)
            # Note: We can't use build_features with nlp_result efficiently in batch,
            # so we just use text features
            from scipy.sparse import vstack

            tfidf_vectors = []
            for q in questions:
                feats = self.feature_engineer.build_features(q)
                tfidf_vectors.append(feats["tfidf_vector"])

            X_batch = vstack(tfidf_vectors)

            # Batch predict
            y_pred_encoded = self.model.predict(X_batch)
            y_pred_proba = self.model.predict_proba(X_batch)

            # Decode batch
            predictions = self.feature_engineer.decode_labels(y_pred_encoded.tolist())
            confidences = np.max(y_pred_proba, axis=1)

            # Build result list
            results = []
            for i, (pred, conf) in enumerate(zip(predictions, confidences)):
                all_scores = {}
                model_classes = self.feature_engineer.decode_labels(self.model.classes_.tolist())
                for idx, cls in enumerate(model_classes):
                    all_scores[cls] = float(y_pred_proba[i][idx])

                # Ensure all INTENTS present
                for intent in INTENTS:
                    if intent not in all_scores:
                        all_scores[intent] = 0.0

                result = {
                    "intent": pred,
                    "confidence": float(conf),
                    "all_scores": all_scores,
                    "needs_clarification": conf < CONFIDENCE_THRESHOLD,
                    "clarification_msg": None,
                    "fallback_used": False
                }

                if result["needs_clarification"]:
                    result["clarification_msg"] = (
                        f"I'm not sure what type of query you mean "
                        f"(confidence: {conf:.0%}). "
                        f"Could you rephrase?"
                    )

                results.append(result)

            logger.info(f"Batch prediction complete: {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            return [self._rule_based_fallback(q) for q in questions]

    def evaluate(self, test_path: str = "data/processed/test.json") -> Dict[str, Any]:
        """
        Evaluate classifier on test set.

        Args:
            test_path: Path to test.json with questions and intents

        Returns:
            Evaluation dictionary with detailed metrics
        """
        logger.info(f"Evaluating classifier on {test_path}")

        if not self.is_loaded:
            logger.error("Cannot evaluate: model not loaded")
            return {}

        # Load test data
        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except FileNotFoundError:
            logger.error(f"Test file not found: {test_path}")
            return {}

        questions = []
        y_true = []

        for record in records:
            questions.append(record["question"])
            intent = record.get("intent", self._classify_intent_from_sql(record["query"]))
            y_true.append(intent)

        # Predict
        predictions = self.predict_batch(questions)
        y_pred = [p["intent"] for p in predictions]
        confidences = [p["confidence"] for p in predictions]

        # Compute metrics
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0.0

        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Per-intent metrics
        per_intent = {}
        for intent in INTENTS:
            intent_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p and t == intent)
            intent_total = sum(1 for t in y_true if t == intent)
            intent_acc = intent_correct / intent_total if intent_total > 0 else 0.0
            per_intent[intent] = {
                "correct": intent_correct,
                "total": intent_total,
                "accuracy": intent_acc
            }

        # Low confidence examples
        low_conf = [
            {"question": q, "intent": t, "confidence": c}
            for q, t, c in zip(questions, y_pred, confidences)
            if c < CONFIDENCE_THRESHOLD
        ]

        result = {
            "test_accuracy": accuracy,
            "test_f1": f1,
            "correct": correct,
            "total": total,
            "per_intent": per_intent,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=INTENTS).tolist(),
            "low_confidence_examples": low_conf,
            "mean_confidence": float(np.mean(confidences)),
            "median_confidence": float(np.median(confidences))
        }

        logger.info(f"Evaluation complete: accuracy={accuracy:.3f}, f1={f1:.3f}")
        return result

    def _classify_intent_from_sql(self, sql: str) -> str:
        """Same fallback logic as in trainer."""
        sql_upper = sql.upper()

        if sql_upper.count("SELECT") >= 2:
            return "COMPLEX"
        if "JOIN" in sql_upper:
            return "SELECT_JOIN"
        if "GROUP BY" in sql_upper:
            return "SELECT_GROUP"
        if any(kw in sql_upper for kw in ["COUNT", "SUM", "AVG", "MAX", "MIN"]):
            return "SELECT_AGGREGATE"
        if "ORDER BY" in sql_upper:
            return "SELECT_ORDER"
        if "WHERE" in sql_upper:
            return "SELECT_WHERE"
        if "LIMIT" in sql_upper:
            return "SELECT_LIMIT"
        return "SELECT"


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import json

    print("="*70)
    print("ML CLASSIFIER — PHASE 3 TEST")
    print("="*70)

    classifier = IntentClassifier()

    if not classifier.is_loaded:
        print("\n[MODEL NOT FOUND]")
        print("Train the model first: python ml/trainer.py")
    else:
        TEST_QUERIES = [
            ("Show all customers from Mumbai", "SELECT"),
            ("Count total orders", "SELECT_AGGREGATE"),
            ("Average salary of employees", "SELECT_AGGREGATE"),
            ("Top 10 products by price", "SELECT_LIMIT"),
            ("Customer names with their orders", "SELECT_JOIN"),
            ("Sales grouped by city", "SELECT_GROUP"),
            ("Show first 5 records", "SELECT_LIMIT"),
            ("Customers who spent more than average", "COMPLEX"),
            ("List all departments", "SELECT"),
            ("Find employees with salary above 80000", "SELECT_WHERE"),
        ]

        print("\nTesting classifier on sample queries...\n")
        passed = 0
        for query, expected in TEST_QUERIES:
            result = classifier.predict(query)
            ok = result["intent"] == expected
            if ok:
                passed += 1
            status = "PASS" if ok else "FAIL"

            print(f"[{status}] {query[:48]:<48} "
                  f"pred={result['intent']:<20} "
                  f"conf={result['confidence']:.0%}")

        print(f"\nResult: {passed}/{len(TEST_QUERIES)} correct")

        if passed == len(TEST_QUERIES):
            print("\n[OK] All test queries passed! Phase 3 is ready.")
        else:
            print("\n[WARN] Some tests failed. Review model performance.")

        # Run evaluation on test set
        try:
            print("\n" + "-"*70)
            print("Running full test set evaluation...")
            report = classifier.evaluate()
            if report:
                print(f"\nTest set accuracy : {report['test_accuracy']:.1%}")
                print(f"Test set F1       : {report['test_f1']:.1%}")
                print(f"Mean confidence   : {report['mean_confidence']:.1%}")
                print(f"Below threshold   : {len(report['low_confidence_examples'])} queries")
        except Exception as e:
            print(f"\n[WARN] Evaluation skipped: {e}")

    print("\n" + "="*70)
