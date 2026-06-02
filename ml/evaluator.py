"""
NL2SQL — Phase 3: Model Evaluator — Deep Analysis & Error Inspection
Run: python ml/evaluator.py
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import logging
import json
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger
from ml.classifier import IntentClassifier

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Deep evaluation and error analysis for intent classifier.

    Loads classifier, runs evaluation on test data, identifies
    common error patterns, and generates human-readable reports.

    Attributes:
        classifier: Loaded IntentClassifier instance
        results: Latest evaluation results
    """

    def __init__(self, classifier: IntentClassifier):
        """
        Initialize evaluator with loaded classifier.

        Args:
            classifier: IntentClassifier instance
        """
        self.classifier = classifier
        self.results = []
        logger.info("ModelEvaluator initialized")

    def run_full_evaluation(
        self,
        test_path: str = "data/processed/test.json"
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on test set.

        Args:
            test_path: Path to test.json

        Returns:
            Dictionary with metrics, error analysis, and statistics
        """
        logger.info("Running full model evaluation")

        if not self.classifier.is_loaded:
            logger.error("Classifier not loaded. Cannot evaluate.")
            return {}

        # Load test data
        try:
            with open(test_path, 'r', encoding='utf-8') as f:
                records = json.load(f)
        except FileNotFoundError:
            logger.error(f"Test file not found: {test_path}")
            return {}

        logger.info(f"Loaded {len(records)} test records")

        questions = []
        y_true = []

        for record in records:
            question = record["question"]
            intent = record.get("intent")
            if not intent:
                # Fallback: derive from SQL
                intent = self.classifier._classify_intent_from_sql(record["query"])
            questions.append(question)
            y_true.append(intent)

        # Predict batch
        predictions = self.classifier.predict_batch(questions)
        y_pred = [p["intent"] for p in predictions]
        confidences = [p["confidence"] for p in predictions]

        # Compute overall metrics
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0.0

        from sklearn.metrics import f1_score
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Per-intent breakdown
        per_intent = {}
        for intent in set(y_true):
            intent_correct = sum(1 for t, p in zip(y_true, y_pred) if t == p and t == intent)
            intent_total = sum(1 for t in y_true if t == intent)
            per_intent[intent] = {
                "correct": intent_correct,
                "total": intent_total,
                "accuracy": intent_correct / intent_total if intent_total > 0 else 0.0
            }

        # Confidence statistics
        conf_array = np.array(confidences)
        mean_conf = float(np.mean(conf_array))
        median_conf = float(np.median(conf_array))
        below_threshold = sum(1 for c in confidences if c < 0.70)

        # Error analysis: collect misclassified examples
        errors = self._analyze_errors(questions, y_true, y_pred, confidences)

        # Low confidence examples
        low_conf_examples = [
            {"question": q, "expected": t, "predicted": p, "confidence": c}
            for q, t, p, c in zip(questions, y_true, y_pred, confidences)
            if c < 0.70
        ]
        low_conf_examples.sort(key=lambda x: x["confidence"])

        evaluation = {
            "accuracy": accuracy,
            "f1_weighted": f1,
            "correct": correct,
            "total": total,
            "per_intent": per_intent,
            "confidence_stats": {
                "mean": mean_conf,
                "median": median_conf,
                "below_threshold": below_threshold,
                "threshold": 0.70
            },
            "error_analysis": errors[:10],  # Top 10 worst errors
            "low_confidence_examples": low_conf_examples[:10]
        }

        self.results = evaluation
        logger.info(f"Evaluation complete: accuracy={accuracy:.3f}, f1={f1:.3f}")
        return evaluation

    def _analyze_errors(
        self,
        questions: List[str],
        y_true: List[str],
        y_pred: List[str],
        confidences: List[float]
    ) -> List[Dict[str, Any]]:
        """
        Analyze misclassified examples.

        Args:
            questions: List of questions
            y_true: True labels
            y_pred: Predicted labels
            confidences: Prediction confidences

        Returns:
            List of error dicts sorted by confidence (ascending)
        """
        errors = []

        for q, true, pred, conf in zip(questions, y_true, y_pred, confidences):
            if true != pred:
                errors.append({
                    "question": q,
                    "expected": true,
                    "predicted": pred,
                    "confidence": conf
                })

        # Sort by confidence (lowest first = worst errors)
        errors.sort(key=lambda x: x["confidence"])

        logger.info(f"Found {len(errors)} misclassified examples")
        return errors

    def print_report(self, evaluation: Dict[str, Any]) -> None:
        """
        Pretty-print evaluation report to console.

        Args:
            evaluation: Output from run_full_evaluation()
        """
        if not evaluation:
            print("\n[ERROR] No evaluation data to report")
            return

        print("\n" + "="*70)
        print("MODEL EVALUATION REPORT — PHASE 3 ML CLASSIFIER")
        print("="*70)

        # Overall metrics
        print(f"\nOverall Accuracy : {evaluation['accuracy']:>6.1%}  ({evaluation['correct']}/{evaluation['total']})")
        print(f"Weighted F1      : {evaluation['f1_weighted']:>6.1%}")

        # Per-intent accuracy
        print("\nPer-Intent Accuracy:")
        print("-"*70)
        per_intent = evaluation["per_intent"]
        for intent in sorted(per_intent.keys()):
            stats = per_intent[intent]
            acc_pct = stats['accuracy'] * 100
            print(f"  {intent:<20} : {acc_pct:>5.1f}%  ({stats['correct']}/{stats['total']})")

        # Confidence statistics
        print("\nConfidence Statistics:")
        print("-"*70)
        conf_stats = evaluation["confidence_stats"]
        print(f"  Mean confidence   : {conf_stats['mean']:.1%}")
        print(f"  Median confidence : {conf_stats['median']:.1%}")
        print(f"  Below threshold   : {conf_stats['below_threshold']} queries (threshold={conf_stats['threshold']:.0%})")

        # Error analysis
        print("\nTop Errors (lowest confidence):")
        print("-"*70)
        errors = evaluation["error_analysis"]
        if errors:
            for i, err in enumerate(errors, 1):
                print(f"\n  [{i}] \"{err['question'][:60]}{'...' if len(err['question']) > 60 else ''}\"")
                print(f"      Expected : {err['expected']}")
                print(f"      Got      : {err['predicted']}  (conf: {err['confidence']:.0%})")
        else:
            print("  No errors found! Perfect classification.")

        # Low confidence examples
        low_conf = evaluation.get("low_confidence_examples", [])
        if low_conf:
            print("\nLow Confidence Examples (<70%):")
            print("-"*70)
            for i, ex in enumerate(low_conf, 1):
                print(f"\n  [{i}] \"{ex['question'][:60]}{'...' if len(ex['question']) > 60 else ''}\"")
                print(f"      Expected : {ex['expected']}")
                print(f"      Predicted: {ex['predicted']} (conf: {ex['confidence']:.0%})")

        print("\n" + "="*70)


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    import json

    print("="*70)
    print("MODEL EVALUATOR — PHASE 3")
    print("="*70)

    classifier = IntentClassifier()

    if not classifier.is_loaded:
        print("\n[MODEL NOT FOUND]")
        print("Train the model first: python ml/trainer.py")
    else:
        evaluator = ModelEvaluator(classifier)

        print("\nRunning evaluation on test set...")
        evaluation = evaluator.run_full_evaluation(
            test_path="data/processed/test.json"
        )

        evaluator.print_report(evaluation)

    print("\n" + "="*70)
