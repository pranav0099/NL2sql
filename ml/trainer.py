"""
NL2SQL — Phase 3: ML Intent Classifier Trainer
Run: python ml/trainer.py
"""

import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import logging
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
import joblib

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config.config import (
    INTENTS,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    ML_MODEL_PATH,
    VECTORIZER_PATH,
    CONFIDENCE_THRESHOLD
)
from utils.logger import get_logger
from ml.feature_engineer import FeatureEngineer

logger = get_logger(__name__)


class IntentClassifierTrainer:
    """
    Trains and evaluates multiple ML classifiers for intent detection.

    Trains SVM (linear/rbf), Logistic Regression, and Random Forest
    on TF-IDF features. Selects best model by F1 score. Saves model
    and feature engineer for production use.

    Attributes:
        feature_engineer: FeatureEngineer instance
        best_model: Best trained sklearn Pipeline
        best_model_name: Name of best model
        best_score: Best validation F1 score
        training_report: Dictionary with training metrics
    """

    def __init__(self):
        """Initialize trainer with feature engineer."""
        logger.info("Initializing IntentClassifierTrainer")
        self.feature_engineer = FeatureEngineer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE
        )
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0.0
        self.training_report = {}
        logger.info("Trainer initialized")

    def load_data(
        self,
        train_path: str,
        val_path: str
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Load training and validation data from JSON files.

        Args:
            train_path: Path to train.json
            val_path: Path to val.json

        Returns:
            Tuple of (train_questions, train_labels, val_questions, val_labels)

        Raises:
            FileNotFoundError: If data files missing
            ValueError: If data format invalid
        """
        logger.info(f"Loading data from {train_path} and {val_path}")

        def load_json_split(path: str) -> Tuple[List[str], List[str]]:
            with open(path, 'r', encoding='utf-8') as f:
                records = json.load(f)

            questions = []
            labels = []

            for record in records:
                # Extract question
                if "question" not in record:
                    raise ValueError(f"Missing 'question' field in record: {record}")

                question = record["question"]

                # Extract or derive intent label
                if "intent" in record:
                    intent = record["intent"]
                else:
                    # Fallback: classify intent from SQL query
                    intent = self._classify_intent_from_sql(record["query"])

                questions.append(question)
                labels.append(intent)

            logger.info(f"Loaded {len(questions)} records from {path}")
            return questions, labels

        X_train, y_train = load_json_split(train_path)
        X_val, y_val = load_json_split(val_path)

        # Log label distribution
        train_dist = {label: y_train.count(label) for label in set(y_train)}
        logger.info(f"Train size: {len(X_train)}, Label distribution: {train_dist}")
        val_dist = {label: y_val.count(label) for label in set(y_val)}
        logger.info(f"Val size: {len(X_val)}, Label distribution: {val_dist}")

        return X_train, y_train, X_val, y_val

    def _classify_intent_from_sql(self, sql: str) -> str:
        """
        Fallback intent classifier from SQL query string.

        Uses pattern matching on SQL to infer intent label.
        Priority order checks most specific patterns first.

        Args:
            sql: SQL query string

        Returns:
            Intent label string
        """
        sql_upper = sql.upper()

        # Check for COMPLEX (2+ SELECT statements)
        if sql_upper.count("SELECT") >= 2:
            return "COMPLEX"

        # Check specific intent patterns
        if "JOIN" in sql_upper:
            return "SELECT_JOIN"
        if "GROUP BY" in sql_upper:
            return "SELECT_GROUP"

        aggregate_keywords = ["COUNT", "SUM", "AVG", "MAX", "MIN"]
        if any(kw in sql_upper for kw in aggregate_keywords):
            return "SELECT_AGGREGATE"

        if "ORDER BY" in sql_upper:
            return "SELECT_ORDER"
        if "WHERE" in sql_upper:
            return "SELECT_WHERE"
        if "LIMIT" in sql_upper:
            return "SELECT_LIMIT"

        return "SELECT"

    def train(
        self,
        train_path: str = "data/processed/train.json",
        val_path: str = "data/processed/val.json"
    ) -> Dict[str, Any]:
        """
        Full training pipeline: load data, train models, evaluate, save best.

        Args:
            train_path: Path to training data
            val_path: Path to validation data

        Returns:
            Training report dictionary with metrics
        """
        logger.info("="*70)
        logger.info("STARTING PHASE 3 TRAINING")
        logger.info("="*70)

        # Step 1: Load data
        logger.info("\n[Step 1/5] Loading data...")
        X_train, y_train, X_val, y_val = self.load_data(train_path, val_path)

        # Check for class imbalance warnings
        train_label_counts = {label: y_train.count(label) for label in INTENTS}
        min_class_count = min(train_label_counts.get(label, 0) for label in set(y_train))
        if min_class_count < 10:
            logger.warning(
                f"Some classes have fewer than 10 training samples. "
                f"Minimum: {min_class_count}. This may affect model performance."
            )

        # Step 2: Fit feature engineer and transform
        logger.info("\n[Step 2/5] Fitting TF-IDF vectorizer...")
        X_train_tfidf = self.feature_engineer.fit_transform(X_train)
        X_val_tfidf = self.feature_engineer.transform(X_val)

        # Encode labels
        y_train_encoded = self.feature_engineer.encode_labels(y_train)
        y_val_encoded = self.feature_engineer.encode_labels(y_val)

        logger.info(f"TF-IDF features: {X_train_tfidf.shape[1]} dimensions")

        # Step 3: Define candidate models
        logger.info("\n[Step 3/5] Training candidate models...")
        models = {
            "SVM_linear": Pipeline([
                ("clf", SVC(
                    kernel="linear",
                    C=1.0,
                    probability=True,
                    random_state=42,
                    class_weight="balanced"
                ))
            ]),
            "SVM_rbf": Pipeline([
                ("clf", SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    random_state=42,
                    class_weight="balanced"
                ))
            ]),
            "LogisticRegression": Pipeline([
                ("clf", LogisticRegression(
                    C=5.0,
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                    solver="lbfgs",
                    multi_class="multinomial"
                ))
            ]),
            "RandomForest": Pipeline([
                ("clf", RandomForestClassifier(
                    n_estimators=200,
                    max_depth=None,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
                ))
            ]),
        }

        model_scores = {}
        classification_reports = {}

        # Step 4: Train and evaluate each model
        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            model.fit(X_train_tfidf, y_train_encoded)

            # Predict on validation set
            y_pred_encoded = model.predict(X_val_tfidf)

            # Decode back to string labels
            y_pred = self.feature_engineer.decode_labels(y_pred_encoded.tolist())

            # Compute metrics
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)
            report = classification_report(
                y_val, y_pred,
                labels=INTENTS,
                target_names=INTENTS,
                zero_division=0
            )

            model_scores[name] = {"accuracy": accuracy, "f1": f1}
            classification_reports[name] = report

            logger.info(f"  {name:<25} acc={accuracy:.3f}  f1={f1:.3f}")

            # Track best model
            if f1 > self.best_score:
                self.best_score = f1
                self.best_model = model
                self.best_model_name = name

        logger.info(f"\nBest model: {self.best_model_name} (f1={self.best_score:.3f})")

        # Step 5: Cross-validation on best model
        logger.info("\n[Step 5/5] Cross-validation (5-fold)...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []

        # Use label encoded data for CV
        if hasattr(X_train_tfidf, 'toarray'):
            X_train_dense = X_train_tfidf.toarray()
            X_val_dense = X_val_tfidf.toarray()
            all_X = np.vstack([X_train_dense, X_val_dense])
        else:
            from scipy.sparse import vstack
            all_X = vstack([X_train_tfidf, X_val_tfidf])

        all_y = np.array(list(y_train_encoded) + list(y_val_encoded))

        for fold, (train_idx, val_idx) in enumerate(cv.split(all_X, all_y), 1):
            fold_X_train = all_X[train_idx]
            fold_y_train = all_y[train_idx]
            fold_X_val = all_X[val_idx]
            fold_y_val = all_y[val_idx]

            # Clone best model
            from sklearn.base import clone
            fold_model = clone(self.best_model)
            fold_model.fit(fold_X_train, fold_y_train)
            fold_pred = fold_model.predict(fold_X_val)

            fold_acc = accuracy_score(fold_y_val, fold_pred)
            cv_scores.append(fold_acc)
            logger.debug(f"  Fold {fold}: acc={fold_acc:.3f}")

        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
        logger.info(f"CV accuracy: {cv_mean:.3f} ± {cv_std:.3f}")

        # Step 6: Save best model + feature engineer
        logger.info("\n[Step 6/5] Saving model artifacts...")
        models_dir = Path(ML_MODEL_PATH).parent
        models_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.best_model, ML_MODEL_PATH, compress=3)
        logger.info(f"Best model saved to {ML_MODEL_PATH}")

        self.feature_engineer.save(VECTORIZER_PATH)

        # Step 7: Build training report
        self.training_report = {
            "best_model": self.best_model_name,
            "val_accuracy": float(model_scores[self.best_model_name]["accuracy"]),
            "val_f1": float(model_scores[self.best_model_name]["f1"]),
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "all_models": {
                name: {
                    "accuracy": float(scores["accuracy"]),
                    "f1": float(scores["f1"])
                }
                for name, scores in model_scores.items()
            },
            "classification_report": classification_reports[self.best_model_name],
            "train_size": len(X_train),
            "val_size": len(X_val),
            "label_distribution": train_label_counts
        }

        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"Best model   : {self.best_model_name}")
        logger.info(f"Val accuracy : {self.training_report['val_accuracy']:.1%}")
        logger.info(f"Val F1       : {self.training_report['val_f1']:.1%}")
        logger.info(f"CV mean±std  : {self.training_report['cv_mean']:.1%} ± {self.training_report['cv_std']:.1%}")

        return self.training_report

    def plot_confusion_matrix(
        self,
        y_true: List[str],
        y_pred: List[str],
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix heatmap.

        Args:
            y_true: True intent labels
            y_pred: Predicted intent labels
            save_path: Optional path to save PNG
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("matplotlib and seaborn required for plotting. Install: pip install matplotlib seaborn")
            return

        logger.info("Generating confusion matrix plot...")

        cm = confusion_matrix(y_true, y_pred, labels=INTENTS)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=INTENTS,
            yticklabels=INTENTS,
            square=True
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix — {self.best_model_name}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
            plt.close()
        else:
            plt.show()


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TRAINING ML INTENT CLASSIFIER — PHASE 3")
    print("="*70)

    try:
        trainer = IntentClassifierTrainer()
        print("\n[INFO] Loading and preparing data...")
        report = trainer.train(
            train_path="data/processed/train.json",
            val_path="data/processed/val.json"
        )

        print("\n" + "="*55)
        print("TRAINING COMPLETE")
        print("="*55)
        print(f"Best model   : {report['best_model']}")
        print(f"Val accuracy : {report['val_accuracy']:.1%}")
        print(f"Val F1       : {report['val_f1']:.1%}")
        print(f"CV mean±std  : {report['cv_mean']:.1%} ± {report['cv_std']:.1%}")

        print("\nAll model scores:")
        for name, scores in report["all_models"].items():
            print(f"  {name:<25} acc={scores['accuracy']:.1%}  f1={scores['f1']:.1%}")

        print("\nClassification report:")
        print(report["classification_report"])

        print("\nLabel distribution in training set:")
        for label, count in report["label_distribution"].items():
            print(f"  {label:<20}: {count:>5}")

    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
