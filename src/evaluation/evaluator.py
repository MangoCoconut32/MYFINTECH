"""ModelEvaluator — compute and persist performance metrics for MYFINTECH models."""

import json
import logging
import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.base_model import BaseFinanceModel

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Calculates a standardized suite of binary classification metrics.

    Attributes:
        model: A fitted instance of a BaseFinanceModel.
        X_test: Engineered feature matrix for the test set.
        y_test: Binary target labels for the test set.
    """

    def __init__(
        self,
        model: BaseFinanceModel,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Initialize the evaluator.

        Args:
            model: Fitted BaseFinanceModel implementation.
            X_test: Test features.
            y_test: Test labels.
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def calculate_metrics(self, threshold: float = 0.5) -> Dict[str, float]:
        """Compute ROC AUC, Brier Score, Precision, Recall, and F1.

        Args:
            threshold: Probability threshold for hard class predictions.

        Returns:
            Dictionary of metrics rounded to 4 decimal places.
        """
        logger.info("ModelEvaluator: calculating metrics on %d samples.", len(self.y_test))

        # Get probabilities and hard predictions
        probs = self.model.predict_proba(self.X_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        metrics = {
            "ROC_AUC": float(roc_auc_score(self.y_test, probs)),
            "Brier_Score": float(brier_score_loss(self.y_test, probs)),
            "Precision": float(precision_score(self.y_test, preds, zero_division=0)),
            "Recall": float(recall_score(self.y_test, preds, zero_division=0)),
            "F1_Score": float(f1_score(self.y_test, preds, zero_division=0)),
        }

        # Round for readability
        rounded_metrics = {k: round(v, 4) for k, v in metrics.items()}
        logger.info("Evaluation results: %s", rounded_metrics)
        return rounded_metrics

    def save_metrics(self, path: str, metrics: Dict[str, float]) -> None:
        """Persist metrics to a JSON file.

        Args:
            path: Target file path (including directory).
            metrics: Dictionary of metrics to save.
        """
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        with open(abs_path, "w") as fh:
            json.dump(metrics, fh, indent=2)

        logger.info("Metrics successfully saved to: %s", abs_path)
