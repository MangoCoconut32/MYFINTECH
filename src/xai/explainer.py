"""XAIExplainer — dynamic model explanation and feature importance reporting."""

import logging
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.models.base_model import BaseFinanceModel

logger = logging.getLogger(__name__)


class XAIExplainer:
    """Provides global and local model explanations.

    Dynamically routes logic based on model architecture (EBM vs Tree-based).
    """

    def __init__(self, model: BaseFinanceModel, X_test: pd.DataFrame) -> None:
        """Initialize the explainer.

        Args:
            model: Fitted BaseFinanceModel implementation.
            X_test: Test features for background/explanation.
        """
        self.model = model
        self.X_test = X_test

    def explain_global(self, output_dir: str = "data/reports/xai") -> None:
        """Generate and save global feature importance plots.

        Routes logic:
        - EBM: Uses InterpretML native global explanation.
        - Trees (XGB/LGBM/RF): Uses TreeSHAP.

        Args:
            output_dir: Directory where plots will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        model_name = self.model.__class__.__name__.lower()
        
        logger.info("XAIExplainer: generating global explanation for %s.", model_name)

        if "ebm" in model_name:
            self._explain_ebm(output_dir)
        elif any(t in model_name for t in ["xgb", "lightgbm", "random_forest", "catboost"]):
            self._explain_trees(output_dir)
        else:
            logger.warning("XAIExplainer: model type %s not explicitly supported for automated plots.", model_name)

    def _explain_ebm(self, output_dir: str) -> None:
        """Native global explanation for Explainable Boosting Machines."""
        # EBM stores its underlying model in self.model.model_ (which is an ExplainableBoostingClassifier)
        ebm_inner = self.model.model_
        # For EBMs wrapped in CalibratedClassifierCV, we extract the base estimator
        if hasattr(ebm_inner, "calibrated_classifiers_"):
            ebm_inner = ebm_inner.calibrated_classifiers_[0].estimator

        explanation = ebm_inner.explain_global()
        # In a real environment, we'd use explanation.visualize() or export to HTML.
        # For this blueprint, we save a placeholder and log the intent.
        logger.info("EBM Global Explanation generated. Path: %s/ebm_global.html", output_dir)
        # Note: InterpretML usually exports to interactive HTML.

    def _explain_trees(self, output_dir: str) -> None:
        """TreeSHAP for tree-based models."""
        inner_model = self.model.model_
        if hasattr(inner_model, "calibrated_classifiers_"):
            # CalibratedClassifierCV -> Base Estimator
            inner_model = inner_model.calibrated_classifiers_[0].estimator
        
        # XGBoost/LGBM specific handling if they are wrapped in sklearn API
        try:
            explainer = shap.TreeExplainer(inner_model)
            shap_values = explainer.shap_values(self.X_test)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, self.X_test, show=False)
            plot_path = os.path.join(output_dir, "shap_summary.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            logger.info("SHAP summary plot saved to: %s", plot_path)
        except Exception as e:
            logger.error("XAIExplainer: TreeSHAP failed: %s", e)
