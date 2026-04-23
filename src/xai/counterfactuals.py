"""DiCEExplainer — generate counterfactual examples for model recourse."""

import logging
import os
from typing import Any, List, Optional

import dice_ml
import pandas as pd

from src.models.base_model import BaseFinanceModel

logger = logging.getLogger(__name__)


class DiCEExplainer:
    """Generates counterfactual examples for binary classification models.

    Used to provide actionable recourse (e.g., 'If your Income was $5k higher, 
    your application would have been approved').
    """

    def __init__(
        self,
        model: BaseFinanceModel,
        train_df: pd.DataFrame,
        continuous_features: List[str],
        outcome_name: str = "target",
    ) -> None:
        """Initialize DiCE.

        Args:
            model: Fitted BaseFinanceModel implementation.
            train_df: Training data (raw/engineered) for DiCE to sample from.
            continuous_features: List of column names to treat as continuous.
            outcome_name: Name of the binary target column.
        """
        self.model = model
        
        # DiCE needs a Data object and a Model object
        self.d = dice_ml.Data(
            dataframe=train_df,
            continuous_features=continuous_features,
            outcome_name=outcome_name
        )
        
        # Wrap the BaseFinanceModel for DiCE
        # DiCE expects a predict_proba that returns probabilities
        self.m = dice_ml.Model(model=self.model, backend="sklearn")
        
        # Initialize the explainer
        self.exp = dice_ml.Dice(self.d, self.m, method="random")

    def generate_counterfactuals(
        self,
        query_instances: pd.DataFrame,
        total_CFs: int = 3,
        desired_class: str = "opposite",
        features_to_vary: Optional[List[str]] = None,
    ) -> Any:
        """Generate counterfactuals for specific query instances.

        Args:
            query_instances: DataFrame containing instances to explain.
            total_CFs: Number of counterfactuals to generate per instance.
            desired_class: "opposite", 0, or 1.
            features_to_vary: List of features DiCE is allowed to change.

        Returns:
            DiCE Counterfactual object.
        """
        logger.info("DiCEExplainer: generating %d counterfactuals...", total_CFs)
        
        dice_exp = self.exp.generate_counterfactuals(
            query_instances,
            total_CFs=total_CFs,
            desired_class=desired_class,
            features_to_vary=features_to_vary or "all"
        )
        
        return dice_exp

    def save_counterfactuals(self, dice_exp: Any, path: str) -> None:
        """Save counterfactuals to a CSV or JSON file.

        Args:
            dice_exp: The counterfactual object returned by generate_counterfactuals.
            path: Target file path.
        """
        # DiCE objects can be converted to JSON/DataFrames
        logger.info("DiCEExplainer: saving counterfactuals to %s", path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dice_exp.to_json(file_name=path)
