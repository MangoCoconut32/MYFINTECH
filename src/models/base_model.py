"""Abstract base class for all finance models in the MYFINTECH pipeline.

Every model family (XGBoost, EBM, TabNet, TwoTower, …) must inherit from
:class:`BaseFinanceModel` and implement all abstract methods.  This contract
guarantees a uniform interface throughout the orchestration layer and makes
it trivially easy to swap models at the ``cfg.model.name`` level.

Design Principles
-----------------
* ``fit`` / ``predict`` / ``predict_proba`` mirror the sklearn API.
* ``save`` / ``load`` use ``pickle`` by default; subclasses may override for
  framework-specific serialisation (e.g. PyTorch ``state_dict``).
* The constructor signature always accepts a plain Python ``dict`` of
  hyperparameters so that Hydra configs are passed through directly.
"""

import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseFinanceModel(ABC):
    """Abstract interface that every MYFINTECH model must satisfy.

    Subclasses receive their hyperparameters through ``params`` so that the
    Hydra config dictionary (``OmegaConf.to_container(cfg.model.hyperparameters)``)
    can be forwarded without modification.

    Attributes:
        params: Model hyperparameters forwarded from the Hydra config.
        model_: The underlying fitted model object (``None`` before training).
        is_fitted_: ``True`` after a successful call to :meth:`fit`.

    Args:
        params: Dictionary of hyperparameters.  Each subclass documents which
            keys it consumes.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self.params: dict[str, Any] = params
        self.model_: Optional[Any] = None
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Abstract methods — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "BaseFinanceModel":
        """Train the model on the given data.

        Args:
            X_train: Feature matrix for the training set.
            y_train: Binary target labels for the training set.
            X_val: Optional validation feature matrix (used for early stopping
                or calibration in applicable subclasses).
            y_val: Optional validation labels.

        Returns:
            ``self`` to allow method chaining.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard class predictions (0 or 1).

        Args:
            X: Feature matrix to run inference on.

        Returns:
            1-D integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities.

        Args:
            X: Feature matrix to run inference on.

        Returns:
            2-D float array of shape ``(n_samples, 2)`` where column 1
            contains the probability of the positive class.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """

    @abstractmethod
    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        trial: Any,
    ) -> float:
        """Evaluate a single Optuna trial using 5-fold Stratified Cross-Validation.

        Args:
            X_train: Full engineered training feature matrix.
            y_train: Target labels for the training set.
            trial: An Optuna ``Trial`` object.

        Returns:
            The mean Out-Of-Fold metric (e.g. ROC AUC).
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the fitted model to disk.

        Args:
            path: Absolute or relative file path (including extension).

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """

    # ------------------------------------------------------------------
    # Concrete methods — shared utilities available to all subclasses
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> "BaseFinanceModel":
        """Deserialise a model from disk.

        Uses ``pickle`` by default.  Subclasses that serialise differently
        (e.g. PyTorch, ONNX) should override this method.

        Args:
            path: Absolute or relative path to the serialised model file.

        Returns:
            The deserialised model object.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        abs_path: str = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Model checkpoint not found at '{abs_path}'.")
        logger.info("Loading model from: %s", abs_path)
        with open(abs_path, "rb") as fh:
            model = pickle.load(fh)
        logger.info("Model loaded successfully.")
        return model

    def _check_is_fitted(self) -> None:
        """Raise ``RuntimeError`` if the model has not been fitted.

        This helper is intended to be called at the start of :meth:`predict`
        and :meth:`predict_proba` in every subclass implementation.

        Raises:
            RuntimeError: If ``self.is_fitted_`` is ``False``.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call .fit() before predict() or predict_proba()."
            )

    def __repr__(self) -> str:
        status: str = "fitted" if self.is_fitted_ else "unfitted"
        return f"{self.__class__.__name__}(status={status}, params={self.params})"
