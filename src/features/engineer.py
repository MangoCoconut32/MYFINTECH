"""FeatureEngineer — anti-leakage feature transformer for the MYFINTECH pipeline.

Fuses two feature-engineering strategies into a single sklearn-style
``fit/transform`` interface:

* **Anima Alois** (domain-knowledge ratios): ``PipelineXTransformer``, ported
  verbatim from ``OLD/Main_x/utilsx.py``.
* **Anima MOA** (brute-force Deep Feature Synthesis): optional, disabled by
  default via the ``features.dfs.enabled`` config key.

Anti-leakage guarantee
----------------------
All statistics used for imputation and clipping (medians, P99 quantiles) are
**fitted exclusively on the training block** and subsequently applied to the
validation/test block without re-fitting.

Usage::

    from omegaconf import DictConfig
    engineer = FeatureEngineer(cfg.features)
    engineer.fit(X_train_df)
    X_train_eng = engineer.transform(X_train_df)
    X_test_eng  = engineer.transform(X_test_df)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — mirror configs/features/features.yaml
# ---------------------------------------------------------------------------
_ALOIS_FEATURES: list[str] = [
    "Wealth_log",
    "Income_log",
    "Wealth_Age_Ratio_log",
    "Wealth_per_person",
    "Income_per_person",
    "Income_Wealth_Ratio_log",
    "Age_bracket_Young",
    "Age_bracket_Mid",
    "Age_bracket_Senior",
]

_AGE_BRACKET_DUMMIES: list[str] = [
    "Age_bracket_Young",
    "Age_bracket_Mid",
    "Age_bracket_Senior",
]


# ===========================================================================
# PipelineXTransformer — the anti-leakage core (ported from utilsx.py)
# ===========================================================================
class PipelineXTransformer:
    """Fits clipping/imputation statistics on training data; transforms any split.

    This class is the anti-leakage shield: statistics are computed exclusively
    on the training block passed to :meth:`fit`, then applied to both the
    training and test blocks via :meth:`transform`.

    Attributes:
        medians_: Median values per numeric column, computed on the training
            block.
        p99_inc_: 99th-percentile clip threshold for the ``Income`` column.
        p99_wth_: 99th-percentile clip threshold for the ``Wealth`` column.
        inc_max_: Maximum income value in the training block (used to impute
            the ``Income / Wealth`` ratio for zero-wealth rows).
        is_fitted_: ``True`` after :meth:`fit` has been called.

    Args:
        cfg: Optional Hydra ``DictConfig`` sub-tree (``cfg.features``).
            Currently unused but retained for future config-driven extension.
    """

    def __init__(self, cfg: Optional["DictConfig"] = None) -> None:  # noqa: F821
        self.medians_: Optional[pd.Series] = None
        self.p99_inc_: Optional[float] = None
        self.p99_wth_: Optional[float] = None
        self.inc_max_: Optional[float] = None
        self.is_fitted_: bool = False
        self._cfg = cfg

    def fit(self, df_train: pd.DataFrame) -> "PipelineXTransformer":
        """Calculate clip thresholds and imputation medians from the training block.

        Args:
            df_train: The training split only — NEVER pass the full dataset.

        Returns:
            ``self``, enabling method chaining (``transformer.fit(X).transform(X)``).

        Raises:
            KeyError: If ``Income`` or ``Wealth`` columns are missing.
        """
        df = df_train.copy()
        self.p99_inc_ = float(df["Income"].quantile(0.99))
        self.p99_wth_ = float(df["Wealth"].quantile(0.99))
        self.inc_max_ = float(df["Income"].max())
        self.medians_ = df.median(numeric_only=True)
        self.is_fitted_ = True
        logger.debug(
            "PipelineXTransformer fitted — p99_inc=%.2f, p99_wth=%.2f, inc_max=%.2f",
            self.p99_inc_, self.p99_wth_, self.inc_max_,
        )
        return self

    def transform(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted statistics to engineer the Alois feature set.

        Computes 9 new columns (log transforms, per-capita ratios, age
        brackets) while preserving all existing raw columns.

        Args:
            df_in: Any split of the dataset (train, val, or test).

        Returns:
            A new ``pd.DataFrame`` with the 9 Alois features appended.

        Raises:
            RuntimeError: If :meth:`fit` has not been called first.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "PipelineXTransformer must be fitted before calling transform(). "
                "Call .fit(X_train) first."
            )

        df = df_in.copy()

        # 1. Median imputation — before any derived feature to avoid NaN propagation
        df.fillna(self.medians_, inplace=True)

        # 2. Age brackets (non-linear life-cycle effect)
        df["Age_bracket"] = pd.cut(
            df["Age"],
            bins=[17, 35, 55, 100],
            labels=["Young", "Mid", "Senior"],
        )
        dummies: pd.DataFrame = pd.get_dummies(
            df["Age_bracket"], prefix="Age_bracket", drop_first=False, dtype=int
        )
        # Guard against categories missing in the batch
        for label in _AGE_BRACKET_DUMMIES:
            if label not in dummies.columns:
                dummies[label] = 0

        df = pd.concat(
            [df.drop(columns=["Age_bracket"]), dummies[_AGE_BRACKET_DUMMIES]],
            axis=1,
        )

        # 3. Clip before ratio computation — prevents extreme outliers from dominating
        clipped_inc: pd.Series = df["Income"].clip(upper=self.p99_inc_)
        clipped_wth: pd.Series = df["Wealth"].clip(upper=self.p99_wth_)

        # 4. Log transforms (reduce right-skew of financial distributions)
        df["Wealth_log"] = np.log1p(df["Wealth"])
        df["Income_log"] = np.log1p(df["Income"])

        # 5. Wealth accumulation speed (log of wealth per adult year)
        adult_years: pd.Series = (df["Age"] - 17).clip(lower=1)
        df["Wealth_Age_Ratio_log"] = np.log1p(clipped_wth / adult_years)

        # 6. Per-capita metrics (divide by safe FamilyMembers)
        safe_fm: pd.Series = (
            df["FamilyMembers"]
            .replace(0, np.nan)
            .fillna(self.medians_.get("FamilyMembers", 1))
        )
        df["Wealth_per_person"] = clipped_wth / safe_fm
        df["Income_per_person"] = clipped_inc / safe_fm

        # 7. Income-to-Wealth ratio (life-cycle proxy; log-compressed)
        safe_wth: pd.Series = clipped_wth.replace(0, np.nan)
        raw_ratio: pd.Series = clipped_inc.div(safe_wth).fillna(self.inc_max_)
        df["Income_Wealth_Ratio_log"] = np.log1p(raw_ratio)

        logger.debug(
            "PipelineXTransformer: transformed %d rows → %d columns.",
            len(df), len(df.columns),
        )
        return df

    def get_params(self) -> dict:
        """Return fitted parameters for inspection or production export.

        Returns:
            A dictionary containing ``medians``, ``p99_inc``, ``p99_wth``,
            and ``inc_max``.
        """
        return {
            "medians": self.medians_.to_dict() if self.medians_ is not None else None,
            "p99_inc": self.p99_inc_,
            "p99_wth": self.p99_wth_,
            "inc_max": self.inc_max_,
        }


# ===========================================================================
# FeatureEngineer — high-level orchestrator exposed to main.py
# ===========================================================================
class FeatureEngineer:
    """Orchestrates the full feature-engineering pipeline.

    Wraps :class:`PipelineXTransformer` (Alois features) and optionally
    applies Deep Feature Synthesis (Anima MOA) when ``cfg.features.dfs.enabled``
    is ``True``.

    The class follows the sklearn ``fit/transform`` convention so it can be
    dropped into ``Pipeline`` objects or called standalone.

    Attributes:
        cfg: Hydra DictConfig sub-tree rooted at ``cfg.features``.
        transformer_: The fitted :class:`PipelineXTransformer` instance.
        feature_cols_: List of output column names after ``fit``.
        is_fitted_: ``True`` after :meth:`fit` has been called.

    Args:
        cfg: Hydra ``DictConfig`` containing:
            - ``base_cols``
            - ``alois_engineered``
            - ``dfs.enabled``
            - ``dfs.top_n``
            - ``corr_threshold``
    """

    def __init__(self, cfg: "DictConfig") -> None:  # noqa: F821
        self.cfg = cfg
        self.transformer_: PipelineXTransformer = PipelineXTransformer(cfg)
        self.feature_cols_: list[str] = []
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Fit the transformer on the training block.

        Args:
            df: The 4 000-row train/validation block. Must NOT include test
                rows to prevent leakage.

        Returns:
            ``self`` for method chaining.
        """
        logger.info("FeatureEngineer: fitting on %d rows...", len(df))
        self.transformer_.fit(df)

        # Determine canonical output column order after transform
        sample = self.transformer_.transform(df.head(1))
        base: list[str] = list(self.cfg.base_cols)
        alois: list[str] = list(self.cfg.alois_engineered)
        # Only keep columns that actually exist in the transformed output
        self.feature_cols_ = [c for c in base + alois if c in sample.columns]

        self.is_fitted_ = True
        logger.info(
            "FeatureEngineer: fitted — %d output features: %s",
            len(self.feature_cols_),
            self.feature_cols_,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformer and return the canonical feature matrix.

        Args:
            df: Any split of the dataset (train, val, or test).  Must contain
                the raw base columns.

        Returns:
            A ``pd.DataFrame`` containing exactly ``self.feature_cols_`` columns
            with no ID, fold, or target columns.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "FeatureEngineer must be fitted before calling transform(). "
                "Call .fit(X_train) first."
            )

        engineered: pd.DataFrame = self.transformer_.transform(df)

        # Select only the canonical feature set; drop missing columns gracefully
        available: list[str] = [c for c in self.feature_cols_ if c in engineered.columns]
        missing: list[str] = [c for c in self.feature_cols_ if c not in engineered.columns]
        if missing:
            logger.warning(
                "FeatureEngineer: %d expected feature(s) missing after transform: %s",
                len(missing), missing,
            )

        result: pd.DataFrame = engineered[available].copy()
        logger.debug(
            "FeatureEngineer.transform: %d rows × %d features returned.",
            len(result), len(result.columns),
        )
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``df`` then immediately transform it.

        Equivalent to calling ``fit(df).transform(df)``.  Intended for use on
        the training block only.

        Args:
            df: Training block DataFrame.

        Returns:
            Transformed feature matrix for the same rows.
        """
        return self.fit(df).transform(df)

    @property
    def n_features_out(self) -> int:
        """Number of output features after fitting.

        Returns:
            Integer count of output features, or 0 if not yet fitted.
        """
        return len(self.feature_cols_)
