"""DataLoader — single-responsibility class for loading the MYFINTECH dataset.

The loader reads ``Dataset2_Needs_DFS.csv`` (or the raw Excel file) and
exposes the frozen fold-based train/val/test splits that are the foundation
of the anti-leakage protocol.

Usage::

    from omegaconf import DictConfig
    loader = DataLoader(cfg.data)
    df_full = loader.load()
    X_tv, y_tv, X_test, y_test = loader.get_splits(target="IncomeInvestment")
"""

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — kept in sync with configs/data/data.yaml defaults
# ---------------------------------------------------------------------------
_DEFAULT_FOLD_COL: str = "stratified_fold"
_DEFAULT_ID_COL: str = "ID"
_TARGET_COLS: list[str] = ["AccumulationInvestment", "IncomeInvestment"]


class DataLoader:
    """Loads and validates the MYFINTECH dataset from CSV or Excel.

    Reads the pre-built DFS CSV by default. Falls back to the raw Excel file
    if explicitly configured. Exposes the frozen stratified-fold splits so
    that every downstream module uses identical train/val/test boundaries.

    Attributes:
        cfg: Hydra DictConfig sub-tree rooted at ``cfg.data``.
        _df: Cached full DataFrame (loaded lazily on first call to
            :meth:`load`).

    Args:
        cfg: Hydra ``DictConfig`` containing at minimum:
            - ``dfs_csv_path``  – path to the pre-built DFS CSV.
            - ``raw_excel_path`` – path to the original ``.xls`` file.
            - ``frozen_csv_path`` – path to the frozen SOTA CSV.
            - ``id_col``
            - ``fold_col``
            - ``target_cols``
    """

    def __init__(self, cfg: "DictConfig") -> None:  # noqa: F821
        self.cfg = cfg
        self._df: Optional[pd.DataFrame] = None
        logger.debug("DataLoader initialised with config keys: %s", list(cfg.keys()))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, use_frozen: bool = True) -> pd.DataFrame:
        """Load the full dataset (all 5000 rows) into memory.

        The call is idempotent — subsequent calls return the cached DataFrame
        without re-reading disk.

        Args:
            use_frozen: If ``True`` (default), reads the frozen SOTA CSV
                (``frozen_csv_path``) which carries the ``stratified_fold``
                column. If ``False``, reads the DFS CSV (``dfs_csv_path``).

        Returns:
            A ``pd.DataFrame`` with shape ``(5000, n_features + n_targets + 1)``
            containing the ID column, all feature columns, both target columns,
            and the ``stratified_fold`` column.

        Raises:
            FileNotFoundError: If the requested file does not exist on disk.
            ValueError: If the dataset fails integrity checks (duplicate IDs,
                missing target columns, unexpected number of rows).
        """
        if self._df is not None:
            logger.debug("Returning cached DataFrame (%d rows).", len(self._df))
            return self._df

        path = self._resolve_path(use_frozen)
        logger.info("Loading dataset from: %s", path)

        self._df = self._read_file(path)
        self._validate(self._df)

        logger.info(
            "Dataset loaded successfully — shape: %s", self._df.shape
        )
        return self._df

    def get_train_val(
        self,
        target: Optional[str] = None,
        use_frozen: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Return the 4 000-row train/validation block (``stratified_fold >= 0``).

        Args:
            target: Target column name. Defaults to ``cfg.data.primary_target``.
            use_frozen: Passed through to :meth:`load`.

        Returns:
            A tuple ``(X_tv, y_tv)`` where ``X_tv`` is the feature matrix and
            ``y_tv`` is the binary label Series for the requested target.

        Raises:
            KeyError: If ``target`` is not a column in the dataset.
        """
        target = target or self.cfg.primary_target
        df = self.load(use_frozen=use_frozen)
        mask: pd.Series = df[self.cfg.fold_col] >= 0
        return self._split_xy(df[mask], target)

    def get_test_set(
        self,
        target: Optional[str] = None,
        use_frozen: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Return the 1 000-row blind test block (``stratified_fold == -1``).

        Args:
            target: Target column name. Defaults to ``cfg.data.primary_target``.
            use_frozen: Passed through to :meth:`load`.

        Returns:
            A tuple ``(X_test, y_test)`` where ``X_test`` is the feature matrix
            and ``y_test`` is the binary label Series for the requested target.
        """
        target = target or self.cfg.primary_target
        df = self.load(use_frozen=use_frozen)
        mask: pd.Series = df[self.cfg.fold_col] == -1
        return self._split_xy(df[mask], target)

    def get_splits(
        self,
        target: Optional[str] = None,
        use_frozen: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Convenience wrapper returning all four split components.

        Args:
            target: Target column name. Defaults to ``cfg.data.primary_target``.
            use_frozen: Passed through to :meth:`load`.

        Returns:
            A 4-tuple ``(X_tv, y_tv, X_test, y_test)``.
        """
        X_tv, y_tv = self.get_train_val(target=target, use_frozen=use_frozen)
        X_test, y_test = self.get_test_set(target=target, use_frozen=use_frozen)
        logger.info(
            "Splits ready — Train/Val: %d rows | Test: %d rows",
            len(X_tv),
            len(X_test),
        )
        return X_tv, y_tv, X_test, y_test

    def get_fold(
        self,
        fold_id: int,
        target: Optional[str] = None,
        use_frozen: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Return a single CV fold for walk-forward validation.

        Args:
            fold_id: Integer in ``[0, n_splits)``. Rows whose
                ``stratified_fold == fold_id`` become the validation set; all
                other train/val rows become the training set.
            target: Target column name. Defaults to ``cfg.data.primary_target``.
            use_frozen: Passed through to :meth:`load`.

        Returns:
            A 4-tuple ``(X_train, y_train, X_val, y_val)``.

        Raises:
            ValueError: If ``fold_id`` is outside the valid range.
        """
        n_splits: int = int(self.cfg.n_splits)
        if fold_id not in range(n_splits):
            raise ValueError(
                f"fold_id must be in [0, {n_splits}), got {fold_id}."
            )
        target = target or self.cfg.primary_target
        df = self.load(use_frozen=use_frozen)
        fold_col: str = self.cfg.fold_col

        train_mask = (df[fold_col] >= 0) & (df[fold_col] != fold_id)
        val_mask = df[fold_col] == fold_id

        X_train, y_train = self._split_xy(df[train_mask], target)
        X_val, y_val = self._split_xy(df[val_mask], target)
        logger.debug(
            "Fold %d — train: %d | val: %d", fold_id, len(X_train), len(X_val)
        )
        return X_train, y_train, X_val, y_val

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, use_frozen: bool) -> str:
        """Resolve the file path from the config, checking existence.

        Args:
            use_frozen: If ``True``, targets the frozen SOTA CSV; otherwise
                targets the DFS CSV.

        Returns:
            Absolute path string to the dataset file.

        Raises:
            FileNotFoundError: If the resolved path does not exist.
        """
        raw_path: str = (
            self.cfg.frozen_csv_path if use_frozen else self.cfg.dfs_csv_path
        )
        # Paths in config are relative to project root; resolve from CWD
        abs_path: str = os.path.abspath(raw_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"Dataset file not found: {abs_path}\n"
                "Run `python main.py pipeline.step=data_prep` first to freeze the dataset."
            )
        return abs_path

    @staticmethod
    def _read_file(path: str) -> pd.DataFrame:
        """Read a CSV or Excel file into a DataFrame.

        Args:
            path: Absolute path to a ``.csv`` or ``.xls``/``.xlsx`` file.

        Returns:
            ``pd.DataFrame`` with column names stripped of whitespace.

        Raises:
            ValueError: If the file extension is not supported.
        """
        ext: str = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in {".xls", ".xlsx"}:
            df = pd.read_excel(path, sheet_name="Needs")
        else:
            raise ValueError(f"Unsupported file format: '{ext}'. Use .csv or .xls/.xlsx.")
        df.columns = df.columns.str.strip()
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """Run integrity checks on the loaded DataFrame.

        Args:
            df: The fully loaded dataset.

        Raises:
            ValueError: On ID duplicates, missing target columns, or if the
                fold column is absent (when ``use_frozen=True``).
        """
        id_col: str = self.cfg.id_col
        fold_col: str = self.cfg.fold_col
        target_cols: list[str] = list(self.cfg.target_cols)

        # ID uniqueness
        if id_col in df.columns and not df[id_col].is_unique:
            raise ValueError(
                f"Integrity check failed: column '{id_col}' contains duplicate values."
            )

        # Target columns present
        missing_targets = [c for c in target_cols if c not in df.columns]
        if missing_targets:
            raise ValueError(
                f"Target columns missing from dataset: {missing_targets}"
            )

        # Fold column present
        if fold_col not in df.columns:
            logger.warning(
                "Column '%s' not found. Data has not been frozen yet. "
                "Run pipeline.step=data_prep to generate fold assignments.",
                fold_col,
            )

        logger.debug("Validation passed — %d rows, %d columns.", *df.shape)

    def _split_xy(
        self, df: pd.DataFrame, target: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Separate features from the target column.

        Drops the ID, fold, and all target columns from X so that no label
        information leaks into the feature matrix.

        Args:
            df: A slice of the full dataset (e.g. train/val block).
            target: The column name of the label to extract as ``y``.

        Returns:
            A tuple ``(X, y)`` where ``X`` is the feature-only DataFrame and
            ``y`` is the target Series.

        Raises:
            KeyError: If ``target`` is not a column in ``df``.
        """
        drop_cols: list[str] = (
            [self.cfg.id_col, self.cfg.fold_col] + list(self.cfg.target_cols)
        )
        drop_cols = [c for c in drop_cols if c in df.columns]

        X: pd.DataFrame = df.drop(columns=drop_cols).copy()
        y: pd.Series = df[target].astype(int).copy()
        return X, y
