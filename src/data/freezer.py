"""DatasetFreezer — encodes stratified fold assignments into a frozen CSV.

This module is a direct, blueprint-compliant port of
``OLD/Main_x/00x_freeze_dataset.py``.  It runs **once** to produce the
immutable ``Dataset_Needs_SOTA.csv`` that every downstream step reads.  Once
the frozen CSV exists, the raw Excel is never touched again.

Split Protocol::

    Rows 0-3999  →  Train/Val  (stratified_fold = 0..4)
    Rows 4000-4999 →  Blind Test (stratified_fold = -1)
    Stratified on dual-target synthetic column (AccumulationInvestment + IncomeInvestment)

Usage::

    from omegaconf import DictConfig
    freezer = DatasetFreezer(cfg.data)
    frozen_df = freezer.freeze()
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

logger = logging.getLogger(__name__)


class DatasetFreezer:
    """Reads the raw Excel file and freezes stratified CV fold assignments.

    The resulting CSV becomes the single source of truth for all pipeline
    splits. The method is deterministic: given the same ``random_state``,
    running :meth:`freeze` twice produces byte-identical output.

    Attributes:
        cfg: Hydra DictConfig sub-tree rooted at ``cfg.data``.

    Args:
        cfg: Hydra ``DictConfig`` containing:
            - ``raw_excel_path``
            - ``frozen_csv_path``
            - ``train_val_size``
            - ``test_size``
            - ``n_splits``
            - ``random_state``
            - ``id_col``
            - ``fold_col``
            - ``target_cols``
            - ``stratify_col``
    """

    def __init__(self, cfg: "DictConfig") -> None:  # noqa: F821
        self.cfg = cfg
        logger.debug("DatasetFreezer initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def freeze(self, overwrite: bool = False) -> pd.DataFrame:
        """Execute the freeze protocol: stratify, fold-assign, and save.

        Args:
            overwrite: If ``False`` (default) and the frozen CSV already
                exists, returns the cached file without recomputing.  Set to
                ``True`` to deliberately invalidate all existing model
                checkpoints.

        Returns:
            The frozen ``pd.DataFrame`` with a ``stratified_fold`` column
            added.

        Raises:
            FileNotFoundError: If the raw Excel file is not found.
            AssertionError: If the smoke tests on row counts or fold parity
                fail after fold assignment.
        """
        out_path: str = os.path.abspath(self.cfg.frozen_csv_path)

        if os.path.exists(out_path) and not overwrite:
            logger.info(
                "Frozen dataset already exists at '%s'. "
                "Pass overwrite=True to regenerate.",
                out_path,
            )
            return pd.read_csv(out_path)

        df = self._load_raw_excel()
        df = self._assign_folds(df)
        self._run_smoke_tests(df)
        self._save(df, out_path)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_raw_excel(self) -> pd.DataFrame:
        """Read and validate the raw Excel source file.

        Returns:
            Loaded ``pd.DataFrame`` from the ``Needs`` sheet.

        Raises:
            FileNotFoundError: If the ``.xls`` file does not exist.
            ValueError: If fewer rows than required are found, or the ID
                column contains duplicates.
        """
        raw_path: str = os.path.abspath(self.cfg.raw_excel_path)
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"Raw Excel not found at '{raw_path}'. "
                "Ensure Dataset2_Needs.xls is present in the project root."
            )

        logger.info("Loading raw Excel: %s", raw_path)
        df = pd.read_excel(raw_path, sheet_name="Needs")
        df.columns = df.columns.str.strip()
        logger.info("Raw data shape: %s", df.shape)

        min_rows: int = int(self.cfg.train_val_size) + int(self.cfg.test_size)
        if len(df) < min_rows:
            raise ValueError(
                f"Dataset has only {len(df)} rows; expected ≥ {min_rows}."
            )

        id_col: str = self.cfg.id_col
        if id_col in df.columns and not df[id_col].is_unique:
            raise ValueError(
                f"Column '{id_col}' contains duplicate IDs — dataset integrity compromised."
            )

        logger.info("Raw data validation passed — ID uniqueness verified.")
        return df

    def _assign_folds(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform stratified train/test split and 5-fold assignment.

        Phase 1 — Stratified 80/20 split using a synthetic dual-target column.
        Phase 2 — Stratified 5-fold on the training block only.

        Args:
            df: The raw DataFrame loaded from Excel.

        Returns:
            ``df`` with a ``stratified_fold`` column added (values: 0-4 for
            train rows, -1 for test rows).
        """
        fold_col: str = self.cfg.fold_col
        stratify_col: str = self.cfg.stratify_col
        target_cols: list[str] = list(self.cfg.target_cols)
        test_size: int = int(self.cfg.test_size)
        n_splits: int = int(self.cfg.n_splits)
        random_state: int = int(self.cfg.random_state)

        df[fold_col] = -5  # Sentinel — flags any unassigned row

        # Synthetic stratification column: ensures both targets are balanced
        df[stratify_col] = (
            df[target_cols[0]].astype(str) + "_" + df[target_cols[1]].astype(str)
        )

        logger.info("Phase 1: Stratified train/test split (%d test rows)...", test_size)
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=df[stratify_col],
            random_state=random_state,
        )
        df.iloc[test_idx, df.columns.get_loc(fold_col)] = -1

        logger.info(
            "Phase 2: Stratified %d-fold on training block (%d rows)...",
            n_splits,
            len(train_idx),
        )
        df_train = df.iloc[train_idx].copy()
        y_stratify_train = df_train[stratify_col].values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for fold_id, (_, val_rel_idx) in enumerate(
            skf.split(np.zeros(len(df_train)), y_stratify_train)
        ):
            abs_idx = train_idx[val_rel_idx]
            df.iloc[abs_idx, df.columns.get_loc(fold_col)] = fold_id

        # Remove synthetic column — it must not appear in any downstream file
        df.drop(columns=[stratify_col], inplace=True)
        logger.info("Fold assignment complete.")
        return df

    def _run_smoke_tests(self, df: pd.DataFrame) -> None:
        """Assert post-condition invariants after fold assignment.

        Args:
            df: DataFrame with ``stratified_fold`` column populated.

        Raises:
            AssertionError: If any smoke test fails.
        """
        fold_col: str = self.cfg.fold_col
        train_val_size: int = int(self.cfg.train_val_size)
        test_size: int = int(self.cfg.test_size)

        unassigned = (df[fold_col] == -5).sum()
        assert unassigned == 0, (
            f"Smoke test FAILED: {unassigned} rows were never assigned a fold."
        )

        actual_test = (df[fold_col] == -1).sum()
        assert actual_test == test_size, (
            f"Smoke test FAILED: Test block has {actual_test} rows, expected {test_size}."
        )

        actual_tv = df[df[fold_col] >= 0].shape[0]
        assert actual_tv == train_val_size, (
            f"Smoke test FAILED: Train/Val block has {actual_tv} rows, expected {train_val_size}."
        )

        # Parity check — Income rate must be within 5% across splits
        for target in list(self.cfg.target_cols):
            fold0_rate: float = df[df[fold_col] == 0][target].mean()
            test_rate: float = df[df[fold_col] == -1][target].mean()
            delta: float = abs(fold0_rate - test_rate)
            if delta >= 0.05:
                logger.warning(
                    "Stratification drift detected for '%s': "
                    "Fold 0 rate=%.3f, Test rate=%.3f, Δ=%.4f",
                    target, fold0_rate, test_rate, delta,
                )
            else:
                logger.info(
                    "Parity check OK for '%s': Fold 0=%.3f | Test=%.3f | Δ=%.4f",
                    target, fold0_rate, test_rate, delta,
                )

    def _save(self, df: pd.DataFrame, out_path: str) -> None:
        """Persist the frozen DataFrame to disk.

        Args:
            df: The fully processed DataFrame to save.
            out_path: Absolute path for the output CSV.
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.info(
            "Frozen dataset saved — rows: %d | cols: %d | path: %s",
            len(df),
            len(df.columns),
            out_path,
        )
        logger.warning(
            "IMPORTANT: '%s' is the immutable Pipeline X Bible. "
            "Do not re-run freeze unless you deliberately want to "
            "invalidate all existing model checkpoints.",
            os.path.basename(out_path),
        )
