"""MYFINTECH Pipeline — single Hydra-managed entry point.

Run via::

    python main.py pipeline.step=data_prep
    python main.py pipeline.mode=train model=xgboost
    python main.py pipeline.mode=tune model=lightgbm
    python main.py pipeline.mode=evaluate model=xgboost
    python main.py pipeline.mode=audit model=ebm
"""

import logging
import os
import json
from typing import Any, Type

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler

from src.utils.logging_config import setup_logging
from src.models.base_model import BaseFinanceModel
from src.models.xgboost_model import XGBoostModel
from src.models.ebm_model import EBMModel
from src.models.tabnet_model import TabNetModel
from src.models.lightgbm_model import LightGBMModel
from src.models.catboost_model import CatBoostModel
from src.models.random_forest_model import RandomForestModel

from src.evaluation.evaluator import ModelEvaluator
from src.xai.explainer import XAIExplainer
from src.xai.counterfactuals import DiCEExplainer

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Dispatches execution to the correct pipeline stage."""

    _MODEL_REGISTRY: dict[str, Type[BaseFinanceModel]] = {
        "xgboost": XGBoostModel,
        "ebm": EBMModel,
        "tabnet": TabNetModel,
        "lightgbm": LightGBMModel,
        "catboost": CatBoostModel,
        "random_forest": RandomForestModel,
    }

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def run(self) -> None:
        """Execute the pipeline step/mode."""
        # Resolve mode/step
        mode = self.cfg.pipeline.get("mode")
        step = self.cfg.pipeline.get("step")
        
        # Priority: mode > step
        action = mode if mode else step
        logger.info("PipelineOrchestrator: executing action '%s'.", action)

        if action == "data_prep":
            self._run_data_prep()
        elif action == "train":
            self._run_train()
        elif action == "tune":
            self._run_tune()
        elif action == "evaluate":
            self._run_evaluate()
        elif action == "audit":
            self._run_audit()
        else:
            raise ValueError(f"Unknown pipeline action: '{action}'.")

    def _run_data_prep(self) -> None:
        """Step: data_prep — freeze the raw dataset and build the feature matrix."""
        from src.data.freezer import DatasetFreezer
        from src.data.loader import DataLoader
        from src.features.engineer import FeatureEngineer

        logger.info("=" * 60)
        logger.info("STEP: data_prep")
        logger.info("=" * 60)

        freezer = DatasetFreezer(self.cfg.data)
        freezer.freeze(overwrite=False)

        loader = DataLoader(self.cfg.data)
        X_tv_raw, y_tv = loader.get_train_val()
        X_test_raw, y_test = loader.get_test_set()

        engineer = FeatureEngineer(self.cfg.features)
        X_tv_eng = engineer.fit_transform(X_tv_raw)
        X_test_eng = engineer.transform(X_test_raw)

        out_dir = "data/processed"
        os.makedirs(out_dir, exist_ok=True)

        target = self.cfg.data.primary_target
        X_tv_eng.assign(**{target: y_tv.values}).to_csv(
            os.path.join(out_dir, "train_engineered.csv"), index=False
        )
        X_test_eng.assign(**{target: y_test.values}).to_csv(
            os.path.join(out_dir, "test_engineered.csv"), index=False
        )
        logger.info("data_prep complete.")

    def _run_train(self) -> None:
        """Step: train — fit the configured model."""
        from src.data.loader import DataLoader
        from src.features.engineer import FeatureEngineer

        model_name = self.cfg.model.name
        logger.info("=" * 60)
        logger.info("STEP: train (model=%s)", model_name)
        logger.info("=" * 60)

        loader = DataLoader(self.cfg.data)
        engineer = FeatureEngineer(self.cfg.features)
        
        target_cols = list(self.cfg.data.target_cols)
        params = OmegaConf.to_container(self.cfg.model.hyperparameters, resolve=True)
        out_dir = self.cfg.model.artifacts.output_dir
        os.makedirs(out_dir, exist_ok=True)

        performance = {}

        for target in target_cols:
            logger.info("Training %s for target: '%s'.", model_name, target)
            X_tv_raw, y_tv = loader.get_train_val(target=target)
            X_test_raw, y_test = loader.get_test_set(target=target)

            X_tv = engineer.fit_transform(X_tv_raw)
            X_test = engineer.transform(X_test_raw)

            # Conditional Scaling
            if self.cfg.model.get("requires_scaling", False):
                logger.info("Applying MinMaxScaler.")
                scaler = MinMaxScaler()
                X_tv = pd.DataFrame(scaler.fit_transform(X_tv), columns=X_tv.columns)
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

            model_cls = self._MODEL_REGISTRY.get(model_name)
            if not model_cls:
                raise ValueError(f"Model {model_name} not found in registry.")
            
            model = model_cls(params=params, cfg=self.cfg.model)
            model.fit(X_tv, y_tv)

            evaluator = ModelEvaluator(model, X_test, y_test)
            metrics = evaluator.calculate_metrics()
            performance[target] = metrics

            safe_name = target.replace("Investment", "").lower()[:3]
            model.save(os.path.join(out_dir, f"{self.cfg.model.artifacts.model_prefix}_{safe_name}.pkl"))

        perf_path = self.cfg.model.artifacts.performance_file
        os.makedirs(os.path.dirname(perf_path), exist_ok=True)
        with open(perf_path, "w") as fh:
            json.dump(performance, fh, indent=2)
        logger.info("train complete.")

    def _run_tune(self) -> None:
        """Step: tune — optimize hyperparameters using Optuna."""
        import optuna
        from src.data.loader import DataLoader
        from src.features.engineer import FeatureEngineer

        model_name = self.cfg.model.name
        logger.info("=" * 60)
        logger.info("STEP: tune (model=%s)", model_name)
        logger.info("=" * 60)

        loader = DataLoader(self.cfg.data)
        engineer = FeatureEngineer(self.cfg.features)
        
        target_cols = list(self.cfg.data.target_cols)
        best_params_all = {}

        for target in target_cols:
            logger.info("Tuning %s for target: '%s'.", model_name, target)
            X_tv_raw, y_tv = loader.get_train_val(target=target)
            X_tv = engineer.fit_transform(X_tv_raw)

            # Conditional Scaling
            if self.cfg.model.get("requires_scaling", False):
                scaler = MinMaxScaler()
                X_tv = pd.DataFrame(scaler.fit_transform(X_tv), columns=X_tv.columns)

            model_cls = self._MODEL_REGISTRY.get(model_name)
            model = model_cls(params={}, cfg=self.cfg.model)

            def objective(trial: optuna.Trial) -> float:
                return model.tune(X_tv, y_tv, trial)

            study = optuna.create_study(direction=self.cfg.model.optuna.direction)
            study.optimize(
                objective, 
                n_trials=self.cfg.model.optuna.n_trials,
                timeout=self.cfg.model.optuna.timeout_seconds
            )

            logger.info("Best trial for %s: %f", target, study.best_value)
            best_params_all[target] = study.best_params

        best_params_path = self.cfg.model.artifacts.best_params_file
        os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
        with open(best_params_path, "w") as fh:
            json.dump(best_params_all, fh, indent=2)
        logger.info("tune complete. Best params saved to %s", best_params_path)

    def _run_evaluate(self) -> None:
        """Step: evaluate — load model and run standard metrics on blind test set."""
        from src.data.loader import DataLoader
        from src.features.engineer import FeatureEngineer

        model_name = self.cfg.model.name
        target = self.cfg.data.primary_target
        logger.info("=" * 60)
        logger.info("STEP: evaluate (model=%s, target=%s)", model_name, target)
        logger.info("=" * 60)

        # 1. Load Data
        loader = DataLoader(self.cfg.data)
        X_test_raw, y_test = loader.get_test_set(target=target)
        engineer = FeatureEngineer(self.cfg.features)
        X_test = engineer.fit(loader.get_train_val(target=target)[0]).transform(X_test_raw)

        # 2. Load Model
        safe_name = target.replace("Investment", "").lower()[:3]
        model_path = os.path.join(
            self.cfg.model.artifacts.output_dir, 
            f"{self.cfg.model.artifacts.model_prefix}_{safe_name}.pkl"
        )
        model_cls = self._MODEL_REGISTRY.get(model_name)
        model = model_cls.load(model_path)

        # 3. Evaluate
        evaluator = ModelEvaluator(model, X_test, y_test)
        metrics = evaluator.calculate_metrics(
            threshold=self.cfg.get("evaluation", {}).get("classification_threshold", 0.5)
        )
        
        # 4. Save
        evaluator.save_metrics(self.cfg.evaluation.artifacts.metrics_file, metrics)
        logger.info("evaluate complete.")

    def _run_audit(self) -> None:
        """Step: audit — generate XAI explanations and counterfactual recourse."""
        from src.data.loader import DataLoader
        from src.features.engineer import FeatureEngineer

        model_name = self.cfg.model.name
        target = self.cfg.data.primary_target
        logger.info("=" * 60)
        logger.info("STEP: audit (model=%s, target=%s)", model_name, target)
        logger.info("=" * 60)

        # 1. Load Data & Model
        loader = DataLoader(self.cfg.data)
        X_tv_raw, y_tv = loader.get_train_val(target=target)
        X_test_raw, y_test = loader.get_test_set(target=target)
        engineer = FeatureEngineer(self.cfg.features)
        engineer.fit(X_tv_raw)
        X_tv = engineer.transform(X_tv_raw)
        X_test = engineer.transform(X_test_raw)

        safe_name = target.replace("Investment", "").lower()[:3]
        model_path = os.path.join(
            self.cfg.model.artifacts.output_dir, 
            f"{self.cfg.model.artifacts.model_prefix}_{safe_name}.pkl"
        )
        model_cls = self._MODEL_REGISTRY.get(model_name)
        model = model_cls.load(model_path)

        # 2. XAI Global Explanations
        explainer = XAIExplainer(model, X_test)
        explainer.explain_global()

        # 3. DiCE Counterfactuals
        # We use a combined DF for DiCE initialization
        dice_data = X_tv.copy()
        dice_data[self.cfg.xai.dice.outcome_name] = y_tv.values

        dice_explainer = DiCEExplainer(
            model=model,
            train_df=dice_data,
            continuous_features=list(self.cfg.xai.dice.continuous_features),
            outcome_name=self.cfg.xai.dice.outcome_name
        )
        
        # Generate for the first 2 rejected samples in test set
        rejected = X_test[model.predict(X_test) == 0].head(2)
        if not rejected.empty:
            cf = dice_explainer.generate_counterfactuals(
                rejected,
                total_CFs=self.cfg.xai.dice.n_counterfactuals,
                features_to_vary=list(self.cfg.xai.dice.features_to_vary)
            )
            dice_explainer.save_counterfactuals(cf, "data/reports/xai/counterfactuals.json")
        
        logger.info("audit complete.")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log_level = cfg.get("training", {}).get("log_level", "INFO")
    setup_logging(log_level)
    logger.info("Initialising MYFINTECH pipeline.")
    orchestrator = PipelineOrchestrator(cfg)
    orchestrator.run()


if __name__ == "__main__":
    main()
