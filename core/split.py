"""
train_pipeline.py
------------------
Trains and evaluates the pneumonia/clinical-outcome classifier.

Renamed from "split.py" because it now does considerably more than split
data: preprocessing, model selection via cross-validation, held-out
evaluation, and artifact/metadata persistence. A few correctness issues
in the original version are fixed here rather than just restyled:

- The RandomForest had no `random_state`, so re-running the script could
  silently produce a different model each time.
- `StandardScaler` was fit and used for Logistic Regression, then
  discarded -- the RandomForest was trained on *unscaled* data, and only
  the RandomForest was the model actually saved. Whatever scaling
  relationship existed during training is gone at inference time.
  Preprocessing and the model now travel together in a single
  `Pipeline`, so the saved artifact is self-contained and can't drift
  out of sync with what it expects as input.
- A single 80/20 split is a noisy way to pick between two models. Model
  selection now uses stratified k-fold cross-validation on the training
  set; the held-out test set is only touched once, for final evaluation
  of the winner.
- Accuracy alone is a poor metric for a clinical classifier, especially
  if the classes are imbalanced. ROC-AUC, precision, recall, F1, and a
  confusion matrix are all reported.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("train_pipeline")

RANDOM_STATE = 42


@dataclass
class TrainConfig:
    data_path: Path = Path("data.csv")
    target_column: str = "target"
    output_dir: Path = Path("./artifacts")
    test_size: float = 0.2
    cv_folds: int = 5


# --------------------------------------------------------------------------- #
# Data loading & preprocessing
# --------------------------------------------------------------------------- #
def load_data(config: TrainConfig) -> tuple[pd.DataFrame, pd.Series]:
    if not config.data_path.exists():
        raise FileNotFoundError(f"Data file not found at '{config.data_path}'.")

    df = pd.read_csv(config.data_path)
    if config.target_column not in df.columns:
        raise ValueError(
            f"Target column '{config.target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[config.target_column])
    y = df[config.target_column]

    class_counts = y.value_counts(normalize=True)
    logger.info("Class distribution:\n%s", class_counts.to_string())
    if class_counts.min() < 0.2:
        logger.warning(
            "Classes are imbalanced (minority class = %.1f%% of data). "
            "Consider class_weight='balanced' or resampling.",
            class_counts.min() * 100,
        )

    return X, y


def build_preprocessing(X: pd.DataFrame) -> ColumnTransformer:
    """Numeric columns get imputed + scaled; categorical columns get
    imputed + one-hot encoded. Built generically so this script keeps
    working if the dataset's columns change."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )


# --------------------------------------------------------------------------- #
# Model selection & evaluation
# --------------------------------------------------------------------------- #
def candidate_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
        ),
    }


def select_best_model(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
) -> tuple[str, Pipeline]:
    """Cross-validate each candidate on the training set only, and return
    the pipeline (preprocessing + estimator) for the best-scoring one."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, float] = {}

    for name, estimator in candidate_models().items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc_ovr")
        results[name] = float(scores.mean())
        logger.info("%-20s CV ROC-AUC: %.4f (+/- %.4f)", name, scores.mean(), scores.std())

    best_name = max(results, key=results.get)
    logger.info("Selected model: %s (CV ROC-AUC = %.4f)", best_name, results[best_name])

    best_pipeline = Pipeline([("preprocess", preprocessor), ("model", candidate_models()[best_name])])
    return best_name, best_pipeline


def evaluate_on_test(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test, y_pred).tolist()

    try:
        y_proba = pipeline.predict_proba(X_test)
        if y_proba.shape[1] == 2:
            roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
    except (AttributeError, ValueError) as exc:
        logger.warning("Could not compute ROC-AUC on test set: %s", exc)
        roc_auc = None

    logger.info("\n%s", classification_report(y_test, y_pred))
    if roc_auc is not None:
        logger.info("Test ROC-AUC: %.4f", roc_auc)

    return {"classification_report": report, "confusion_matrix": matrix, "roc_auc": roc_auc}


def extract_feature_importance(pipeline: Pipeline) -> dict[str, float] | None:
    """Best-effort feature importance for interpretability -- useful
    context for a physician reviewing why the model flagged a case."""
    model = pipeline.named_steps["model"]
    try:
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_[0])
    else:
        return None

    ranked = sorted(zip(feature_names, importances), key=lambda pair: pair[1], reverse=True)
    return {name: float(score) for name, score in ranked[:15]}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def train(config: TrainConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(config)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=RANDOM_STATE
    )
    logger.info("X_train shape: %s | X_test shape: %s", X_train.shape, X_test.shape)

    preprocessor = build_preprocessing(X_train)
    best_name, best_pipeline = select_best_model(preprocessor, X_train, y_train, config.cv_folds)

    best_pipeline.fit(X_train, y_train)
    test_metrics = evaluate_on_test(best_pipeline, X_test, y_test)
    importance = extract_feature_importance(best_pipeline)

    model_path = config.output_dir / "clinical_model.joblib"
    joblib.dump(best_pipeline, model_path)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "selected_model": best_name,
        "random_state": RANDOM_STATE,
        "cv_folds": config.cv_folds,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "test_metrics": test_metrics,
        "top_features": importance,
    }
    metadata_path = config.output_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    try:
        ConfusionMatrixDisplay.from_estimator(best_pipeline, X_test, y_test).figure_.savefig(
            config.output_dir / "confusion_matrix.png", bbox_inches="tight"
        )
    except Exception as exc:
        logger.warning("Could not save confusion matrix plot: %s", exc)

    logger.info("Saved pipeline to %s", model_path)
    logger.info("Saved metadata to %s", metadata_path)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the clinical classifier.")
    parser.add_argument("--data-path", type=Path, default=TrainConfig.data_path)
    parser.add_argument("--target-column", type=str, default=TrainConfig.target_column)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--test-size", type=float, default=TrainConfig.test_size)
    parser.add_argument("--cv-folds", type=int, default=TrainConfig.cv_folds)
    args = parser.parse_args()
    return TrainConfig(
        data_path=args.data_path,
        target_column=args.target_column,
        output_dir=args.output_dir,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    train(parse_args())