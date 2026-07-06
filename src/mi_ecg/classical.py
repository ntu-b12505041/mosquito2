from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_classical_model(name: str, seed: int = 42):
    if name == "logistic_regression":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=250,
            l2_regularization=0.01,
            early_stopping=True,
            random_state=seed,
        )
    raise ValueError(f"Unknown classical model: {name}")


def predict_scores(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError("Model does not expose predict_proba or decision_function")

