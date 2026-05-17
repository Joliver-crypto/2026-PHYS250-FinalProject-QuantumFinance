"""Evaluation helpers for the simplified quantum market model.

The primary binary labels are multi-day compounded-return signs. Optional
next-day, SP500-excess, and magnitude-bucket columns are analysis-only fields.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from event_schema import load_training_events as load_training_events_schema

FEATURE_COLUMNS = [
    "q0_overall_intensity",
    "q1_geo_macro_score",
    "q2_disaster_disruption_score",
    "q3_technology_adoption_score",
    "q4_regulation_policy_score",
]

LABEL_COLUMNS = ["usd_up", "gold_up", "sp500_up", "btc_up"]
LABEL_NAMES = ["USD", "Gold", "SP500", "BTC"]
TARGET_COLUMN = "target_bitstring_usd_gold_sp500_btc"
PRIMARY_LABEL_RULE = "multi_day_compounded_return_sign"


def load_training_events(path: str = "data/training_events.csv") -> pd.DataFrame:
    """Load training_events.csv with schema validation."""
    return load_training_events_schema(path)


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Stack q0..q4 columns into an (n, 5) float array for the circuit."""
    return df[FEATURE_COLUMNS].to_numpy(dtype=float)


def get_label_matrix(df: pd.DataFrame) -> np.ndarray:
    """Stack usd_up..btc_up into an (n, 4) int array of realized directions."""
    return df[LABEL_COLUMNS].to_numpy(dtype=int)


def get_target_bitstrings(df: pd.DataFrame) -> list[str]:
    """Four-bit strings in USD-Gold-SP500-BTC order for joint NLL."""
    return df[TARGET_COLUMN].astype(str).tolist()


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    return float(-(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)).mean())


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def binary_accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    return float(np.mean(y_true == y_pred))


def marginal_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    return binary_cross_entropy(y_true=y_true, y_prob=y_prob, eps=eps)


def summarize_label_balance(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for col, name in zip(LABEL_COLUMNS, LABEL_NAMES):
        out[name] = {"positive_rate": float(df[col].mean()), "n": float(len(df))}
    return out


def split_training_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "split" in df.columns and set(df["split"].unique()).issubset({"train", "val", "test"}):
        train = df[df["split"] == "train"].copy()
        val = df[df["split"] == "val"].copy()
        test = df[df["split"] == "test"].copy()
        return train, val, test

    ordered = df.sort_values(["event_date", "event_id"]).reset_index(drop=True).copy()
    n = len(ordered)
    n_train = int(np.floor(0.70 * n))
    n_val = int(np.floor(0.15 * n))
    train = ordered.iloc[:n_train].copy()
    val = ordered.iloc[n_train : n_train + n_val].copy()
    test = ordered.iloc[n_train + n_val :].copy()
    return train, val, test


def independent_bernoulli_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Dict[str, object]:
    if eval_df.empty:
        return {"status": "skipped", "reason": "empty eval set"}
    probs = np.asarray([float(train_df[c].mean()) for c in LABEL_COLUMNS], dtype=float)
    y_true = get_label_matrix(eval_df)
    y_prob = np.tile(probs.reshape(1, -1), (len(eval_df), 1))
    return evaluate_predictions(y_true=y_true, y_prob=y_prob, label_names=LABEL_NAMES)


def logistic_regression_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Dict[str, object]:
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return {"status": "skipped", "reason": "scikit-learn not available"}

    if train_df.empty or eval_df.empty:
        return {"status": "skipped", "reason": "empty split"}

    x_train = get_feature_matrix(train_df)
    x_eval = get_feature_matrix(eval_df)
    y_train = get_label_matrix(train_df)
    y_eval = get_label_matrix(eval_df)

    pred_cols = []
    for i in range(y_train.shape[1]):
        y_col = y_train[:, i]
        if len(np.unique(y_col)) < 2:
            prob = np.full(len(x_eval), float(y_col[0]))
        else:
            clf = LogisticRegression(max_iter=1000, random_state=7)
            clf.fit(x_train, y_col)
            prob = clf.predict_proba(x_eval)[:, 1]
        pred_cols.append(prob)

    y_prob = np.vstack(pred_cols).T
    return evaluate_predictions(y_true=y_eval, y_prob=y_prob, label_names=LABEL_NAMES)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str] | None = None,
) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_prob={y_prob.shape}")
    if y_true.ndim != 2 or y_true.shape[1] != 4:
        raise ValueError("Expected y_true/y_prob shape (n_rows, 4)")

    names = label_names or LABEL_NAMES
    per_label = []
    for idx, label in enumerate(names):
        yt = y_true[:, idx]
        yp = y_prob[:, idx]
        per_label.append(
            {
                "label": label,
                "bce_log_loss": binary_cross_entropy(yt, yp),
                "brier_score": brier_score(yt, yp),
                "accuracy": binary_accuracy(yt, yp),
                "mean_predicted_probability": float(np.mean(yp)),
            }
        )

    mean_metrics = {
        "bce_log_loss": float(np.mean([m["bce_log_loss"] for m in per_label])),
        "brier_score": float(np.mean([m["brier_score"] for m in per_label])),
        "accuracy": float(np.mean([m["accuracy"] for m in per_label])),
    }
    return {"status": "ok", "metrics": per_label, "mean_metrics": mean_metrics}
