"""Schema helpers and validators for simplified local quantum-finance data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

ALL_EVENTS_COLUMNS = [
    "event_id",
    "date",
    "event_name",
    "summary",
    "source_title",
    "source_url",
    "primary_category",
    "q1_geo_macro_score",
    "q2_disaster_disruption_score",
    "q3_technology_adoption_score",
    "q4_regulation_policy_score",
    "q0_overall_intensity",
    "sentiment",
    "confidence",
    "notes",
    "legacy_event_type",
    "legacy_asset",
    "legacy_impact_score",
]

TRAINING_EVENTS_COLUMNS = [
    "event_id",
    "event_date",
    "label_start_date",
    "label_end_date",
    "label_horizon_trading_days",
    "prediction_date",
    "primary_category",
    "q0_overall_intensity",
    "q1_geo_macro_score",
    "q2_disaster_disruption_score",
    "q3_technology_adoption_score",
    "q4_regulation_policy_score",
    "raw_q0_overall_intensity",
    "raw_q1_geo_macro_score",
    "raw_q2_disaster_disruption_score",
    "raw_q3_technology_adoption_score",
    "raw_q4_regulation_policy_score",
    "daily_event_count",
    "daily_q0_sum",
    "daily_q0_max",
    "daily_q0_top2_gap",
    "daily_relative_q0",
    "event_attention_factor",
    "pre_event_trend_magnitude",
    "trend_dampening_factor",
    "usd_return",
    "gold_return",
    "sp500_return",
    "btc_return",
    "usd_next_return",
    "gold_next_return",
    "sp500_next_return",
    "btc_next_return",
    "usd_excess_sp500_return",
    "gold_excess_sp500_return",
    "sp500_excess_sp500_return",
    "btc_excess_sp500_return",
    "usd_magnitude_bucket",
    "gold_magnitude_bucket",
    "sp500_magnitude_bucket",
    "btc_magnitude_bucket",
    "usd_up",
    "gold_up",
    "sp500_up",
    "btc_up",
    "target_bitstring_usd_gold_sp500_btc",
    "split",
]

PRIMARY_CATEGORIES = {
    "q1_geo_macro",
    "q2_disaster_disruption",
    "q3_technology_adoption",
    "q4_regulation_policy",
}

TARGET_BITSTRING_COLUMN = "target_bitstring_usd_gold_sp500_btc"


@dataclass
class ValidationResult:
    name: str
    ok: bool
    issues: list[str]


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> list[str]:
    issues: list[str] = []
    missing = [col for col in columns if col not in df.columns]
    extra = [col for col in df.columns if col not in columns]
    if missing:
        issues.append(f"{name}: missing columns: {missing}")
    if extra:
        issues.append(f"{name}: unexpected columns: {extra}")
    return issues


def load_all_events(path: Path | str = DATA_DIR / "all_events.csv") -> pd.DataFrame:
    """Load all-events CSV with stable dtypes for text columns."""
    dtype = {
        "event_id": "string",
        "date": "string",
        "event_name": "string",
        "summary": "string",
        "source_title": "string",
        "source_url": "string",
        "primary_category": "string",
        "sentiment": "string",
        "notes": "string",
        "legacy_event_type": "string",
        "legacy_asset": "string",
        "legacy_impact_score": "string",
    }
    return pd.read_csv(path, dtype=dtype, keep_default_na=False)


def load_training_events(path: Path | str = DATA_DIR / "training_events.csv") -> pd.DataFrame:
    """Load model-ready training CSV with bitstrings preserved as strings."""
    dtype = {
        "event_id": "string",
        "event_date": "string",
        "label_start_date": "string",
        "label_end_date": "string",
        "prediction_date": "string",
        "primary_category": "string",
        "split": "string",
        "target_bitstring_usd_gold_sp500_btc": "string",
        "usd_magnitude_bucket": "string",
        "gold_magnitude_bucket": "string",
        "sp500_magnitude_bucket": "string",
        "btc_magnitude_bucket": "string",
    }
    return pd.read_csv(path, dtype=dtype, keep_default_na=False)


def validate_all_events(df: pd.DataFrame) -> ValidationResult:
    issues = _ensure_columns(df, ALL_EVENTS_COLUMNS, "all_events")
    if len(df) < 200:
        issues.append(f"all_events: expected at least 200 rows, found {len(df)}")

    if not df["event_id"].str.fullmatch(r"EVT_\d{4}").all():
        issues.append("all_events: event_id must be EVT_####")

    if not df["event_id"].is_unique:
        issues.append("all_events: event_id values must be unique")

    categories = set(df["primary_category"].unique())
    if not categories.issubset(PRIMARY_CATEGORIES):
        issues.append(f"all_events: category set mismatch: {sorted(categories)}")
    if categories != PRIMARY_CATEGORIES:
        issues.append(f"all_events: missing categories: {sorted(PRIMARY_CATEGORIES - categories)}")

    counts = df["primary_category"].value_counts().to_dict()
    for category in sorted(PRIMARY_CATEGORIES):
        if counts.get(category, 0) < 50:
            issues.append(
                f"all_events: expected at least 50 rows for {category}, found {counts.get(category, 0)}"
            )

    if (df["source_url"].str.strip() == "").any():
        issues.append("all_events: source_url contains blank values")

    for column in [
        "q1_geo_macro_score",
        "q2_disaster_disruption_score",
        "q3_technology_adoption_score",
        "q4_regulation_policy_score",
    ]:
        if not pd.to_numeric(df[column], errors="coerce").between(-1.0, 1.0).all():
            issues.append(f"all_events: {column} must be in [-1, 1]")

    if not pd.to_numeric(df["q0_overall_intensity"], errors="coerce").between(0.0, 1.0).all():
        issues.append("all_events: q0_overall_intensity must be in [0, 1]")

    if not pd.to_numeric(df["confidence"], errors="coerce").between(0.0, 1.0).all():
        issues.append("all_events: confidence must be in [0, 1]")

    return ValidationResult(name="all_events", ok=not issues, issues=issues)


def validate_training_events(df: pd.DataFrame) -> ValidationResult:
    issues = _ensure_columns(df, TRAINING_EVENTS_COLUMNS, "training_events")

    if len(df) == 0:
        issues.append("training_events: dataset is empty")

    for col in ["event_date", "label_start_date", "label_end_date", "prediction_date"]:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.isna().any():
            issues.append(f"training_events: invalid dates in {col}")

    if "label_horizon_trading_days" in df.columns:
        horizons = pd.to_numeric(df["label_horizon_trading_days"], errors="coerce")
        if horizons.isna().any() or (horizons <= 0).any():
            issues.append("training_events: label_horizon_trading_days must be positive")

    label_cols = ["usd_up", "gold_up", "sp500_up", "btc_up"]
    for col in label_cols:
        values = set(pd.to_numeric(df[col], errors="coerce").dropna().astype(int).tolist())
        if not values.issubset({0, 1}):
            issues.append(f"training_events: {col} must be binary")

    for column in [
        "q0_overall_intensity",
        "raw_q0_overall_intensity",
        "daily_relative_q0",
        "event_attention_factor",
        "trend_dampening_factor",
    ]:
        if column in df.columns and not pd.to_numeric(df[column], errors="coerce").between(0.0, 1.0).all():
            issues.append(f"training_events: {column} must be in [0, 1]")

    for column in [
        "q1_geo_macro_score",
        "q2_disaster_disruption_score",
        "q3_technology_adoption_score",
        "q4_regulation_policy_score",
        "raw_q1_geo_macro_score",
        "raw_q2_disaster_disruption_score",
        "raw_q3_technology_adoption_score",
        "raw_q4_regulation_policy_score",
    ]:
        if column in df.columns and not pd.to_numeric(df[column], errors="coerce").between(-1.0, 1.0).all():
            issues.append(f"training_events: {column} must be in [-1, 1]")

    bitstring = df[TARGET_BITSTRING_COLUMN].astype("string")
    if not bitstring.str.fullmatch(r"[01]{4}").all():
        issues.append("training_events: target bitstring must be exactly four binary digits")

    reconstructed = (
        df["usd_up"].astype(int).astype(str)
        + df["gold_up"].astype(int).astype(str)
        + df["sp500_up"].astype(int).astype(str)
        + df["btc_up"].astype(int).astype(str)
    )
    if not (reconstructed == bitstring).all():
        issues.append("training_events: target bitstring does not match usd/gold/sp500/btc labels")

    split_values = set(df["split"].astype(str))
    if not split_values.issubset({"train", "val", "test"}):
        issues.append(f"training_events: invalid split values: {sorted(split_values)}")

    return ValidationResult(name="training_events", ok=not issues, issues=issues)


def validate_holdout_events(df: pd.DataFrame) -> ValidationResult:
    issues = _ensure_columns(df, ALL_EVENTS_COLUMNS, "holdout_events")
    if len(df) != 3:
        issues.append(f"holdout_events: expected exactly 3 rows, found {len(df)}")

    if not df["event_id"].str.fullmatch(r"EVT_\d{4}").all():
        issues.append("holdout_events: event_id must be EVT_####")
    if not df["event_id"].is_unique:
        issues.append("holdout_events: event_id values must be unique")

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    if parsed_dates.isna().any():
        issues.append("holdout_events: invalid dates in date")

    categories = set(df["primary_category"].unique())
    if not categories.issubset(PRIMARY_CATEGORIES):
        issues.append(f"holdout_events: category set mismatch: {sorted(categories)}")

    if (df["source_url"].str.strip() == "").any():
        issues.append("holdout_events: source_url contains blank values")

    for column in [
        "q1_geo_macro_score",
        "q2_disaster_disruption_score",
        "q3_technology_adoption_score",
        "q4_regulation_policy_score",
    ]:
        if not pd.to_numeric(df[column], errors="coerce").between(-1.0, 1.0).all():
            issues.append(f"holdout_events: {column} must be in [-1, 1]")

    if not pd.to_numeric(df["q0_overall_intensity"], errors="coerce").between(0.0, 1.0).all():
        issues.append("holdout_events: q0_overall_intensity must be in [0, 1]")
    if not pd.to_numeric(df["confidence"], errors="coerce").between(0.0, 1.0).all():
        issues.append("holdout_events: confidence must be in [0, 1]")

    return ValidationResult(name="holdout_events", ok=not issues, issues=issues)


def main() -> None:
    all_events_path = DATA_DIR / "all_events.csv"
    training_path = DATA_DIR / "training_events.csv"

    if not all_events_path.exists():
        raise FileNotFoundError(f"Missing required file: {all_events_path}")

    all_events = load_all_events(all_events_path)
    results = [validate_all_events(all_events)]

    if training_path.exists():
        training_events = load_training_events(training_path)
        results.append(validate_training_events(training_events))
    else:
        print(f"[SKIP] training_events (missing file: {training_path})")

    holdout_path = DATA_DIR / "holdout_events.csv"
    if holdout_path.exists():
        holdout_events = load_all_events(holdout_path)
        results.append(validate_holdout_events(holdout_events))
    else:
        print(f"[SKIP] holdout_events (missing file: {holdout_path})")

    all_ok = True
    for result in results:
        if result.ok:
            print(f"[OK] {result.name}")
            continue
        all_ok = False
        print(f"[FAIL] {result.name}")
        for issue in result.issues:
            print(f"  - {issue}")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
