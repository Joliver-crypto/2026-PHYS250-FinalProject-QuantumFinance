"""Build model-ready training events from local all_events + asset returns."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from event_schema import validate_training_events

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

ALL_EVENTS_PATH = DATA_DIR / "all_events.csv"
HOLDOUT_EVENTS_PATH = DATA_DIR / "holdout_events.csv"
RETURNS_PATH = DATA_DIR / "asset_returns_clean.csv"
OUTPUT_PATH = DATA_DIR / "training_events.csv"
REPORT_PATH = DATA_DIR / "training_events_report.md"

DEFAULT_LABEL_HORIZON = 5
TREND_DAMPENING_THRESHOLD = 0.035
TREND_DAMPENING_FLOOR = 0.45

FEATURE_COLUMNS = [
    "q0_overall_intensity",
    "q1_geo_macro_score",
    "q2_disaster_disruption_score",
    "q3_technology_adoption_score",
    "q4_regulation_policy_score",
]

RAW_FEATURE_COLUMNS = [f"raw_{column}" for column in FEATURE_COLUMNS]

RETURN_COLS = ["usd_return", "gold_return", "sp500_return", "btc_return"]
NEXT_RETURN_COLS = ["usd_next_return", "gold_next_return", "sp500_next_return", "btc_next_return"]
EXCESS_RETURN_COLS = [
    "usd_excess_sp500_return",
    "gold_excess_sp500_return",
    "sp500_excess_sp500_return",
    "btc_excess_sp500_return",
]
MAGNITUDE_BUCKET_COLS = [
    "usd_magnitude_bucket",
    "gold_magnitude_bucket",
    "sp500_magnitude_bucket",
    "btc_magnitude_bucket",
]
LABEL_COLS = ["usd_up", "gold_up", "sp500_up", "btc_up"]


@dataclass
class BuildStats:
    input_events: int = 0
    output_rows: int = 0
    dropped_no_future: int = 0
    dropped_on_or_after_cutoff: int = 0
    dropped_label_crosses_cutoff: int = 0


def _rolling_sum_last(arr: np.ndarray, window: int) -> float:
    if arr.size == 0:
        return 0.0
    w = min(window, arr.size)
    return float(np.sum(arr[-w:]))


def _rolling_std_last(arr: np.ndarray, window: int) -> float:
    if arr.size < 2:
        return 0.0
    w = min(window, arr.size)
    seg = arr[-w:]
    if seg.size < 2:
        return float(np.abs(seg[-1])) if seg.size else 0.0
    return float(np.std(seg, ddof=0))


def _market_context_from_history(hist: pd.DataFrame) -> dict[str, float]:
    """Past-only stats from daily returns rows strictly before an event date."""
    out: dict[str, float] = {}
    if hist.empty:
        for c in RETURN_COLS:
            out[f"{c}_sum5"] = 0.0
            out[f"{c}_vol20"] = 0.0
        return out
    for c in RETURN_COLS:
        s = pd.to_numeric(hist[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        out[f"{c}_sum5"] = _rolling_sum_last(s, 5)
        out[f"{c}_vol20"] = _rolling_std_last(s, 20)
    return out


def _trend_magnitude(ctx: dict[str, float]) -> float:
    sums = [abs(float(ctx[f"{c}_sum5"])) for c in RETURN_COLS]
    return float(np.mean(sums))


def _trend_dampening_factor(trend_magnitude: float) -> float:
    if trend_magnitude <= TREND_DAMPENING_THRESHOLD:
        return 1.0
    factor = TREND_DAMPENING_THRESHOLD / max(trend_magnitude, 1e-12)
    return float(np.clip(factor, TREND_DAMPENING_FLOOR, 1.0))


def fuse_event_features_with_market(
    q0: float,
    q1: float,
    q2: float,
    q3: float,
    q4: float,
    ctx: dict[str, float],
) -> tuple[float, float, float, float, float]:
    """Blend small, bounded past-only market signals into the five circuit inputs (same names)."""
    ret_scale = 0.028
    vol_scale = 0.018

    def _t(x: float) -> float:
        return float(np.tanh(x / ret_scale))

    usd_5 = ctx["usd_return_sum5"]
    gold_5 = ctx["gold_return_sum5"]
    sp_5 = ctx["sp500_return_sum5"]
    btc_5 = ctx["btc_return_sum5"]
    vol_mean = float(
        np.mean(
            [
                ctx["usd_return_vol20"],
                ctx["gold_return_vol20"],
                ctx["sp500_return_vol20"],
                ctx["btc_return_vol20"],
            ]
        )
    )
    cross = (usd_5 + gold_5 + sp_5 + btc_5) / 4.0

    q0n = float(np.clip(q0 + 0.10 * _t(cross) - 0.06 * float(np.tanh(vol_mean / vol_scale)), 0.0, 1.0))
    q1n = float(np.clip(q1 + 0.12 * _t(usd_5), -1.0, 1.0))
    q2n = float(np.clip(q2 + 0.12 * _t(gold_5), -1.0, 1.0))
    q3n = float(np.clip(q3 + 0.12 * _t(sp_5), -1.0, 1.0))
    q4n = float(np.clip(q4 + 0.12 * _t(btc_5), -1.0, 1.0))
    return q0n, q1n, q2n, q3n, q4n


def _load_returns(path: Path) -> pd.DataFrame:
    returns = pd.read_csv(path)
    expected = {"Date", "USD", "GOLD", "SP500", "BTC"}
    if set(returns.columns) != expected:
        raise ValueError(f"Unexpected returns columns: {returns.columns.tolist()}")
    returns = returns.rename(
        columns={
            "Date": "prediction_date",
            "USD": "usd_return",
            "GOLD": "gold_return",
            "SP500": "sp500_return",
            "BTC": "btc_return",
        }
    )
    returns["prediction_date"] = pd.to_datetime(returns["prediction_date"], errors="raise")
    returns = returns.sort_values("prediction_date").reset_index(drop=True)
    return returns


def _load_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path, dtype="string", keep_default_na=False)
    events["event_date"] = pd.to_datetime(events["date"], errors="coerce")
    if events["event_date"].isna().any():
        bad = events.loc[events["event_date"].isna(), "event_id"].tolist()[:5]
        raise ValueError(f"Found invalid event dates. Examples: {bad}")
    for column in FEATURE_COLUMNS:
        events[column] = pd.to_numeric(events[column], errors="raise")
    return events


def _load_temporal_cutoff(path: Path | None) -> pd.Timestamp | None:
    if path is None or not path.exists():
        return None
    holdouts = _load_events(path)
    return pd.Timestamp(holdouts["event_date"].min()).normalize()


def _assign_split(n_rows: int) -> np.ndarray:
    n_train = int(np.floor(0.70 * n_rows))
    remaining = n_rows - n_train
    n_val = int(np.round(0.5 * remaining))
    n_test = remaining - n_val
    split = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test, dtype=object)
    return split


def _daily_context(events: pd.DataFrame) -> pd.DataFrame:
    base = events[["event_id", "event_date", "q0_overall_intensity"]].copy()
    base["event_day"] = base["event_date"].dt.date.astype(str)

    grouped = base.groupby("event_day")["q0_overall_intensity"]
    stats = grouped.agg(
        daily_event_count="count",
        daily_q0_sum="sum",
        daily_q0_max="max",
    )
    top2_gap = {}
    for day, values in grouped:
        ordered = np.sort(values.to_numpy(dtype=float))[::-1]
        if ordered.size == 1:
            gap = float(ordered[0])
        else:
            gap = float(ordered[0] - ordered[1])
        top2_gap[day] = gap
    stats["daily_q0_top2_gap"] = pd.Series(top2_gap)

    out = base.merge(stats, left_on="event_day", right_index=True, how="left")
    out["daily_relative_q0"] = np.where(
        out["daily_q0_max"] > 0.0,
        out["q0_overall_intensity"] / out["daily_q0_max"],
        1.0,
    )
    out["daily_relative_q0"] = out["daily_relative_q0"].clip(0.0, 1.0)
    return out.set_index("event_id")[
        [
            "daily_event_count",
            "daily_q0_sum",
            "daily_q0_max",
            "daily_q0_top2_gap",
            "daily_relative_q0",
        ]
    ]


def _magnitude_bucket(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value < 0.002:
        return "flat"
    direction = "up" if value > 0.0 else "down"
    size = "small" if abs_value < 0.015 else "large"
    return f"{size}_{direction}"


def _compound_window_returns(window: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for column in RETURN_COLS:
        values = pd.to_numeric(window[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        out[column] = float(np.prod(1.0 + values) - 1.0)
    return out


def build_labeled_event_frame(
    events: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    label_horizon: int = DEFAULT_LABEL_HORIZON,
    fuse_market: bool = False,
    temporal_cutoff: pd.Timestamp | None = None,
    enforce_temporal_cutoff: bool = True,
) -> tuple[pd.DataFrame, BuildStats]:
    """Create labeled rows with adjusted five-feature circuit inputs.

    Temporal exclusion rule: when ``temporal_cutoff`` is provided and enforcement is on,
    a row is eligible only if both its event date and the full label window end date are
    strictly before ``T_cut``. That means no return row dated on or after ``T_cut`` can
    enter labels, feature history, optimizer loss, or validation/test metrics.
    """
    if label_horizon < 1:
        raise ValueError("label_horizon must be >= 1")

    events = events.sort_values(["event_date", "event_id"]).reset_index(drop=True).copy()
    returns = returns.sort_values("prediction_date").reset_index(drop=True).copy()
    return_dates = returns["prediction_date"].to_numpy(dtype="datetime64[ns]")
    daily = _daily_context(events)
    stats = BuildStats(input_events=len(events))

    rows = []
    for _, event in events.iterrows():
        event_id = str(event["event_id"])
        event_ts = pd.Timestamp(event["event_date"]).normalize()
        if temporal_cutoff is not None and enforce_temporal_cutoff and event_ts >= temporal_cutoff:
            stats.dropped_on_or_after_cutoff += 1
            continue

        event_date64 = np.datetime64(event_ts.to_datetime64())
        label_start_idx = int(np.searchsorted(return_dates, event_date64, side="right"))
        label_end_idx = label_start_idx + label_horizon
        if label_end_idx > len(returns):
            stats.dropped_no_future += 1
            continue

        window = returns.iloc[label_start_idx:label_end_idx]
        label_start = pd.Timestamp(window.iloc[0]["prediction_date"]).normalize()
        label_end = pd.Timestamp(window.iloc[-1]["prediction_date"]).normalize()
        if temporal_cutoff is not None and enforce_temporal_cutoff and label_end >= temporal_cutoff:
            stats.dropped_label_crosses_cutoff += 1
            continue

        hist_end_idx = int(np.searchsorted(return_dates, event_date64, side="left"))
        hist = returns.iloc[:hist_end_idx]
        ctx = _market_context_from_history(hist)
        trend_mag = _trend_magnitude(ctx)
        trend_factor = _trend_dampening_factor(trend_mag)

        daily_row = daily.loc[event_id]
        daily_relative_q0 = float(daily_row["daily_relative_q0"])
        event_attention_factor = float(np.clip(0.55 + 0.45 * daily_relative_q0, 0.55, 1.0))
        feature_factor = event_attention_factor * trend_factor

        raw_features = [float(event[column]) for column in FEATURE_COLUMNS]
        q0, q1, q2, q3, q4 = raw_features
        q0 = float(np.clip(q0 * feature_factor, 0.0, 1.0))
        q1 = float(np.clip(q1 * feature_factor, -1.0, 1.0))
        q2 = float(np.clip(q2 * feature_factor, -1.0, 1.0))
        q3 = float(np.clip(q3 * feature_factor, -1.0, 1.0))
        q4 = float(np.clip(q4 * feature_factor, -1.0, 1.0))
        if fuse_market:
            q0, q1, q2, q3, q4 = fuse_event_features_with_market(q0, q1, q2, q3, q4, ctx)

        cumulative_returns = _compound_window_returns(window)
        next_returns = {column: float(window.iloc[0][column]) for column in RETURN_COLS}
        sp500_cumulative = cumulative_returns["sp500_return"]
        usd_up = int(cumulative_returns["usd_return"] > 0.0)
        gold_up = int(cumulative_returns["gold_return"] > 0.0)
        sp500_up = int(cumulative_returns["sp500_return"] > 0.0)
        btc_up = int(cumulative_returns["btc_return"] > 0.0)
        bitstring = f"{usd_up}{gold_up}{sp500_up}{btc_up}"

        row = {
            "event_id": event_id,
            "event_date": event_ts.date().isoformat(),
            "label_start_date": label_start.date().isoformat(),
            "label_end_date": label_end.date().isoformat(),
            "label_horizon_trading_days": int(label_horizon),
            "prediction_date": label_end.date().isoformat(),
            "primary_category": event["primary_category"],
            "q0_overall_intensity": q0,
            "q1_geo_macro_score": q1,
            "q2_disaster_disruption_score": q2,
            "q3_technology_adoption_score": q3,
            "q4_regulation_policy_score": q4,
            "raw_q0_overall_intensity": raw_features[0],
            "raw_q1_geo_macro_score": raw_features[1],
            "raw_q2_disaster_disruption_score": raw_features[2],
            "raw_q3_technology_adoption_score": raw_features[3],
            "raw_q4_regulation_policy_score": raw_features[4],
            "daily_event_count": int(daily_row["daily_event_count"]),
            "daily_q0_sum": float(daily_row["daily_q0_sum"]),
            "daily_q0_max": float(daily_row["daily_q0_max"]),
            "daily_q0_top2_gap": float(daily_row["daily_q0_top2_gap"]),
            "daily_relative_q0": daily_relative_q0,
            "event_attention_factor": event_attention_factor,
            "pre_event_trend_magnitude": trend_mag,
            "trend_dampening_factor": trend_factor,
            "usd_return": cumulative_returns["usd_return"],
            "gold_return": cumulative_returns["gold_return"],
            "sp500_return": cumulative_returns["sp500_return"],
            "btc_return": cumulative_returns["btc_return"],
            "usd_next_return": next_returns["usd_return"],
            "gold_next_return": next_returns["gold_return"],
            "sp500_next_return": next_returns["sp500_return"],
            "btc_next_return": next_returns["btc_return"],
            "usd_excess_sp500_return": cumulative_returns["usd_return"] - sp500_cumulative,
            "gold_excess_sp500_return": cumulative_returns["gold_return"] - sp500_cumulative,
            "sp500_excess_sp500_return": 0.0,
            "btc_excess_sp500_return": cumulative_returns["btc_return"] - sp500_cumulative,
            "usd_magnitude_bucket": _magnitude_bucket(cumulative_returns["usd_return"]),
            "gold_magnitude_bucket": _magnitude_bucket(cumulative_returns["gold_return"]),
            "sp500_magnitude_bucket": _magnitude_bucket(cumulative_returns["sp500_return"]),
            "btc_magnitude_bucket": _magnitude_bucket(cumulative_returns["btc_return"]),
            "usd_up": usd_up,
            "gold_up": gold_up,
            "sp500_up": sp500_up,
            "btc_up": btc_up,
            "target_bitstring_usd_gold_sp500_btc": bitstring,
        }
        rows.append(row)

    output = pd.DataFrame(rows)
    stats.output_rows = len(output)
    if output.empty:
        return output, stats
    output = output.sort_values(["event_date", "event_id"]).reset_index(drop=True)
    return output, stats


def _ordered_columns() -> list[str]:
    return [
        "event_id",
        "event_date",
        "label_start_date",
        "label_end_date",
        "label_horizon_trading_days",
        "prediction_date",
        "primary_category",
        *FEATURE_COLUMNS,
        *RAW_FEATURE_COLUMNS,
        "daily_event_count",
        "daily_q0_sum",
        "daily_q0_max",
        "daily_q0_top2_gap",
        "daily_relative_q0",
        "event_attention_factor",
        "pre_event_trend_magnitude",
        "trend_dampening_factor",
        *RETURN_COLS,
        *NEXT_RETURN_COLS,
        *EXCESS_RETURN_COLS,
        *MAGNITUDE_BUCKET_COLS,
        *LABEL_COLS,
        "target_bitstring_usd_gold_sp500_btc",
        "split",
    ]


def _format_optional_cutoff(cutoff: pd.Timestamp | None) -> str:
    return "none" if cutoff is None else cutoff.date().isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training_events.csv from all_events + returns.")
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=DEFAULT_LABEL_HORIZON,
        help="Primary label horizon in trading rows after event_date. Default: 5.",
    )
    parser.add_argument(
        "--fuse-market",
        action="store_true",
        help="After explicit event attention adjustment, blend small past-only 5d momentum and 20d vol "
        "signals into q0-q4. Circuit unchanged (still five angles).",
    )
    parser.add_argument(
        "--holdout-events",
        default=str(HOLDOUT_EVENTS_PATH),
        help="Holdout CSV used only to compute T_cut=min(holdout date).",
    )
    parser.add_argument(
        "--no-temporal-cutoff",
        action="store_true",
        help="Disable holdout T_cut filtering. Use only for exploratory debugging.",
    )
    args = parser.parse_args()

    events = _load_events(ALL_EVENTS_PATH)
    returns = _load_returns(RETURNS_PATH)
    holdout_path = Path(args.holdout_events) if args.holdout_events else None
    temporal_cutoff = None if args.no_temporal_cutoff else _load_temporal_cutoff(holdout_path)

    training, stats = build_labeled_event_frame(
        events,
        returns,
        label_horizon=args.label_horizon,
        fuse_market=bool(args.fuse_market),
        temporal_cutoff=temporal_cutoff,
        enforce_temporal_cutoff=not args.no_temporal_cutoff,
    )
    if training.empty:
        raise ValueError("No training rows produced.")

    training["split"] = _assign_split(len(training))
    training = training[_ordered_columns()]
    training.to_csv(OUTPUT_PATH, index=False)

    validation = validate_training_events(training)
    if not validation.ok:
        joined = "\n".join(validation.issues)
        raise ValueError(f"Validation failed:\n{joined}")

    split_counts = training["split"].value_counts().to_dict()
    report_lines = [
        "# Training Events Build Report",
        "",
        "## Temporal Exclusion",
        f"- T_cut: {_format_optional_cutoff(temporal_cutoff)}",
        "- Rule: training_events includes only rows with `event_date < T_cut` and "
        "`label_end_date < T_cut` when a holdout file is present.",
        "- Consequence: no return rows dated on or after T_cut are used in labels, feature-history "
        "construction, optimizer loss, validation metrics, or test metrics.",
        "",
        "## Primary Label Rule",
        f"- Horizon: {args.label_horizon} trading rows after `event_date`.",
        "- `usd_return`, `gold_return`, `sp500_return`, and `btc_return` are compounded returns over "
        "that full horizon: `prod(1 + daily_return) - 1`.",
        "- `usd_up`, `gold_up`, `sp500_up`, and `btc_up` are the primary training labels and equal "
        "`1` when the corresponding compounded return is greater than zero.",
        "- `*_next_return`, `*_excess_sp500_return`, and `*_magnitude_bucket` are retained only as "
        "analysis columns; they are not used by the quantum loss.",
        "",
        "## Event Attention Adjustment",
        "- Raw event inputs are preserved as `raw_q0_*` through `raw_q4_*`; adjusted values keep the "
        "same five feature names consumed by the 9-qubit circuit.",
        "- Per calendar day, `daily_relative_q0 = raw_q0 / daily_q0_max` and "
        "`event_attention_factor = 0.55 + 0.45*daily_relative_q0`.",
        f"- Past-only trend magnitude is the mean absolute 5-row return sum across the four assets. "
        f"If it exceeds {TREND_DAMPENING_THRESHOLD:.3f}, "
        f"`trend_dampening_factor = clip({TREND_DAMPENING_THRESHOLD:.3f}/trend, "
        f"{TREND_DAMPENING_FLOOR:.2f}, 1.00)`; otherwise it is 1.",
        "- Adjusted circuit inputs are `raw_q* * event_attention_factor * trend_dampening_factor` "
        "clipped to the valid feature range.",
        f"- fuse_market: {bool(args.fuse_market)}",
        "",
        "## Counts",
        f"- Input all_events rows: {stats.input_events}",
        f"- Output training_events rows: {len(training)}",
        f"- Rows dropped (no full future return window): {stats.dropped_no_future}",
        f"- Rows dropped (event_date on/after T_cut): {stats.dropped_on_or_after_cutoff}",
        f"- Rows dropped (label window crosses T_cut): {stats.dropped_label_crosses_cutoff}",
        f"- Split counts: train={split_counts.get('train', 0)}, "
        f"val={split_counts.get('val', 0)}, test={split_counts.get('test', 0)}",
        f"- Label balance USD (mean usd_up): {training['usd_up'].mean():.3f}",
        f"- Label balance Gold (mean gold_up): {training['gold_up'].mean():.3f}",
        f"- Label balance SP500 (mean sp500_up): {training['sp500_up'].mean():.3f}",
        f"- Label balance BTC (mean btc_up): {training['btc_up'].mean():.3f}",
        "- Target bitstring order: USD, Gold, SP500, BTC",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH} ({len(training)} rows)")
    print(
        "Split counts:",
        {
            "train": split_counts.get("train", 0),
            "val": split_counts.get("val", 0),
            "test": split_counts.get("test", 0),
        },
    )
    print(f"T_cut: {_format_optional_cutoff(temporal_cutoff)}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()  # pragma: no cover
