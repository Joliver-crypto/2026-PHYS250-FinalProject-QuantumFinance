"""Run fixed holdout case studies through the trained quantum market model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_training_events import (
    DEFAULT_LABEL_HORIZON,
    FEATURE_COLUMNS,
    HOLDOUT_EVENTS_PATH,
    RETURNS_PATH,
    _load_events,
    _load_returns,
    build_labeled_event_frame,
)
from quantum_market_model import ASSET_LABELS, build_circuit, predict_one

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = ROOT / "models" / "updated_circuit_params.json"
DEFAULT_OUTPUT = ROOT / "data" / "holdout_eval_report.md"

# Aer classical-register bit index in count strings (matches measure order in build_circuit)
SHOT_ASSET_BIT_INDEX = {"BTC": 0, "SP500": 1, "Gold": 2, "USD": 3}

ASSET_COLUMNS = {
    "USD": ("usd_return", "usd_up"),
    "Gold": ("gold_return", "gold_up"),
    "SP500": ("sp500_return", "sp500_up"),
    "BTC": ("btc_return", "btc_up"),
}


def _load_params(path: Path) -> tuple[np.ndarray, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    theta = np.asarray(payload["parameter_vector"], dtype=float)
    num_layers = int(payload.get("num_layers", 1))
    return theta, num_layers


def holdout_event_shot_seed(event_id: str, *, base: int = 1000) -> int:
    """Deterministic Aer seed per holdout row (matches `fine_tuned_circuit` notebook)."""
    digits = "".join(ch for ch in str(event_id) if ch.isdigit())
    return base + (int(digits) if digits else 0) % 997


def marginal_probs_from_aer_shots(
    features: np.ndarray,
    theta: np.ndarray,
    num_layers: int,
    *,
    shots: int,
    seed: int,
) -> dict[str, float]:
    """Marginal P(asset readout = 'up') from raw AerSimulator shot counts (one circuit execution)."""
    from qiskit_aer import AerSimulator  # local import keeps CLI import cost low

    qc = build_circuit(
        np.asarray(features, dtype=float),
        np.asarray(theta, dtype=float),
        num_layers=num_layers,
        add_measurements=True,
    )
    counts = AerSimulator().run(qc, shots=shots, seed_simulator=seed).result().get_counts()
    tot = sum(counts.values())
    if tot <= 0:
        raise RuntimeError("AerSimulator returned no counts")
    out: dict[str, float] = {}
    for asset in ASSET_LABELS:
        pos = SHOT_ASSET_BIT_INDEX[asset]
        up_ct = sum(v for bitstr, v in counts.items() if len(bitstr) == 4 and bitstr[pos] == "1")
        out[asset] = float(up_ct / tot)
    return out


def _markdown_table(rows: list[dict[str, object]], *, prob_header: str) -> str:
    lines = [
        f"| Event | Event Date | Label Window | Asset | {prob_header} | Realized Return | Realized Sign |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['event_name']} | {row['event_date']} | {row['label_window']} | "
            f"{row['asset']} | {float(row['prob_up']):.4f} | "
            f"{float(row['realized_return']):+.4%} | {row['realized_sign']} |"
        )
    return "\n".join(lines)


def run_holdout_eval(
    *,
    params_path: Path,
    holdout_path: Path,
    returns_path: Path,
    output_path: Path,
    label_horizon: int,
    fuse_market: bool,
    use_shot_marginals: bool = False,
    shots: int = 1000,
    shot_seed_base: int = 1000,
) -> pd.DataFrame:
    theta, num_layers = _load_params(params_path)
    holdouts = _load_events(holdout_path)
    temporal_cutoff = pd.Timestamp(holdouts["event_date"].min()).normalize()
    returns = _load_returns(returns_path)
    labeled, stats = build_labeled_event_frame(
        holdouts,
        returns,
        label_horizon=label_horizon,
        fuse_market=fuse_market,
        temporal_cutoff=None,
        enforce_temporal_cutoff=False,
    )
    if len(labeled) != 3:
        raise ValueError(
            f"Expected exactly 3 holdout rows with full return windows, built {len(labeled)} "
            f"(dropped_no_future={stats.dropped_no_future})"
        )

    event_names = holdouts.set_index("event_id")["event_name"].to_dict()
    records: list[dict[str, object]] = []
    for _, row in labeled.iterrows():
        features = row[FEATURE_COLUMNS].to_numpy(dtype=float)
        if use_shot_marginals:
            seed = holdout_event_shot_seed(str(row["event_id"]), base=shot_seed_base)
            marginals = marginal_probs_from_aer_shots(
                features, theta, num_layers, shots=int(shots), seed=int(seed)
            )
        else:
            pred = predict_one(features, theta, num_layers=num_layers)
            marginals = pred["marginals"]  # type: ignore[assignment]
        label_window = f"{row['label_start_date']} to {row['label_end_date']}"
        for asset in ASSET_LABELS:
            return_col, up_col = ASSET_COLUMNS[asset]
            sign = "up" if int(row[up_col]) == 1 else "down/flat"
            records.append(
                {
                    "event_id": row["event_id"],
                    "event_name": event_names[row["event_id"]],
                    "event_date": row["event_date"],
                    "label_window": label_window,
                    "asset": asset,
                    "prob_up": float(marginals[asset]),  # type: ignore[index]
                    "realized_return": float(row[return_col]),
                    "realized_sign": sign,
                }
            )

    result = pd.DataFrame(records)
    if use_shot_marginals:
        prob_header = f"P_up (Aer shots, n={int(shots)})"
        marginal_note = (
            f"- **P_up** in the table is the marginal fraction of `1` at each asset wire in **one** "
            f"`AerSimulator` run per event ({int(shots)} shots, seed = {shot_seed_base} + digits(event_id) mod 997)."
        )
    else:
        prob_header = "P_up (Born / statevector)"
        marginal_note = (
            "- **P_up** in the table is the exact marginal Born probability from `predict_one` "
            "(statevector), not a finite-shot estimate."
        )
    lines = [
        "# Holdout Case Study Evaluation",
        "",
        f"- Params: `{params_path}`",
        f"- Label horizon: {label_horizon} trading rows after event date",
        f"- T_cut: {temporal_cutoff.date().isoformat()}",
        "- Holdout events are outside `data/training_events.csv`; return rows on or after T_cut are "
        "used only here after training.",
        marginal_note,
        "",
        _markdown_table(records, prob_header=prob_header),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the three fixed holdout case studies.")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS))
    parser.add_argument("--holdout-events", default=str(HOLDOUT_EVENTS_PATH))
    parser.add_argument("--returns", default=str(RETURNS_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--label-horizon", type=int, default=DEFAULT_LABEL_HORIZON)
    parser.add_argument("--fuse-market", action="store_true")
    parser.add_argument(
        "--shot-marginals",
        action="store_true",
        help="Use AerSimulator shot fractions for P_up (same seed rule as the notebook).",
    )
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--shot-seed-base", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_holdout_eval(
        params_path=Path(args.params),
        holdout_path=Path(args.holdout_events),
        returns_path=Path(args.returns),
        output_path=Path(args.output),
        label_horizon=int(args.label_horizon),
        fuse_market=bool(args.fuse_market),
        use_shot_marginals=bool(args.shot_marginals),
        shots=int(args.shots),
        shot_seed_base=int(args.shot_seed_base),
    )
    print(result.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
