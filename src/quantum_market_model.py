"""Parameterized 9-qubit quantum model for simplified market-event experiments."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
except ImportError as exc:  # pragma: no cover - environment dependent
    QuantumCircuit = None  # type: ignore[assignment]
    Statevector = None  # type: ignore[assignment]
    _QISKIT_IMPORT_ERROR = exc
else:
    _QISKIT_IMPORT_ERROR = None

FEATURE_ORDER = [
    "q0_overall_intensity",
    "q1_geo_macro_score",
    "q2_disaster_disruption_score",
    "q3_technology_adoption_score",
    "q4_regulation_policy_score",
]

ASSET_LABELS = ["USD", "Gold", "SP500", "BTC"]


def _ensure_qiskit() -> None:
    if _QISKIT_IMPORT_ERROR is not None:
        raise ImportError(
            "Qiskit is required for quantum model simulation. Install qiskit to continue."
        ) from _QISKIT_IMPORT_ERROR


def get_qubit_map() -> Dict[str, int]:
    return {
        "q0_overall_intensity": 0,
        "q1_geo_macro_score": 1,
        "q2_disaster_disruption_score": 2,
        "q3_technology_adoption_score": 3,
        "q4_regulation_policy_score": 4,
        "USD": 5,
        "Gold": 6,
        "SP500": 7,
        "BTC": 8,
    }


def get_asset_pairs() -> List[Tuple[int, int]]:
    qmap = get_qubit_map()
    assets = [qmap["USD"], qmap["Gold"], qmap["SP500"], qmap["BTC"]]
    pairs: List[Tuple[int, int]] = []
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            pairs.append((assets[i], assets[j]))
    return pairs


def count_parameters(num_layers: int = 1) -> int:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    return num_layers * (4 + 20 + 6 + 4)


def initialize_parameters(seed: int = 7, scale: float = 0.05, num_layers: int = 1) -> np.ndarray:
    """Initialize around pi/2 with small Gaussian noise."""
    rng = np.random.default_rng(seed)
    total = count_parameters(num_layers=num_layers)
    return (np.pi / 2.0) + rng.normal(loc=0.0, scale=scale, size=total)


def flatten_parameters(param_dict: Dict[str, np.ndarray]) -> np.ndarray:
    out: List[float] = []
    for key in ["q0_to_event_cry", "news_to_asset_cry", "asset_rzz", "asset_readout_ry"]:
        values = np.asarray(param_dict[key], dtype=float).ravel()
        out.extend(values.tolist())
    return np.asarray(out, dtype=float)


def unflatten_parameters(param_vector: np.ndarray, num_layers: int = 1) -> Dict[str, np.ndarray]:
    expected = count_parameters(num_layers=num_layers)
    vec = np.asarray(param_vector, dtype=float).ravel()
    if vec.size != expected:
        raise ValueError(f"Expected {expected} parameters, received {vec.size}")

    offset = 0
    q0_to_event = vec[offset : offset + num_layers * 4].reshape(num_layers, 4)
    offset += num_layers * 4
    news_to_asset = vec[offset : offset + num_layers * 20].reshape(num_layers, 5, 4)
    offset += num_layers * 20
    asset_rzz = vec[offset : offset + num_layers * 6].reshape(num_layers, 6)
    offset += num_layers * 6
    asset_readout = vec[offset : offset + num_layers * 4].reshape(num_layers, 4)

    return {
        "q0_to_event_cry": q0_to_event,
        "news_to_asset_cry": news_to_asset,
        "asset_rzz": asset_rzz,
        "asset_readout_ry": asset_readout,
    }


def draw_circuit_mpl(qc: "QuantumCircuit", **kwargs: object):
    """Matplotlib draw tuned for readable figures (barriers hidden but still shape layout)."""
    draw_kw = {"fold": -1, "scale": 0.6, "plot_barriers": False, "cregbundle": False}
    draw_kw.update(kwargs)
    return qc.draw("mpl", **draw_kw)


def build_circuit(
    feature_vector: np.ndarray | List[float],
    param_vector: np.ndarray | List[float],
    num_layers: int = 1,
    add_measurements: bool = False,
) -> "QuantumCircuit":
    """Build the PQC. Barriers after each gate (or tight gate row) improve matplotlib layout; they are inert for ideal simulation."""
    _ensure_qiskit()
    feats = np.asarray(feature_vector, dtype=float).ravel()
    if feats.size != 5:
        raise ValueError("feature_vector must contain exactly five values: q0..q4")

    qmap = get_qubit_map()
    params = unflatten_parameters(np.asarray(param_vector, dtype=float), num_layers=num_layers)
    qc = QuantumCircuit(9, 4 if add_measurements else 0)

    # 1) Input encoding (barrier after each RY for mpl column breaks)
    qc.ry(np.pi * feats[0], qmap["q0_overall_intensity"])
    qc.ry((np.pi / 2.0) * feats[1], qmap["q1_geo_macro_score"])
    qc.ry((np.pi / 2.0) * feats[2], qmap["q2_disaster_disruption_score"])
    qc.ry((np.pi / 2.0) * feats[3], qmap["q3_technology_adoption_score"])
    qc.ry((np.pi / 2.0) * feats[4], qmap["q4_regulation_policy_score"])
    qc.barrier()

    feature_qubits = [
        qmap["q0_overall_intensity"],
        qmap["q1_geo_macro_score"],
        qmap["q2_disaster_disruption_score"],
        qmap["q3_technology_adoption_score"],
        qmap["q4_regulation_policy_score"],
    ]
    asset_qubits = [qmap["USD"], qmap["Gold"], qmap["SP500"], qmap["BTC"]]
    # news_to_asset_cry[:, j] matches ASSET_LABELS[j] (USD=0..BTC=3). Append CRYs in physical wire order q5→q8.

    for layer in range(num_layers):
        p0 = params["q0_to_event_cry"][layer]
        p_na = params["news_to_asset_cry"][layer]
        p_rzz = params["asset_rzz"][layer]
        p_ro = params["asset_readout_ry"][layer]

        q0, q1, q2, q3, q4 = feature_qubits
        a_usd, a_gold, a_sp, a_btc = asset_qubits

        # --- q0 -> event qubits (4 CRY, barrier after each) ---
        qc.cry(p0[0], q0, q1)
        qc.cry(p0[1], q0, q2)
        qc.cry(p0[2], q0, q3)
        qc.cry(p0[3], q0, q4)
        qc.barrier()

        # --- news q0 -> assets (q5 USD, q6 Gold, q7 SP500, q8 BTC) ---
        qc.cry(p_na[0, 0], q0, a_usd)
        qc.cry(p_na[0, 1], q0, a_gold)
        qc.cry(p_na[0, 2], q0, a_sp)
        qc.cry(p_na[0, 3], q0, a_btc)
        qc.barrier()
        # --- news q1 -> assets ---
        qc.cry(p_na[1, 0], q1, a_usd)
        qc.cry(p_na[1, 1], q1, a_gold)
        qc.cry(p_na[1, 2], q1, a_sp)
        qc.cry(p_na[1, 3], q1, a_btc)
        qc.barrier()
        # --- news q2 -> assets ---
        qc.cry(p_na[2, 0], q2, a_usd)
        qc.cry(p_na[2, 1], q2, a_gold)
        qc.cry(p_na[2, 2], q2, a_sp)
        qc.cry(p_na[2, 3], q2, a_btc)
        qc.barrier()
        # --- news q3 -> assets ---
        qc.cry(p_na[3, 0], q3, a_usd)
        qc.cry(p_na[3, 1], q3, a_gold)
        qc.cry(p_na[3, 2], q3, a_sp)
        qc.cry(p_na[3, 3], q3, a_btc)
        qc.barrier()
        # --- news q4 -> assets ---
        qc.cry(p_na[4, 0], q4, a_usd)
        qc.cry(p_na[4, 1], q4, a_gold)
        qc.cry(p_na[4, 2], q4, a_sp)
        qc.cry(p_na[4, 3], q4, a_btc)
        qc.barrier()

        # --- asset–asset RZZ (barrier after each; same pair order as get_asset_pairs()) ---
        qc.rzz(p_rzz[0], a_usd, a_gold)
        qc.rzz(p_rzz[1], a_usd, a_sp)
        qc.rzz(p_rzz[2], a_usd, a_btc)
        qc.rzz(p_rzz[3], a_gold, a_sp)
        qc.rzz(p_rzz[4], a_gold, a_btc)
        qc.rzz(p_rzz[5], a_sp, a_btc)
        qc.barrier()

        # --- readout RY on assets (barrier after each) ---
        qc.ry(p_ro[0], a_usd)
        qc.ry(p_ro[1], a_gold)
        qc.ry(p_ro[2], a_sp)
        qc.ry(p_ro[3], a_btc)
        qc.barrier()

    if add_measurements:
        q_meas = [qc.qubits[qmap["USD"]], qc.qubits[qmap["Gold"]], qc.qubits[qmap["SP500"]], qc.qubits[qmap["BTC"]]]
        c_meas = [qc.clbits[0], qc.clbits[1], qc.clbits[2], qc.clbits[3]]
        try:
            qc.measure(q_meas, c_meas)
        except (TypeError, ValueError):
            for qbit, cbit in zip(q_meas, c_meas):
                qc.measure(qbit, cbit)
    return qc


def asset_distribution_from_statevector(
    feature_vector: np.ndarray | List[float],
    param_vector: np.ndarray | List[float],
    num_layers: int = 1,
) -> Dict[str, float]:
    """Exact Born probabilities for 4 asset bits (q5..q8) from the statevector."""
    _ensure_qiskit()
    qc = build_circuit(feature_vector, param_vector, num_layers=num_layers, add_measurements=False)
    state = Statevector.from_instruction(qc)
    probs = state.probabilities()

    # Aggregate 2^9 full-basis probabilities into 2^4 asset bitstrings,
    # explicitly extracting qubits q5..q8 to avoid endianness ambiguity.
    distribution = {f"{idx:04b}": 0.0 for idx in range(16)}
    for basis_index, prob in enumerate(probs):
        b5 = (basis_index >> 5) & 1  # USD
        b6 = (basis_index >> 6) & 1  # Gold
        b7 = (basis_index >> 7) & 1  # SP500
        b8 = (basis_index >> 8) & 1  # BTC
        key = f"{b5}{b6}{b7}{b8}"
        distribution[key] += float(prob)
    return distribution


def marginal_probabilities_from_distribution(distribution: Dict[str, float]) -> Dict[str, float]:
    """Sum joint probabilities where each asset bit equals 1."""
    usd = sum(p for bit, p in distribution.items() if bit[0] == "1")
    gold = sum(p for bit, p in distribution.items() if bit[1] == "1")
    sp500 = sum(p for bit, p in distribution.items() if bit[2] == "1")
    btc = sum(p for bit, p in distribution.items() if bit[3] == "1")
    return {"USD": usd, "Gold": gold, "SP500": sp500, "BTC": btc}


def predict_one(
    feature_vector: np.ndarray | List[float],
    param_vector: np.ndarray | List[float],
    num_layers: int = 1,
) -> Dict[str, object]:
    """Marginals, full joint distribution, and mode bitstring for one headline."""
    distribution = asset_distribution_from_statevector(
        feature_vector=feature_vector,
        param_vector=param_vector,
        num_layers=num_layers,
    )
    marginals = marginal_probabilities_from_distribution(distribution)
    mode_bitstring = max(distribution.items(), key=lambda kv: kv[1])[0]
    return {
        "distribution": distribution,
        "marginals": marginals,
        "mode_bitstring": mode_bitstring,
    }
