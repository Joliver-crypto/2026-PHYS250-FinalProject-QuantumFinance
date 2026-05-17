# 2026 PHYS 250 Final Project — Quantum Finance

**Author:** Justin L. Oliver  
**Course:** PHYS 250 — Quantum Programming (SJSU)

A nine-qubit parameterized quantum circuit that maps headline-style event features to correlated up/down labels for USD, Gold, S&P 500, and Bitcoin. All execution is software simulation (Qiskit Aer), not hardware.

## Repository contents

| Path | Purpose |
|------|---------|
| `fine_tuned_circuit.ipynb` | Main narrative notebook: data overview, circuit, training-set accuracy, holdout case studies |
| `final_poster.pdf` | Course poster (PHYS 250 final presentation) |
| `data/` | CSV inputs required by the notebook (events, prices, returns) |
| `models/` | Trained circuit angles (`updated_circuit_params.json`) and smoke fallback |
| `src/` | Python modules imported by the notebook (`quantum_market_model`, holdout eval, etc.) |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook fine_tuned_circuit.ipynb
```

Open the notebook with the working directory set to this repository root (`data/` and `models/` are resolved relative to `Path.cwd()`).

## Data files (`data/`)

- `all_events.csv` — full event catalog (~216 headlines)
- `training_events.csv` — 200 pre-cutoff events used for circuit tuning (cutoff 2026-04-20)
- `holdout_events.csv` — three case studies held out from training
- `asset_prices_clean.csv` / `asset_returns_clean.csv` — daily price levels and returns for figures and label windows

## AI disclaimer

Generative AI (ChatGPT, Claude) assisted with data collection, labeling drafts, and much of the code. Research direction, topic choices, circuit design decisions, and validation are the author’s. See the notebook introduction for full disclosure.
