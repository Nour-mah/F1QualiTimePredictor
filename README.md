# F1 Quali Time Predictor

Predicting each driver's Q3 qualifying gap to pole, from Friday/Saturday free-practice (FP1-3) data — built with [FastF1](https://github.com/theOehrly/Fast-F1).

This is a portfolio project focused as much on **MLOps and Docker practices** (reproducible pipelines, experiment tracking, containerized serving) as on the modeling itself. It's a work in progress, built and documented incrementally.

## What it does

For each Grand Prix, the pipeline:
1. Pulls FP1, FP2, FP3, and Qualifying session data via FastF1 for a given season.
2. Aggregates each driver's best lap time per free-practice session.
3. Computes the target: `gap_to_pole_pct = (driver's Q3 time - pole time) / pole time * 100`, sourced from the official Q3 results (not just the fastest lap of the session).
4. Builds one row per driver × Grand Prix, restricted to the Q3 top 10.

Data window: 2022–2026 seasons, conventional (non-sprint) weekends only.

## Project status

- [x] Data collection from FastF1 (`src/data/fetch_fastf1.py`)
- [x] Per-GP feature engineering + full dataset assembly (`src/data/build_features_tabular.py`)
- [ ] Baseline model (XGBoost) — in progress (`src/models/train_classical.py`)
- [ ] Tensor-based pipeline for a PyTorch model (Phase 2)
- [ ] Experiment tracking (MLflow)
- [ ] Serving API
- [ ] Docker / docker-compose
- [ ] CI/CD
- [ ] Tests

## Stack

Python, pandas, FastF1, scikit-learn / XGBoost (planned), PyTorch (planned), MLflow (planned), Docker (planned).

## Repo structure

```
src/
├── data/
│   ├── fetch_fastf1.py           # Pulls raw session data from FastF1
│   └── build_features_tabular.py # Builds the per-GP tabular dataset
└── models/
    └── train_classical.py        # Baseline model training (in progress)

data/            # raw/processed data, gitignored (not versioned in git)
```

## Running the data pipeline

```bash
cd src/data
python fetch_fastf1.py          # populates data/raw/ (long-running, respects FastF1 API rate limits)
python build_features_tabular.py  # builds data/processed/tabular/F1_features_dataset_<timestamp>.csv
```
