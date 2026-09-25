# F1 Quali Time Predictor

Predicting each driver's Q3 qualifying gap to pole, from Friday/Saturday free-practice (FP1-3) data — built with [FastF1](https://github.com/theOehrly/Fast-F1).

This is a portfolio project focused as much on **MLOps and Docker practices** (reproducible pipelines, experiment tracking, containerized serving) as on the modeling itself. It's a work in progress, built and documented incrementally.

## What it does

For each Grand Prix, the pipeline:
1. Pulls FP1, FP2, FP3, and Qualifying session data via FastF1.
2. Computes, for each free-practice session, each driver's best lap as a **percentage gap to the best lap of that session** (`FP{1,2,3}_gap_to_best_pct`), computed over all drivers.
3. Computes the target: `gap_to_pole_pct = (driver's Q3 time - pole time) / pole time * 100`, sourced from the official Q3 results (not just the fastest lap of the session).
4. Builds one row per driver × Grand Prix, restricted to the Q3 top 10, and sorts the dataset chronologically.

Data window: 2022–2026 seasons, conventional (non-sprint) weekends only (78 Grands Prix, ~760 rows).

## Modeling approach

- **Relative features.** Absolute lap times depend on the circuit (a 1'30" is great at Monza, poor at Monaco), so the model is fed relative gaps. Switching from absolute times to relative gaps took the XGBoost RMSE on a single split from 1.80 to 0.62.
- **Temporal validation.** No random split: it would leak information between drivers of the same Grand Prix and between past and future. The model is evaluated with a **time-series cross-validation** in blocks of 10 Grands Prix (expanding training window, 5 folds).
- **Naive baseline.** Every fold also reports the RMSE of a model that always predicts the training mean, to check that the model actually learns something.

### Current results (5-fold temporal CV, RMSE in percentage points)

| Model | RMSE (mean ± std) |
|---|---|
| XGBoost (default params) | 0.9414 ± 0.4711 |
| Naive baseline (train mean) | 0.9094 ± 0.4367 |

The FP-gap features alone do **not** beat the naive baseline: the single-split RMSE of 0.6165 obtained earlier only reflected one favorable fold. Next step: enrich the features (weather, team, session rank), one at a time, and compare each against the baseline.

## Project status

- [x] Data collection from FastF1 (`src/data/fetch_fastf1.py`)
- [x] Per-GP feature engineering + full dataset assembly (`src/data/build_features_tabular.py`)
- [x] Baseline XGBoost with temporal cross-validation (`src/models/train_classical.py`)
- [ ] Feature enrichment (weather, team, session rank)
- [ ] Tensor-based pipeline for a PyTorch model (Phase 2)
- [ ] Experiment tracking (MLflow)
- [ ] Serving API
- [ ] Docker / docker-compose
- [ ] CI/CD
- [ ] Tests
- [ ] Data versioning (DVC)

## Stack

Python, pandas, FastF1, scikit-learn, XGBoost. Planned: PyTorch, MLflow, Docker, DVC.

## Repo structure

```
src/
├── data/
│   ├── fetch_fastf1.py           # Pulls raw session data from FastF1
│   └── build_features_tabular.py # Builds the per-GP tabular dataset
└── models/
    └── train_classical.py        # XGBoost baseline + temporal cross-validation

data/            # raw/processed data, gitignored (not versioned in git)
```

## Running the pipeline

```bash
cd src/data
python fetch_fastf1.py            # populates data/raw/ (long-running, respects FastF1 API rate limits)
python build_features_tabular.py  # builds data/processed/tabular/F1_features_dataset_<timestamp>.csv

cd ../models
python train_classical.py         # loads the latest dataset, runs the temporal CV, prints RMSE per fold
```
