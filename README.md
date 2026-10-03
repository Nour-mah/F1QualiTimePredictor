# F1 Quali Time Predictor

Predicting each driver's Q3 qualifying gap to pole, from Friday/Saturday free-practice (FP1-3) data — built with [FastF1](https://github.com/theOehrly/Fast-F1).

This is a portfolio project focused as much on **MLOps and Docker practices** (reproducible pipelines, experiment tracking, containerized serving) as on the modeling itself. It's a work in progress, built and documented incrementally.

## What it does

For each Grand Prix, the pipeline:
1. Pulls FP1, FP2, FP3, and Qualifying session data via FastF1.
2. Builds per-driver features for each free-practice session (computed over all drivers of the session):
   - `FP{n}_gap_to_best_pct`: best lap as a percentage gap to the best lap of the session
   - `FP{n}_compound`: tyre compound of that best lap (0 = soft, 1 = medium, 2 = hard, 3 = intermediate, 4 = wet)
   - `FP{n}_laps_ratio`: number of clean laps divided by the maximum of the session
3. Adds relative weather features from the free-practice sessions only (qualifying weather would leak information): track/air temperature change between the first and last FP session, and rain.
4. Computes the target: `gap_to_pole_pct = (driver's Q3 time - pole time) / pole time * 100`, sourced from the official Q3 results (not just the fastest lap of the session).
5. Builds one row per driver × Grand Prix, restricted to the Q3 top 10, and sorts the dataset chronologically.

Drivers need at least 2 valid FP sessions. Missing session values are left as `NaN` (no imputation; XGBoost handles them natively).

Data window: 2022–2026 seasons, conventional (non-sprint) weekends only (78 Grands Prix, 761 rows).

## Modeling approach

- **Relative features.** Absolute lap times depend on the circuit (a 1'30" is great at Monza, poor at Monaco), so the model is fed relative gaps. Switching from absolute times to relative gaps took the XGBoost RMSE on a single split from 1.80 to 0.62.
- **Temporal validation.** No random split: it would leak information between drivers of the same Grand Prix and between past and future. The model is evaluated with a **time-series cross-validation** in blocks of 10 Grands Prix (expanding training window, 5 folds).
- **Naive baseline.** Every fold also reports the RMSE of a model that always predicts the training mean, to check that the model actually learns something.

- **One feature group at a time.** Each new group is added on top of the FP gaps and compared to the baseline, to know which one actually helps.

### Current results (5-fold temporal CV, XGBoost default params, RMSE in percentage points)

| Features | RMSE (mean) |
|---|---|
| Naive baseline (train mean) | **0.9094** |
| FP gaps only | 0.9235 |
| + tyre compound | 0.95 |
| + rain | 0.95 |
| + temperature change | 1.13 |
| + laps ratio | 1.36 |
| + absolute weather (temperatures, humidity, wind) | 2.09 |

**No feature set beats the naive baseline yet.** Absolute weather values act as a fingerprint of the circuit, so the model memorizes Grands Prix instead of learning general rules; relative weather is much less harmful but still doesn't help. With only 78 Grands Prix and noisy free-practice data (different run programs, fuel loads), the signal is weak. A learning curve suggests more data would help a little; options under consideration: adding the 2018–2021 seasons, a more regularized or simpler model.

## Project status

- [x] Data collection from FastF1 (`src/data/fetch_fastf1.py`)
- [x] Per-GP feature engineering + full dataset assembly (`src/data/build_features_tabular.py`)
- [x] Baseline XGBoost with temporal cross-validation (`src/models/train_classical.py`)
- [x] Feature experiments: tyre compound, laps ratio, weather (selectable feature groups)
- [ ] Improve the model (more seasons, regularization, simpler models)
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
    └── train_classical.py        # XGBoost baseline + temporal cross-validation, selectable feature groups

data/            # raw/processed data, gitignored (not versioned in git)
```

## Running the pipeline

```bash
cd src/data
python fetch_fastf1.py            # populates data/raw/ (long-running, respects FastF1 API rate limits)
python build_features_tabular.py  # builds data/processed/tabular/F1_features_dataset_<timestamp>.csv

cd ../models
python train_classical.py         # loads the latest dataset, runs the temporal CV, prints RMSE per fold
python train_classical.py --features gap compound rain   # choose feature groups: gap (default), compound, laps, rain, temp_delta
```
