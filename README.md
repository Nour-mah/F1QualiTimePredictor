# F1 Quali Time Predictor

Predicting the Q3 qualifying order (top 10) of a Formula 1 Grand Prix from free-practice (FP1-3) data and each driver's recent qualifying form — built with [FastF1](https://github.com/theOehrly/Fast-F1).

This is a portfolio project focused as much on **MLOps and Docker practices** (reproducible pipelines, experiment tracking, containerized serving) as on the modeling itself. It's a work in progress, built and documented incrementally.

## What it does

For each Grand Prix, the pipeline:
1. Pulls FP1, FP2, FP3, and Qualifying session data via FastF1.
2. Builds per-driver features for each free-practice session (computed over all drivers of the session):
   - `FP{n}_gap_to_best_pct`: best lap as a percentage gap to the best lap of the session
   - `FP{n}_compound`: tyre compound of that best lap (0 = soft, 1 = medium, 2 = hard, 3 = intermediate, 4 = wet)
   - `FP{n}_laps_ratio`: number of clean laps divided by the maximum of the session
3. Adds the driver's recent form: `driver_quali_pos_avg`, the mean qualifying position over the 5 previous Grands Prix (strictly earlier ones only, to avoid leaking the target).
4. Computes two targets from the official qualifying results:
   - `quali_position`: final position in the top 10 (main target)
   - `gap_to_pole_pct = (driver's Q3 time - pole time) / pole time * 100` (secondary target)
5. Builds one row per driver × Grand Prix, restricted to the official top 10, and sorts the dataset chronologically.

Drivers need at least 2 valid FP sessions. Missing values are left as `NaN` (no imputation; XGBoost handles them natively).

Data window: 2022–2026 seasons, conventional (non-sprint) weekends only (78 Grands Prix, 780 rows).

## Modeling approach

- **Learning to rank.** The main task is to order the 10 drivers within each Grand Prix, so the model is an `XGBRanker` trained with one group per Grand Prix. It is scored with the **Spearman correlation** between predicted and actual order, computed per Grand Prix and averaged (1 = perfect order).
- **A strong reference.** The model is compared to the order of the last free-practice session (FP3, falling back on FP2 then FP1). It is a much more demanding baseline than predicting a constant.
- **Temporal validation.** No random split: it would leak information between drivers of the same Grand Prix and between past and future. Evaluation uses a **time-series cross-validation** in blocks of 10 Grands Prix (expanding training window, 5 folds).
- **Relative features.** Absolute lap times depend on the circuit, so the model is fed gaps relative to the session's best lap.
- **A deliberately simple model.** With ~700 rows, default XGBoost parameters (depth 6) overfit and score below the reference. The model uses depth-1 trees, 100 estimators and a 0.05 learning rate.

### Current results (5-fold temporal CV)

Ranking the top 10 (Spearman, higher is better):

| Model | Spearman (mean ± std) |
|---|---|
| Reference: free-practice order | 0.604 ± 0.077 |
| XGBRanker, FP gaps only | 0.606 ± 0.063 |
| XGBRanker, driver history only | 0.578 ± 0.065 |
| **XGBRanker, FP gaps + driver history** | **0.651 ± 0.042** |
| + tyre compound + laps ratio | 0.649 ± 0.042 |

Free-practice pace and recent form complement each other: combined, they beat the free-practice order on average and are more consistent across folds. The gain comes mostly from the two oldest folds; on the three most recent ones the reference stays slightly ahead, so the improvement is real but modest. Hyperparameters were chosen on these same folds, which makes the scores slightly optimistic.

Predicting the gap to pole (RMSE in percentage points, lower is better): 0.805 with FP gaps + driver history, against 0.910 for a naive model predicting the training mean.

### What did not work

- **Absolute lap times** as features: circuit-dependent, far worse than the naive baseline.
- **Weather** (removed): identical for every driver of a Grand Prix, so it cannot change their order; absolute values also act as a fingerprint of the circuit and cause overfitting.
- **Tyre compound and laps ratio**: no measurable gain so far (still available as feature groups).

## Project status

- [x] Data collection from FastF1 (`src/data/fetch_fastf1.py`)
- [x] Per-GP feature engineering + full dataset assembly (`src/data/build_features_tabular.py`)
- [x] Ranking model with temporal cross-validation and selectable feature groups (`src/models/train_classical.py`)
- [ ] Improve the model (teammate-relative features, more seasons)
- [ ] Tensor-based pipeline for a PyTorch model (Phase 2)
- [ ] Experiment tracking (MLflow)
- [ ] Serving API
- [ ] Docker / docker-compose
- [ ] CI/CD
- [ ] Tests
- [ ] Data versioning (DVC)

## Stack

Python, pandas, FastF1, scikit-learn, SciPy, XGBoost. Planned: PyTorch, MLflow, Docker, DVC.

## Repo structure

```
src/
├── data/
│   ├── fetch_fastf1.py           # Pulls raw session data from FastF1
│   └── build_features_tabular.py # Builds the per-GP tabular dataset
└── models/
    └── train_classical.py        # XGBoost ranker/regressor + temporal cross-validation

data/            # raw/processed data, gitignored (not versioned in git)
```

## Running the pipeline

```bash
cd src/data
python fetch_fastf1.py            # populates data/raw/ (long-running, respects FastF1 API rate limits)
python build_features_tabular.py  # builds data/processed/tabular/F1_features_dataset_<timestamp>.csv

cd ../models
python train_classical.py                                # ranking of the top 10, features: gap + history
python train_classical.py --features gap history compound laps   # feature groups: gap, history, compound, laps
python train_classical.py --target gap                   # secondary task: regression of the gap to pole
```
