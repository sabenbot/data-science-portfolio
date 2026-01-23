# NASA Turbofan Engine Degradation Analysis
## Predictive Maintenance for Sensor Time-Series

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

---

## Summary

This project predicts the Remaining Useful Life (RUL) of turbofan engines using multivariate sensor data. I treated the 21 engine sensors like analytical instruments—applying signal quality checks, feature engineering based on physical degradation patterns, and validation that prevents data leakage.

**Results:**
- RMSE: 17.25 cycles (8.6% of average engine lifespan)
- R²: 0.950
- 68.9% improvement over baseline (55.54 → 17.25 cycles)
- Competitive with published benchmarks (12-18 cycles)

The key insight: **time normalization** (where the engine is in its lifecycle) captured 87% of the predictive signal. Cumulative sensor trends captured the rest. Raw sensor values barely mattered.

---

## The Problem

Unplanned engine failures are expensive. In aerospace:
- Unscheduled removal: €500K+ per event
- Unplanned downtime: €22,000/hour for widebody aircraft
- Premature replacement: 40% of replaced parts still have 50%+ useful life

Predictive maintenance aims to find the sweet spot: replace parts late enough to maximize use, early enough to avoid failure. That requires predicting RUL with enough precision to schedule maintenance 2-3 weeks out.

---

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation), FD001 subset:
- 100 engines, run-to-failure trajectories
- 21 sensors + 3 operational settings per timestep
- ~200 cycles per engine on average

The challenge: sensors are noisy, some are flatlined, and degradation patterns vary between engines.

---

## Approach

### 1. Signal Quality (Notebooks 01 & 02)

First pass: identify which sensors actually carry information.

- 7 sensors had zero variance (dead signals)—removed
- 3 operational settings were constant—removed
- 7 sensors showed clear degradation trends—kept as primary features

This mirrors how you'd audit an analytical instrument: check calibration, identify dead channels, validate signal quality before trusting the data.

### 2. Feature Engineering (Notebook 03)

The real work. I engineered 86 features from the 7 useful sensors:

| Feature Type | Count | Rationale |
|--------------|-------|-----------|
| Rolling stats (mean, std, min, max) | 28 | Short-term sensor behavior |
| Lag features (t-5, t-10) | 14 | Recent operational history |
| Rate of change | 7 | Degradation acceleration |
| Cumulative (cumsum, cummax) | 14 | Lifetime degradation accumulation |
| Time normalization | 1 | Lifecycle position (0-1 scale) |

The cumulative features matter because mechanical degradation is irreversible—once a component wears, it doesn't recover. Rolling statistics capture short-term noise; cumulative features capture the underlying trend.

### 3. Model Selection

Compared Random Forest vs. Gradient Boosting:

| Model | RMSE | R² | Notes |
|-------|------|-----|-------|
| Random Forest | 22.21 | 0.918 | Solid baseline |
| **Gradient Boosting** | **17.25** | **0.950** | Better at capturing non-linear degradation |

Gradient Boosting won. Hyperparameters: 200 estimators, max_depth=5, learning_rate=0.05.

### 4. Validation Strategy

Critical: I used engine-level train/test split (engines 1-80 for training, 81-100 for testing). This prevents data leakage—you can't train on one timestep and predict another from the same engine.

Many published results on this dataset use random splits, which inflates accuracy. My approach is more realistic: the model has never seen these engines before.

---

## Key Finding: Time Normalization Dominates

Feature importance analysis revealed something striking:

| Feature | Importance |
|---------|------------|
| time_normalized | 87.1% |
| s_7_cumsum | 2.4% |
| s_12_cumsum | 2.3% |
| Everything else | <2% |

The engine's position in its lifecycle (time_normalized = current_cycle / max_cycle) contains almost all the predictive information. This makes physical sense: engines degrade predictably over time, and the sensors mostly reflect that underlying trend.

The practical implication: if you know how long an engine has been running and you have a rough sense of its expected lifespan, you can predict RUL surprisingly well—even without sophisticated sensor analysis.

---

## What I Actually Struggled With

### Train/Test Split Mistake
My first version used random 80/20 splits. Got an R² of 0.95 and thought I was done. Then I realized: the same engine appeared in both train and test sets. The model was just memorizing individual engines.

Switched to engine-level splits (train on engines 1-80, test on 81-100). R² dropped to 0.486. That was the real baseline—everything after was honest improvement.

### Rolling Window Size
Tried window=5 (too noisy), window=20 (too laggy), settled on window=10. The same tradeoff I dealt with in analytical chemistry: smoothing vs. response time.

### The min_periods Bug
Spent 30 minutes debugging why my test set was empty before realizing that rolling(window=10) without min_periods=1 turns the first 10 rows of each engine into NaN. Then dropna() killed 20% of my data.

### Things That Didn't Work
- Exponential smoothing: no better than simple rolling mean
- Polynomial features: overfit immediately
- LSTM: promising but 100 engines isn't enough data to tune it properly

---

## Files

```
nasa-turbofan-predictive-maintenance/
├── 01_data_exploration.ipynb      # Initial data audit
├── 02_data_quality_and_drift.ipynb # Sensor quality analysis
├── 03_predictive_modeling.ipynb    # Feature engineering + modeling
├── train_FD001.txt                 # NASA C-MAPSS training data
├── test_FD001.txt                  # Test data
├── RUL_FD001.txt                   # Ground truth RUL
└── README.md
```

---

## Reproducing Results

```bash
git clone https://github.com/alexdbatista/data-science-portfolio.git
cd data-science-portfolio/nasa-turbofan-predictive-maintenance
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook
```

Data source: [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

---

## What I Learned

1. **Feature engineering > model selection.** Gradient Boosting beat Random Forest by 5 cycles, but good features beat bad features by 40+ cycles.

2. **Physical intuition matters.** Cumulative features work because degradation is irreversible. I wouldn't have known to try them without understanding the underlying process.

3. **Validation design is everything.** Random splits would have given me better numbers and worse real-world performance.

4. **Sometimes the answer is simple.** 87% of predictive power came from one feature. The sophisticated sensor analysis was almost unnecessary.
