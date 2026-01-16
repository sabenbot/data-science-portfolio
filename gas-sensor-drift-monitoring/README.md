# Gas Sensor Drift Monitoring
## When Your ML Model Slowly Goes Wrong

**Tech:** Python, scikit-learn, matplotlib  
**Author:** Alex Domingues Batista

---

## The Short Version

- A model trained on Month 1 sensor data drops from **100% → 33% accuracy** by Month 36
- This happens because the sensors themselves drift over time (same problem I dealt with in analytical chemistry)
- Windowed retraining (last 3 batches) keeps accuracy at **100%** throughout
- Static models silently fail. That's the point of this project.

---

## What's the Problem?

In analytical chemistry, I spent years dealing with instrument calibration drift. Sensors age. They get contaminated. Temperature changes affect readings. You have to recalibrate regularly or your results go bad.

ML models have the same problem. A model trained on last year's sensor data will make increasingly wrong predictions as the sensors drift. This is called **concept drift**, and it's one of the biggest reasons production ML systems fail.

The tricky part: the model doesn't throw an error. It just quietly gets worse. By the time someone notices, the damage is done.

---

## Dataset

UCI Gas Sensor Array Drift Dataset:
- 16 metal oxide sensors measuring 6 different gases
- 36 months of data split into 10 batches
- Sensors were left to age naturally (no maintenance)

This is a worst-case scenario for concept drift—perfect for studying the problem.

---

## What I Did

### Notebook 1: Proving the Drift is Real

Before assuming there's a problem, I checked if the sensor data actually changes over time.

**Statistical tests (Kolmogorov-Smirnov):**
- 94 out of 128 features showed significant drift (p < 0.001)
- Median KS statistic around 0.3-0.4 (that's a lot of shift)

**PCA trajectory:**
- Plotted the centroid of each batch in PCA space
- Clear monotonic drift over 36 months—not random noise

This matters because you need evidence to justify retraining costs. "The model feels off" doesn't work in budget meetings.

### Notebook 2: Measuring the Decay

Trained a Random Forest on Batch 1, then tested on all 10 batches without retraining.

| Batch | Accuracy |
|-------|----------|
| 1 | 100% |
| 5 | 68% |
| 10 | 33% |

The model loses 67% of its accuracy over 3 years. And nothing breaks or throws an error—it just returns wrong answers with confidence.

### Notebook 3: Fixing It

Compared four retraining strategies:

| Strategy | Mean Accuracy | Final Accuracy |
|----------|---------------|----------------|
| Static (no retraining) | 53% | 33% |
| Previous batch only | 91% | 89% |
| **Windowed (last 3)** | **100%** | **99.8%** |
| Cumulative (all history) | 98% | 97% |

Windowed retraining wins. Use recent data, but not too much—old calibration patterns can hurt more than help.

---

## Why Windowed Works Best

Using only the previous batch is too volatile—you're betting everything on one measurement campaign.

Using all historical data sounds smart but actually hurts. The model learns old sensor behaviors that no longer apply.

Three batches hits the sweet spot: enough data to be stable, recent enough to match current sensor state.

This is exactly how analytical labs work. You recalibrate with fresh standards regularly, but you don't throw out all your institutional knowledge either.

---

## The Real Point

This project isn't really about gas sensors. It's about something that happens to every production ML system: silent degradation.

The model doesn't crash. It doesn't throw errors. It just slowly gets worse, and by the time you notice, you've been making bad decisions for months.

The fix is monitoring + adaptive retraining. This project shows how to do both.

---

## Files

```
gas-sensor-drift-monitoring/
├── 01_visualizing_the_drift.ipynb    # Statistical proof that drift exists
├── 02_model_decay_analysis.ipynb     # How bad the decay gets
├── 03_adaptive_calibration.ipynb     # Retraining strategies compared
├── Dataset/                          # 10 batch files
└── README.md
```

---

## Running It

```bash
# Get the data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00270/Dataset.zip
unzip Dataset.zip

# Clone and run
git clone https://github.com/alexdbatista/data-science-portfolio.git
cd data-science-portfolio/gas-sensor-drift-monitoring
pip install -r requirements.txt
jupyter notebook
```

Run notebooks in order: 01 → 02 → 03

---

## What I'd Do Next

1. **Automatic drift detection** - Trigger retraining when KS statistics exceed a threshold, not on a fixed schedule
2. **Online learning** - Update the model incrementally instead of full retraining
3. **Cost-optimal scheduling** - Balance retraining cost vs accuracy loss (there's a calculus problem hiding here)

---

## Why This Matters for My Background

I spent 10+ years in analytical chemistry calibrating instruments, tracking drift, setting up QC protocols. The mental model is identical:

| Lab Work | ML Work |
|----------|---------|
| HPLC baseline drift | Model accuracy decay |
| Calibration curves shift | Feature distributions change |
| QC samples catch problems | Drift detection catches problems |
| Recalibrate with standards | Retrain with new data |

The tools are different. The problem is the same.

---

## Dataset Citation

```
Vergara, A., et al. (2012). Chemical gas sensor drift compensation 
using classifier ensembles. Sensors and Actuators B: Chemical, 166, 320-329.
```

Data: https://archive.ics.uci.edu/ml/datasets/gas+sensor+array+drift+dataset
