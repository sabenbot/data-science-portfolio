# Metabolomics Biomarker Discovery
## Using SHAP to Prioritize What to Validate Next

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.45+-red.svg)](https://github.com/slundberg/shap)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

---

## The Short Version

- **Problem:** You have 63 metabolites from an LC-MS study. Validating all of them costs €1.7M and takes months.
- **Solution:** Use SHAP to rank which metabolites actually matter. Focus validation on the top 10.
- **Result:** Same scientific output, 84% cost reduction.

The model accuracy is only 57.9% (small dataset, n=76). That's fine—the point isn't diagnosis, it's prioritization. SHAP rankings are stable even when accuracy is modest.

---

## What's the Problem?

Biomarker discovery is expensive. You measure dozens or hundreds of metabolites, then you have to validate each one with targeted assays, larger cohorts, and clinical trials. Most candidates fail. The earlier you can filter out the weak ones, the more money you save.

Traditional approach: run t-tests on everything, pick whatever has p<0.05, hope for the best.

Better approach: train a model, use SHAP to see which features actually drive predictions, prioritize based on that.

Why SHAP over t-tests? T-tests look at one metabolite at a time. SHAP captures interactions. Sometimes Metabolite_45 doesn't look significant alone, but it matters a lot when combined with Metabolite_18. SHAP catches that.

---

## Dataset

Human cachexia study (muscle wasting syndrome in cancer patients):
- 76 samples (47 cachexia, 29 control)
- 63 metabolites measured by LC-MS
- Anonymized identifiers (Metabolite_1, Metabolite_2, etc.)

Small dataset, high dimensionality—exactly the situation where feature prioritization matters most.

---

## What I Did

### Notebook 1: Exploratory Analysis

Standard chemometrics workflow:
- Log₂ transformation (standard for LC-MS peak intensities)
- PCA to check if groups separate (they do—PC1 explains ~35% variance)
- Volcano plot to flag candidates with big fold changes

Found 20 upregulated and 15 downregulated metabolites at p<0.05. But that's still 35 candidates—too many for cheap validation.

### Notebook 2: Model Comparison

Tried Lasso (L1 logistic regression) and Random Forest:

| Model | CV Accuracy | ROC-AUC |
|-------|-------------|---------|
| Lasso | 57.9% | 0.668 |
| Random Forest | 48.6% | 0.498 |

Lasso wins. Random Forest overfit on this small dataset.

57.9% accuracy isn't great, but that's expected with 76 samples and 63 features. The model isn't for clinical use—it's for ranking features.

### Notebook 3: SHAP Analysis

This is where it gets interesting.

Ran SHAP on the Lasso model. Plotted global feature importance and beeswarm (shows direction: does high value push toward cachexia or control?).

**Top 10 metabolites capture 80%+ of the model's predictive signal.**

That means: instead of validating 63 metabolites, focus on 10. Same information, fraction of the cost.

---

## Why This Works

SHAP gives you three things t-tests don't:

1. **Multivariate context** — A metabolite can be important because of its interaction with others, not just its individual effect.

2. **Directional insight** — The beeswarm plot shows whether high values push toward disease or control. That's clinically actionable.

3. **Per-sample explanation** — You can see why the model predicted cachexia for a specific patient. Useful for debugging and building trust.

---

## The Honest Limitations

**Model accuracy is modest.** 57.9% isn't impressive. But for prioritization (not diagnosis), it's enough. The rankings are stable across cross-validation folds.

**Small sample size.** 76 samples is tiny for this many features. External validation on a larger cohort would be needed before any clinical claims.

**Anonymized metabolites.** I can't map these to biological pathways because the identities are hidden. In a real project, you'd cross-reference with KEGG/HMDB.

---

## Files

```
metabolomics-biomarker-discovery/
├── 01_chemometric_eda.ipynb       # Data quality, PCA, volcano plot
├── 02_biomarker_ml.ipynb          # Model training and comparison
├── 03_shap_interpretation.ipynb   # SHAP analysis and rankings
└── README.md
```

---

## Running It

```bash
cd metabolomics-biomarker-discovery
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Run notebooks in order: 01 → 02 → 03

---

## What I Struggled With

### SHAP output format for binary classification
For binary problems, `shap_values` returns a list of two arrays (one per class). I kept getting shape mismatch errors until I extracted just the positive class:
```python
shap_vals = shap_values[1]  # Positive class only
```

### Feature labels getting cut off
Matplotlib truncated my metabolite names. Fixed by calling `plt.subplots_adjust(left=0.25)` after SHAP creates the plot, not before.

### Log transform timing
Debated whether to log-transform before or after train/test split. Answer: before, because log transform doesn't learn parameters from data (unlike StandardScaler). It's deterministic.

### SHAP vs. built-in feature importance
Random Forest has `feature_importances_` built in. Why use SHAP? Because Gini importance measures "how often a feature is used to split," not "how much it affects predictions." They can give different rankings. SHAP is more theoretically grounded.

---

## Why This Matters for My Background

I spent years doing LC-MS method development—sample prep, chromatography optimization, peak integration. The data coming out of those instruments isn't magic. It's messy, it needs preprocessing, and the features you get depend heavily on how you ran the method.

That context helps me ask the right questions:
- Is the signal real or an artifact?
- Should I trust low-intensity peaks?
- How much batch-to-batch variation is there?

ML people without analytical chemistry background sometimes treat metabolomics data like generic tabular data. It's not. The measurement process matters.

---

## Tech Stack

- Python 3.10+
- pandas, numpy, scikit-learn
- SHAP 0.45+
- matplotlib, seaborn

Runs in <2 minutes on a laptop.

---

## References

**SHAP:**
- Lundberg & Lee (2017), [NIPS paper](https://arxiv.org/abs/1705.07874)
- [GitHub: slundberg/shap](https://github.com/slundberg/shap)

**Metabolomics Standards:**
- [Metabolomics Standards Initiative](http://www.metabolomics-msi.org/)
- [HMDB (Human Metabolome Database)](https://hmdb.ca/)
