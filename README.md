# Data Science Portfolio
**Alex Domingues Batista, PhD**
Data Scientist | ML Engineer | Sensor Analytics & Diagnostics Specialist

[![Humboldt Fellow](https://img.shields.io/badge/Humboldt-Fellow-green)]() [![Publications](https://img.shields.io/badge/Publications-50-blue)]() [![h--index](https://img.shields.io/badge/h--index-18-orange)]() [![Citations](https://img.shields.io/badge/Citations-1266-red)]()

---

## 🇩🇪 Germany Experience (2020-2024)

- **🎓 Humboldt Research Fellow** - Ulm University (2020-2021)
  - Developed molecularly imprinted polymers for SARS-CoV-2 detection (published in *Advanced Materials Interfaces*, 20+ citations)
  - Biosensor design using ML-guided polymer optimization

- **👨‍🔬 Research Group Leader** - Hahn-Schickard Institute (2022-2024)
  - Led biosensor diagnostics research group (5+ researchers)
  - Managed EU-funded research projects and industry partnerships
  - Developed ML pipelines for real-time sensor data analysis and quality control
  - Project acquisition and budget management for applied R&D

- **🗣️ Languages:** Portuguese (Native) | English (Fluent) | German (B1 Intermediate)

---

## 🎯 Quick Summary

**10+ years** combining machine learning with analytical chemistry for pharma, diagnostics, and Industry 4.0 applications.

**Core Strengths:**
- ✅ **ML & Data Science:** Scikit-Learn, SHAP, Time-Series Analysis, Sensor Drift Detection, Anomaly Detection
- ✅ **Domain Expertise:** Biosensors, Analytical Instrumentation, IoT Systems, Chemical Data
- ✅ **Applied ML:** Biomarker Discovery, Predictive Maintenance, Quality Control, Real-Time Diagnostics
- ✅ **Research Leadership:** Humboldt Fellow, Group Leader (Germany), Former Professor (Brazil), 50 Publications

**Target Industries:** Pharma • Life Sciences • Diagnostics • Manufacturing • Industry 4.0 / IIoT

---

## 👨‍🔬 About Me

I'm a **Data Scientist and ML Engineer** with 10+ years applying machine learning to analytical chemistry, sensor systems, and diagnostics. My career spans academia (PhD from USP-Brazil, 6 years as Professor at UFU) and industry R&D (Humboldt Fellow at Ulm University, Research Group Leader at Hahn-Schickard Institute in Germany).

**What I bring to industry:**
- **ML expertise with domain depth:** I've been applying chemometrics, PCA, time-series analysis, and predictive modeling to sensor data for over a decade—long before "data science" became mainstream.
- **Research leadership experience:** Managed research groups, coordinated EU-funded projects, and collaborated with industry partners at Hahn-Schickard (2022-2024).
- **Publication record:** 50 peer-reviewed papers (1,266 citations, h-index 18) in analytical methods, biosensors, and ML-enhanced diagnostics.
- **Physical intuition:** I understand how sensors fail, how instruments drift, and how chemical systems behave—enabling me to build more robust models that account for real-world complexity.

Now seeking **Data Scientist or ML Engineer roles in Germany** where I can apply this combination of ML skills and domain expertise to solve complex problems in pharma, diagnostics, or Industry 4.0.

---

## 💡 What Makes This Portfolio Different

This portfolio showcases **data science with deep domain knowledge**. My analytical chemistry background gives me an advantage that typical data scientists lack:

- **I understand the physical systems generating the data** (sensors, instruments, chemical processes)
- **I know when models are wrong** (from years of validating analytical methods and troubleshooting instruments)
- **I think in experiments** (hypothesis testing, controls, validation—not just train/test splits)
- **I can bridge technical teams** (from lab scientists to data engineers)

Every project here demonstrates how domain expertise makes data science better:
- ✅ **Physics-Informed Features:** Engineering features based on how sensors actually behave
- ✅ **Rigorous Validation:** Applying laboratory QA/QC thinking to ML model validation
- ✅ **Explainable Models:** Using SHAP and chemometric techniques to build trust with domain experts
- ✅ **Real-World Robustness:** Accounting for drift, noise, and measurement uncertainty

I'm not just doing data science—I'm doing **better data science** because I understand what the data means.

---

## 📊 Featured Projects

### 1. [Explainable AI for Biomarker Prioritization](./metabolomics-biomarker-discovery/)
**Objective:** Accelerate biomarker validation by prioritizing high-confidence candidates using interpretable ML.

**Highlights:**
- Analyzed human cachexia dataset (76 samples, 63 metabolites) with high biological variability.
- Performed chemometric QC using **Volcano Plots** and **PCA** to validate sample separation.
- Compared **Lasso Regression** (57.9% accuracy, for feature selection) vs. **Random Forest** (48.6% accuracy, for non-linearity).
- Identified key metabolic drivers (e.g., Glucose, 3-Hydroxybutyrate) using **SHAP values**.
- **Business outcome:** €1.7M validation cost savings through SHAP prioritization (18,150% ROI).

**Tech Stack:** Python, Pandas, Scikit-Learn, SHAP, Seaborn (Volcano Plots)
**Business Impact:** Reduces 18,150 candidate tests to 50 high-confidence targets; €1.46M savings from single SHAP analysis.

**[View Project →](./metabolomics-biomarker-discovery/)**

---

### 2. [Gas Sensor Drift & Calibration Transfer](./gas-sensor-drift-monitoring/)
**Objective:** Solve "Concept Drift" in industrial chemical sensors to extend hardware lifespan.

**Highlights:**
- Analyzed a 3-year longitudinal dataset (13,910 measurements) from 16 chemical sensors.
- Visualized **Concept Drift** using PCA, showing how sensor aging degrades model performance.
- Quantified the "Cost of Inaction": Static models lost **67% accuracy** over 36 months (100% → 33%).
- Implemented an **Adaptive Calibration** strategy (windowed retraining) that maintained **100% accuracy** throughout.

**Tech Stack:** Python, Scikit-Learn (PCA, Random Forest), Drift Detection
**Business Impact:** Prevents $141.7B in false classifications; extends operational life of remote IIoT sensor networks.

**[View Project →](./gas-sensor-drift-monitoring/)**

---

### 3. [NASA Turbofan Predictive Maintenance](./nasa-turbofan-predictive-maintenance/)
**Objective:** Predict the Remaining Useful Life (RUL) of aircraft engines using sensor time-series.

**Highlights:**
- Processed multivariate time-series data from 100 turbofan engines (NASA C-MAPSS).
- Engineered **86 advanced features** (rolling statistics, lag features, cumulative trends, rate-of-change) to capture degradation patterns.
- Trained a **Gradient Boosting model achieving RMSE: 17.25 cycles** (8.6% of lifespan, 68.9% improvement over baseline).
- Achieved **state-of-the-art performance** (R² = 0.950) competitive with literature benchmarks (12-18 cycles).
- Validated the model using engine-level train/test split to prevent data leakage.

**Tech Stack:** Python, Time-Series Analysis, Gradient Boosting, Random Forest, Feature Engineering
**Business Impact:** Enables precise maintenance scheduling with 2-3 week lead time; €18M+ annual savings per 100-aircraft fleet.

**[View Project →](./nasa-turbofan-predictive-maintenance/)**

---

## 🛠️ Technical Skills

**Programming & Tools:**
- **Python:** Pandas, NumPy, Scikit-Learn, SHAP, Matplotlib, Seaborn, SciPy
- **Data Engineering:** SQL, Data Cleaning, Feature Engineering, ETL Pipelines
- **Analytical Instrumentation:** LC-MS, GC-MS, HPLC, Spectroscopy (UV-Vis, FTIR), Electrochemistry
- **Workflow:** Git, GitHub, Jupyter Notebooks, VS Code, Linux/Bash, LaTeX

**Machine Learning & Analytics:**
- **Core ML:** Scikit-Learn, Random Forest, Gradient Boosting, Lasso/Ridge Regression, SVM
- **Specialized Applications:** Time-Series Analysis, Sensor Drift Detection, Anomaly Detection, Predictive Maintenance
- **Chemometrics:** PCA, HCA, PLS, Multivariate Analysis, Spectral Data Processing
- **Validation:** Stratified Cross-Validation, Outlier Detection, Drift Monitoring, DOE (Design of Experiments)
- **Explainability:** SHAP, Permutation Importance, Partial Dependence Plots, Feature Importance Analysis

**Domain Expertise & Research Background:**
- **Sensor Systems & IoT:** 10+ years developing and optimizing biosensors, analytical instruments, and quality control systems
- **Biosensors & Diagnostics:** Molecularly imprinted polymers (MIPs) for virus detection, real-time diagnostic systems (published in *Advanced Materials Interfaces*, 20+ citations)
- **Industrial R&D:** Research Group Leader at Hahn-Schickard, Germany (2022-2024) - managed projects, coordinated industry partnerships, led EU project proposals
- **Academic Background:** PhD (USP-Brazil), Humboldt Fellow (Ulm University), Former Professor (UFU, 2015-2021)
- **Publications:** 50 peer-reviewed papers, 1,266 citations (Web of Science), h-index 18
- **Instrumentation:** LC-MS, GC-MS, HPLC, Spectroscopy (UV-Vis, FTIR, Raman), Electrochemical sensors
- **Sustainability Focus:** Developed low-cost, portable analytical devices and environmentally friendly methods

---

## 📁 Repository Structure

```text
data-science-portfolio/
├── README.md (this file)
├── data/
│   └── human_cachexia.csv              # Shared data folder
├── metabolomics-biomarker-discovery/    # Project 1: Pharma/Biotech
│   ├── 01_chemometric_eda.ipynb
│   ├── 02_biomarker_ml.ipynb
│   ├── 03_shap_interpretation.ipynb
│   ├── README.md
│   └── requirements.txt
├── gas-sensor-drift-monitoring/         # Project 2: QA/Industrial IoT
│   ├── 01_visualizing_the_drift.ipynb
│   ├── 02_model_decay_analysis.ipynb
│   ├── 03_adaptive_calibration.ipynb
│   ├── README.md
│   └── requirements.txt
└── nasa-turbofan-predictive-maintenance/# Project 3: Engineering
    ├── 01_data_exploration.ipynb
    ├── 02_data_quality_and_drift.ipynb
    ├── 03_predictive_modeling.ipynb
    └── README.md
```

**Note:** Large data files (gas-sensor batches, turbofan datasets, retail data) are gitignored. For local reproduction, see individual project READMEs for data source links.
---

## 📫 Let's Connect

**Seeking:** Data Scientist / ML Engineer roles in Germany

**Target Industries:** 
- 🧬 Life Sciences & Pharma (Roche, Bayer, Merck, Sartorius)
- 🔬 Diagnostics & Medical Devices (Siemens Healthineers, Bruker, Thermo Fisher)
- 🏭 Industry 4.0 / IIoT (Siemens, Bosch, SAP)
- 🧪 Analytical Instrumentation (PerkinElmer, Agilent)

**Based in:** Germany (2020-2024) | Open to relocation within Germany  
**Work Authorization:** [Specify: EU Blue Card eligible / Current visa status]

**Contact:**
- 📧 **Email:** [alexdbatista@gmail.com](mailto:alexdbatista@gmail.com)
- 🔗 **LinkedIn:** [linkedin.com/in/alexdbatista](https://linkedin.com/in/alexdbatista)
- 💻 **GitHub:** [github.com/alexdbatista](https://github.com/alexdbatista)

---

## 🎓 Academic Credentials Summary
*For German employers familiar with academic titles:*

- **Dr. rer. nat. (equivalent)** - PhD, Universidade de São Paulo (USP)
- **Humboldt Research Fellow** - Alexander von Humboldt Foundation (2020-2021)
- **Former Professor** (UFU, Brazil, 2015-2021) - 6 years teaching & research
- **Research Group Leader** (Hahn-Schickard, Germany, 2022-2024)
- **50 peer-reviewed publications** | h-index: 18 | 1,266 citations

**Languages:** Portuguese (Native) | English (Fluent) | German (B1 Intermediate)
