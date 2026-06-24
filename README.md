# 🛡️ FraudShield AI
### AI-Powered Credit Card Fraud Detection & Customer Segmentation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3b82f6?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-f59e0b?style=flat-square&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-ef4444?style=flat-square&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-6.x-8b5cf6?style=flat-square&logo=plotly&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-PaySim%206.35M-10b981?style=flat-square)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9991-10b981?style=flat-square)
![F1](https://img.shields.io/badge/F1%20Score-99.82%25-10b981?style=flat-square)

**End-to-End Machine Learning Portfolio Project**

[Overview](#-overview) · [Results](#-results) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Dashboard](#️-dashboard) · [Notebooks](#-notebooks) · [Project Structure](#-project-structure)

</div>

---

## 📌 Overview

FraudShield AI is a production-grade end-to-end machine learning system that:

- **Detects fraudulent transactions** in real-time using a tuned Random Forest classifier
- **Segments 2.77M customers** into actionable behavioural clusters via K-Means
- **Surfaces insights** through a 5-page interactive Streamlit dashboard
- **Trains on the full PaySim dataset** (6,354,407 synthetic mobile money transactions)

The project demonstrates the complete ML lifecycle — from raw data ingestion through feature engineering, model training, evaluation, deployment, and business intelligence.

---

## 🏆 Results

### Fraud Detection Model (Random Forest — Tuned)

| Metric | Score |
|---|:---:|
| **ROC-AUC** | **0.9991** |
| **F1 Score** | **99.82%** |
| **Precision** | **99.94%** |
| **Recall** | **99.70%** |
| **Accuracy** | **99.99%** |
| CV F1 (5-fold mean) | 0.9976 ± 0.0006 |
| CV AUC (5-fold mean) | 0.9989 ± 0.0007 |

### Best Hyperparameters (GridSearchCV)

| Parameter | Value |
|---|---|
| `n_estimators` | 50 |
| `max_depth` | 10 |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 1 |
| `class_weight` | `'balanced'` |

### Customer Segmentation (K-Means, K=4)

| Segment | Share | Profile |
|---|---|---|
| Low-Risk Regular Customers | ~68% | Low frequency, zero fraud history |
| High-Value Customers | ~15% | Large balances, high spending |
| High-Value Customers (Group 2) | ~11% | Multi-transaction, high volume |
| Fraud-Prone Customers | ~6% | High fraud scores, confirmed fraud history |

---

## 🏗️ Architecture

```
PaySim Dataset (6.35M rows)
        │
        ▼
┌─────────────────────┐
│  Data Preprocessing │  clean_data() · filter_fraud_susceptible()
│  (TRANSFER+CASH_OUT)│  stratified 80/20 split
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Feature Engineering │  13 domain features · StandardScaler
│  (src/feature_eng.) │  balance diffs · amount ratio · velocity
└────────┬────────────┘
         │
         ├──────────────────────────────────────┐
         ▼                                      ▼
┌────────────────────┐              ┌───────────────────────┐
│  Fraud Detection   │              │  Customer Segmentation│
│  Random Forest     │              │  K-Means (K=4)        │
│  GridSearchCV opt  │              │  PCA 2-D viz          │
│  ROC-AUC: 0.9991  │              │  2.77M customers      │
└────────┬───────────┘              └───────────┬───────────┘
         │                                      │
         ▼                                      ▼
┌────────────────────┐              ┌───────────────────────┐
│ scored_transactions│              │  customer_segments    │
│ .csv (6.35M rows) │              │  .csv (2.77M rows)    │
└────────────────────┘              └───────────────────────┘
         │                                      │
         └──────────────┬───────────────────────┘
                        ▼
            ┌───────────────────────┐
            │  Streamlit Dashboard  │
            │  5 interactive pages  │
            │  localhost:8501       │
            └───────────────────────┘
```

---

## ⚡ Quick Start

### 1. Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset

Download the **PaySim** dataset from Kaggle and place it at:

```
data/raw/PS_20174392719_1491204439457_log.csv
```

- **Source**: https://www.kaggle.com/datasets/ealaxi/paysim1
- **Size**: ~470 MB (6,354,407 rows)

### 3. Run the Training Pipeline

```bash
python -m src.train_pipeline
```

This single command executes the full pipeline (~30 min):
- Loads and cleans data
- Engineers 13 features
- Trains Baseline → Weighted → SMOTE → Tuned RF
- Runs GridSearchCV + 5-fold CV
- Scores all 6.35M transactions
- Clusters 2.77M customers with K-Means
- Saves all models, data, and plots

### 4. Launch the Dashboard

```bash
# From project root
python -m streamlit run dashboard/app.py

# OR
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 5. Explore Notebooks (Optional)

```bash
# Notebooks are pre-built; open them directly:
jupyter notebook notebooks/
```

Or regenerate them fresh:

```bash
python generate_notebooks.py
```

---

## 🖥️ Dashboard

The Streamlit dashboard has **5 pages**:

| Page | Description |
|---|---|
| **🏠 Overview** | Executive KPIs, model architecture card, pipeline stages, top-10 feature importance chart, artifact inventory |
| **🔍 Fraud Detection** | Real-time single transaction scoring with animated gauge · batch CSV upload and download |
| **📊 Transaction Explorer** | 500K transactions · sidebar filters (type, risk, amount, score, step) · fraud score histogram · time-series · risk pie · amount boxplots · sortable data table |
| **👥 Customer Segmentation** | PCA 2-D scatter map · segment radar charts · bar comparison · full customer data table with export |
| **📈 Model Performance** | Confusion matrix · ROC curve · 5-fold CV stability chart · feature importance · strategy comparison · GridSearchCV hyperparameter audit · K-Means elbow curve |

---

## 📓 Notebooks

| Notebook | Content |
|---|---|
| [`01_EDA.ipynb`](notebooks/01_EDA.ipynb) | Dataset schema, class imbalance, type distribution, amount analysis, temporal patterns, balance anomaly deep-dive, correlation matrix |
| [`02_Preprocessing.ipynb`](notebooks/02_Preprocessing.ipynb) | Data cleaning, fraud-type filtering, stratified split, feature engineering walkthrough, one-hot encoding, StandardScaler |
| [`03_Fraud_Model.ipynb`](notebooks/03_Fraud_Model.ipynb) | Baseline/weighted/SMOTE comparison, GridSearchCV, confusion matrix, ROC curve, cross-validation, feature importance, fraud scoring |
| [`04_Segmentation.ipynb`](notebooks/04_Segmentation.ipynb) | Customer aggregation, K-Means elbow, K=4 training, PCA visualisation, radar profile chart, business segment insights |

---

## 📁 Project Structure

```
End-to-End ML Fraud Detection/
│
├── app.py                          # Root Streamlit entry point
├── generate_notebooks.py           # Script to regenerate notebooks
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── src/                            # Core ML modules
│   ├── __init__.py
│   ├── utils.py                    # Logging, model save/load helpers
│   ├── data_preprocessing.py       # Load, clean, filter, split
│   ├── feature_engineering.py      # 13 domain features + scaling
│   ├── fraud_model.py              # RF variants, GridSearchCV, scoring
│   ├── segmentation.py             # Customer aggregation, K-Means, PCA
│   ├── evaluation.py               # Metrics, plots, cross-validation
│   └── train_pipeline.py           # Master orchestrator (runs everything)
│
├── dashboard/
│   ├── app.py                      # Dashboard homepage (multipage entry)
│   └── pages/
│       ├── 1_fraud_detection.py    # Real-time scoring + batch upload
│       ├── 2_transaction_explorer.py  # Interactive transaction analytics
│       ├── 3_customer_segmentation.py # Cluster map + segment profiles
│       └── 4_model_performance.py     # Model evaluation & audit
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Fraud_Model.ipynb
│   └── 04_Segmentation.ipynb
│
├── models/                         # Saved model artifacts
│   ├── random_forest.pkl           # Tuned RF classifier
│   ├── scaler.pkl                  # Transaction feature scaler
│   ├── kmeans.pkl                  # K-Means (K=4)
│   └── scaler_kmeans.pkl           # Customer feature scaler
│
├── data/
│   ├── raw/                        # PaySim CSV (not committed to git)
│   └── processed/
│       ├── scored_transactions.csv # 6.35M rows + fraud scores
│       └── customer_segments.csv   # 2.77M customers + cluster labels
│
├── reports/
│   └── figures/                    # All generated plots
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── feature_importance.png
│       ├── elbow_curve.png
│       └── pca_clusters.png
│
└── tests/                          # Unit tests
```

---

## 🔬 Feature Engineering

All 13 model features are domain-engineered from raw PaySim columns:

| Feature | Formula / Logic | Why It Matters |
|---|---|---|
| `balance_difference_orig` | `oldBal − newBal − amount` | ≈0 when account is fully drained (primary fraud signal, 48.3% importance) |
| `balance_difference_dest` | `newBalDest − oldBalDest − amount` | Receiver balance often unchanged in fraud (PaySim anomaly) |
| `transaction_amount_ratio` | `amount / oldBal` | ≈1.0 when fraudster transfers entire balance |
| `dest_is_merchant` | `1 if nameDest starts with 'M'` | Fraud rarely targets merchant accounts |
| `transaction_velocity` | `# txns by sender in same step` | Rapid repeat transactions indicate automated fraud |
| `type_TRANSFER` | One-hot encoding | 100% of TRANSFER fraud |
| `type_CASH_OUT` | One-hot encoding | Complementary fraud channel |
| `step`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest` | Raw scaled features | Direct transaction context |

---

## 🧠 Modelling Decisions

| Decision | Rationale |
|---|---|
| **Filter to TRANSFER + CASH_OUT** | 100% of fraud occurs in these 2 types — removes noise from 5 irrelevant types |
| **Random Forest** | Handles class imbalance well, interpretable via feature importance, robust to outliers |
| **`class_weight='balanced'`** | Corrects for 773:1 class ratio without synthetic data generation |
| **GridSearchCV on 20K sample** | Full-dataset tuning would take hours; stratified sample gives equivalent results |
| **K=4 clusters** | Elbow curve shows clear bend at K=4; yields interpretable business segments |
| **PCA sample = 50K** | Plotting 2.77M points causes memory/rendering issues |

---

## 📦 Dependencies

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
imbalanced-learn>=0.11
matplotlib>=3.7
seaborn>=0.12
plotly>=5.0
streamlit>=1.30
joblib>=1.3
nbformat>=5.9
jupyter
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

**PaySim Mobile Money Fraud Simulation**

| Property | Value |
|---|---|
| Source | Kaggle (ealaxi/paysim1) |
| Rows | 6,354,407 |
| Columns | 11 |
| Fraud cases | 8,213 (0.129%) |
| Time span | 31 days (744 hourly steps) |
| Transaction types | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
| Fraud-prone types | TRANSFER, CASH_OUT only |

---

## 🚀 Key Achievements

- ✅ **ROC-AUC of 0.9991** — near-perfect class separation
- ✅ **Recalls 99.70% of all fraud** while maintaining 99.94% precision (≈1 false positive per 1,643 fraud cases)
- ✅ **Near-zero cross-validation variance** — proves robust generalisation, no overfitting
- ✅ **2.77M customers segmented** in real-time ready clusters
- ✅ **5-page production dashboard** with real-time scoring, batch upload, and interactive analytics
- ✅ **Fully modular codebase** — each `src/` module is independently testable

---

## 👤 Author

Built as an end-to-end ML portfolio project demonstrating:
- ML engineering (feature design, model selection, hyperparameter tuning)
- Data engineering (large-scale processing, stratified sampling)
- ML Ops (model serialisation, pipeline orchestration)
- Product thinking (dashboard UX, business segment labelling)

---

<div align="center">
<strong>FraudShield AI</strong> · Random Forest + K-Means · PaySim · ROC-AUC: 0.9991
</div>
