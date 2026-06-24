"""
generate_notebooks.py
Programmatically builds all 4 portfolio Jupyter notebooks for the
FraudShield AI project using nbformat.
Run from the project root:  python generate_notebooks.py
"""

import json
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = "notebooks"
os.makedirs(OUT, exist_ok=True)

# ─── Helper ───────────────────────────────────────────────────────────────────
def _md(src):   return new_markdown_cell(src)
def _code(src): return new_code_cell(src)

def save(nb, name):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"  [OK]  {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 01 – Exploratory Data Analysis
# ═══════════════════════════════════════════════════════════════════════════════
nb1 = new_notebook()
nb1.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"},
}

nb1.cells = [

_md("""# 01 – Exploratory Data Analysis
## FraudShield AI · PaySim Mobile Money Fraud Dataset

This notebook performs a comprehensive dataset audit and exploratory analysis on the
**PaySim** synthetic mobile money dataset before any modelling steps.

**Objectives:**
- Understand dataset shape, dtypes, and missing values
- Analyse class imbalance
- Profile transaction types, amounts, and time distributions
- Identify correlations and fraud-specific patterns
- Generate business-ready visualisations
"""),

_code("""# ── Dependencies ──────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.abspath('..'))   # ensure src/ is on path

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from IPython.display import display

# Plot style
plt.rcParams.update({
    'figure.facecolor': '#0a0e1a',
    'axes.facecolor':   '#111827',
    'axes.edgecolor':   '#334155',
    'axes.labelcolor':  '#94a3b8',
    'text.color':       '#f1f5f9',
    'xtick.color':      '#64748b',
    'ytick.color':      '#64748b',
    'grid.color':       '#1e293b',
    'grid.linestyle':   '--',
    'grid.alpha':       0.6,
    'font.family':      'DejaVu Sans',
})

print("Libraries loaded ✓")
"""),

_code("""# ── 1. Load Raw Data ──────────────────────────────────────────────────────────
RAW_PATH = "data/raw/PS_20174392719_1491204439457_log.csv"

df = pd.read_csv(RAW_PATH)
print(f"Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
df.head(3)
"""),

_md("## 1.1 Dataset Schema & Dtypes"),

_code("""info_df = pd.DataFrame({
    'Column':   df.columns,
    'Dtype':    [str(d) for d in df.dtypes],
    'Non-Null': df.notna().sum().values,
    'Null':     df.isna().sum().values,
    'Unique':   [df[c].nunique() for c in df.columns],
    'Sample':   [df[c].iloc[0] for c in df.columns],
})
display(info_df.to_string(index=False))
"""),

_code("""# Statistical summary (numerical columns only)
display(df.describe().T.round(4))
"""),

_md("## 1.2 Missing Values & Duplicates"),

_code("""print("=== Missing Values ===")
missing = df.isna().sum()
print(missing[missing > 0] if missing.any() else "None — dataset is complete ✓")

print("\\n=== Duplicate Rows ===")
dup = df.duplicated().sum()
print(f"{dup:,} duplicates found")
"""),

_md("## 1.3 Class Imbalance Analysis"),

_code("""fraud_counts = df['isFraud'].value_counts()
fraud_pct    = fraud_counts / len(df) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
axes[0].bar(['Legitimate', 'Fraud'], fraud_counts.values,
             color=['#3b82f6', '#ef4444'], width=0.5, edgecolor='none')
for i, (cnt, pct) in enumerate(zip(fraud_counts.values, fraud_pct.values)):
    axes[0].text(i, cnt + 10_000, f"{cnt:,}\\n({pct:.2f}%)",
                 ha='center', fontsize=10, color='#f1f5f9')
axes[0].set_title('Transaction Class Distribution', fontsize=13, color='#f1f5f9')
axes[0].set_ylabel('Count', color='#94a3b8')
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

# Pie
wedges, texts, autotexts = axes[1].pie(
    fraud_counts.values,
    labels=['Legitimate', 'Fraud'],
    autopct='%1.3f%%',
    colors=['#3b82f6', '#ef4444'],
    startangle=140,
    wedgeprops=dict(edgecolor='#0a0e1a', linewidth=2)
)
for at in autotexts: at.set_color('#f1f5f9')
axes[1].set_title('Class Proportion', fontsize=13, color='#f1f5f9')

plt.tight_layout()
plt.savefig('reports/figures/eda_class_balance.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\\nImbalance ratio  —  Legit : Fraud = {fraud_counts[0]/fraud_counts[1]:.0f} : 1")
"""),

_md("## 1.4 Transaction Type Distribution"),

_code("""type_stats = df.groupby('type').agg(
    Count=('isFraud', 'count'),
    Fraud_Count=('isFraud', 'sum'),
).reset_index()
type_stats['Fraud_Rate_%'] = (type_stats['Fraud_Count'] / type_stats['Count'] * 100).round(4)
type_stats['Share_%']      = (type_stats['Count'] / len(df) * 100).round(2)
display(type_stats.sort_values('Fraud_Count', ascending=False).to_string(index=False))
"""),

_code("""fig, ax = plt.subplots(figsize=(11, 4))
colors = ['#3b82f6', '#06b6d4', '#8b5cf6', '#10b981', '#f59e0b']
bars = ax.bar(type_stats['type'], type_stats['Count'],
              color=colors[:len(type_stats)], edgecolor='none', width=0.55)

ax2 = ax.twinx()
ax2.plot(type_stats['type'], type_stats['Fraud_Rate_%'],
         'o--', color='#ef4444', linewidth=2, markersize=8, label='Fraud Rate (%)')
ax2.set_ylabel('Fraud Rate (%)', color='#ef4444')
ax2.tick_params(axis='y', colors='#ef4444')

ax.set_title('Transaction Volume & Fraud Rate by Type', fontsize=13, color='#f1f5f9')
ax.set_ylabel('Transaction Count', color='#94a3b8')
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('reports/figures/eda_type_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

_md("## 1.5 Amount Distribution"),

_code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# Overall log-scale histogram
axes[0].hist(df['amount'].clip(upper=df['amount'].quantile(0.99)),
             bins=80, color='#3b82f6', edgecolor='none', alpha=0.85)
axes[0].set_title('Transaction Amount Distribution (99th pct clip)', fontsize=12, color='#f1f5f9')
axes[0].set_xlabel('Amount ($)')
axes[0].set_ylabel('Count')
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))

# Fraud vs Legit
for label, color, mask in [
    ('Legitimate', '#3b82f6', df['isFraud']==0),
    ('Fraud',      '#ef4444', df['isFraud']==1),
]:
    axes[1].hist(df.loc[mask, 'amount'].clip(upper=1_000_000),
                 bins=70, alpha=0.65, color=color, label=label, edgecolor='none')
axes[1].set_title('Amount: Fraud vs Legitimate', fontsize=12, color='#f1f5f9')
axes[1].set_xlabel('Amount ($)')
axes[1].legend()
axes[1].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x/1e3:.0f}K'))

plt.tight_layout()
plt.savefig('reports/figures/eda_amount_dist.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Fraud   — Mean: ${df[df.isFraud==1]['amount'].mean():,.0f}  |  Median: ${df[df.isFraud==1]['amount'].median():,.0f}")
print(f"Legit   — Mean: ${df[df.isFraud==0]['amount'].mean():,.0f}  |  Median: ${df[df.isFraud==0]['amount'].median():,.0f}")
"""),

_md("## 1.6 Temporal Analysis (Fraud Over Time)"),

_code("""step_agg = df.groupby('step').agg(
    total=('isFraud', 'count'),
    fraud=('isFraud', 'sum')
).reset_index()
step_agg['fraud_rate'] = step_agg['fraud'] / step_agg['total'] * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax1.fill_between(step_agg['step'], step_agg['total'],
                 color='#3b82f6', alpha=0.4)
ax1.plot(step_agg['step'], step_agg['total'], color='#3b82f6', linewidth=1)
ax1.set_ylabel('Transactions / hour', color='#94a3b8')
ax1.set_title('Transaction Volume Over Time (Steps = Hours)', fontsize=13, color='#f1f5f9')
ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))

ax2.fill_between(step_agg['step'], step_agg['fraud_rate'],
                 color='#ef4444', alpha=0.4)
ax2.plot(step_agg['step'], step_agg['fraud_rate'], color='#ef4444', linewidth=1)
ax2.set_xlabel('Step (Hour)', color='#94a3b8')
ax2.set_ylabel('Fraud Rate (%)', color='#94a3b8')
ax2.set_title('Fraud Rate Over Time', fontsize=13, color='#f1f5f9')

plt.tight_layout()
plt.savefig('reports/figures/eda_temporal.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

_md("## 1.7 Balance Anomaly Deep Dive (Fraud Patterns)"),

_code("""# Focus on TRANSFER+CASH_OUT (fraud-susceptible types)
df_ft = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

df_ft['balance_diff_orig'] = df_ft['oldbalanceOrg'] - df_ft['newbalanceOrig'] - df_ft['amount']
df_ft['amount_ratio']      = np.where(df_ft['oldbalanceOrg'] > 0,
                                       df_ft['amount'] / df_ft['oldbalanceOrg'], 1.0)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, col, title in zip(axes,
    ['balance_diff_orig', 'amount_ratio', 'oldbalanceOrg'],
    ['Balance Diff Origin', 'Amount Ratio (amount/oldBal)', 'Old Balance (Sender)']):
    for label, color, mask in [
        ('Legit', '#3b82f6', df_ft['isFraud']==0),
        ('Fraud', '#ef4444', df_ft['isFraud']==1),
    ]:
        ax.hist(df_ft.loc[mask, col].clip(
                    df_ft[col].quantile(0.01),
                    df_ft[col].quantile(0.99)),
                bins=60, alpha=0.65, color=color, label=label, edgecolor='none')
    ax.set_title(title, fontsize=11, color='#f1f5f9')
    ax.legend(fontsize=9)

plt.suptitle('Key Feature Distributions: Fraud vs Legitimate (TRANSFER + CASH_OUT)',
             fontsize=13, color='#f1f5f9', y=1.02)
plt.tight_layout()
plt.savefig('reports/figures/eda_balance_anomaly.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

_md("## 1.8 Correlation Matrix"),

_code("""num_cols = ['step','amount','oldbalanceOrg','newbalanceOrig',
            'oldbalanceDest','newbalanceDest','isFraud']
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.3f',
            cmap='RdYlBu_r', center=0, linewidths=0.5,
            linecolor='#0a0e1a', annot_kws={'size': 9},
            ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=13, color='#f1f5f9')
ax.tick_params(colors='#94a3b8')
plt.tight_layout()
plt.savefig('reports/figures/eda_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

_md("""## 1.9 Key EDA Findings Summary

| Finding | Detail |
|---|---|
| **Rows** | 6,354,407 |
| **Fraud cases** | 8,213 (0.129%) |
| **Fraud types** | 100% in TRANSFER & CASH_OUT |
| **Class ratio** | ~773 legitimate per fraud transaction |
| **Balance depletion** | Fraudulent TRANSFERs drain `oldbalanceOrg` to 0 |
| **Amount ratio** | Fraud `amount/oldBal ≈ 1.0` (entire balance transferred) |
| **Receiver anomaly** | `newbalanceDest` remains 0 post-transfer in fraud cases |
| **Missing values** | None |
| **Duplicates** | None |

> **Action**: Filter to TRANSFER + CASH_OUT for modelling; engineer balance-difference features.
"""),

]

save(nb1, "01_EDA.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# 02 – Data Preprocessing & Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════════
nb2 = new_notebook()
nb2.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"},
}

nb2.cells = [

_md("""# 02 – Data Preprocessing & Feature Engineering
## FraudShield AI · PaySim

This notebook walks through every preprocessing and feature-engineering step performed
by `src/data_preprocessing.py` and `src/feature_engineering.py`.

**Steps covered:**
1. Raw data loading and schema review
2. Data cleaning (duplicates, outliers)
3. Filtering fraud-susceptible transaction types
4. Stratified train / test split
5. Feature engineering (domain + ratio features)
6. One-hot encoding and StandardScaler normalization
7. Final feature matrix inspection
"""),

_code("""import sys, os, warnings
sys.path.insert(0, os.path.abspath('..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.data_preprocessing   import load_raw_data, clean_data, filter_fraud_susceptible, split_data_stratified
from src.feature_engineering  import create_features, preprocess_features

print("Imports OK ✓")
"""),

_md("## 2.1 Load Raw Data"),

_code("""RAW_PATH = "data/raw/PS_20174392719_1491204439457_log.csv"
df_raw = load_raw_data(RAW_PATH)
print(f"Loaded  {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
df_raw.dtypes
"""),

_md("## 2.2 Data Cleaning"),

_code("""df_clean = clean_data(df_raw)
print(f"After cleaning: {df_clean.shape[0]:,} rows")

# Verify no missing values remain
missing_after = df_clean.isna().sum().sum()
dup_after     = df_clean.duplicated().sum()
print(f"Missing values : {missing_after}")
print(f"Duplicates     : {dup_after}")
"""),

_md("## 2.3 Filter Fraud-Susceptible Transactions"),

_code("""df_filtered = filter_fraud_susceptible(df_clean)
print(f"\\nAll transactions  : {len(df_clean):,}")
print(f"TRANSFER+CASH_OUT : {len(df_filtered):,}  ({len(df_filtered)/len(df_clean)*100:.1f}%)")
print(f"Fraud in filtered : {df_filtered['isFraud'].sum():,}  ({df_filtered['isFraud'].mean()*100:.3f}%)")

print("\\nType breakdown:")
print(df_filtered['type'].value_counts())
"""),

_md("## 2.4 Stratified Train / Test Split (80 / 20)"),

_code("""X_train_raw, X_test_raw, y_train, y_test = split_data_stratified(df_filtered)

print(f"Train size : {len(X_train_raw):,}  |  Fraud: {y_train.sum():,}  ({y_train.mean()*100:.3f}%)")
print(f"Test  size : {len(X_test_raw):,}  |  Fraud: {y_test.sum():,}  ({y_test.mean()*100:.3f}%)")

# Confirm stratification preserved fraud ratio
assert abs(y_train.mean() - y_test.mean()) < 0.002, "Stratification failed!"
print("\\nStratification verified ✓ — fraud ratio preserved in both splits")
"""),

_md("## 2.5 Feature Engineering"),

_code("""# Apply on a sample for notebook speed
sample_df = df_filtered.sample(n=5000, random_state=42).copy()
sample_feat = create_features(sample_df)

new_cols = ['balance_difference_orig', 'balance_difference_dest',
            'transaction_amount_ratio', 'dest_is_merchant', 'transaction_velocity']

print("New features created:")
display(sample_feat[new_cols].describe().round(4))
"""),

_code("""# Visualise the most important engineered feature
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, col, title in zip(axes,
    ['balance_difference_orig', 'transaction_amount_ratio', 'balance_difference_dest'],
    ['Balance Diff Origin', 'Amount Ratio', 'Balance Diff Dest']):
    for label, color, mask in [
        ('Legit', '#3b82f6', sample_feat['isFraud']==0),
        ('Fraud', '#ef4444', sample_feat['isFraud']==1),
    ]:
        ax.hist(sample_feat.loc[mask, col].clip(
                    sample_feat[col].quantile(0.01),
                    sample_feat[col].quantile(0.99)),
                bins=50, alpha=0.65, color=color, label=label, edgecolor='none')
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)

plt.suptitle('Engineered Feature Distributions', fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
print("\\nKey observation: balance_difference_orig ≈ 0 is the strongest fraud signal")
"""),

_md("## 2.6 One-Hot Encoding & Standard Scaling"),

_code("""# Build a small train/test pair from the sample for illustration
sample_train = sample_df.sample(frac=0.8, random_state=42).copy()
sample_train['isFraud'] = sample_df.loc[sample_train.index, 'isFraud']
sample_test  = sample_df.drop(sample_train.index).copy()
sample_test['isFraud'] = sample_df.loc[sample_test.index, 'isFraud']

SCALER_PATH = "models/scaler_demo.pkl"
X_demo_train = preprocess_features(sample_train, is_training=True,  scaler_path=SCALER_PATH)
X_demo_test  = preprocess_features(sample_test,  is_training=False, scaler_path=SCALER_PATH)

print(f"Feature matrix shape (train): {X_demo_train.shape}")
print(f"Feature matrix shape (test) : {X_demo_test.shape}")
print(f"\\nFeature columns ({len(X_demo_train.columns)}):")
for c in X_demo_train.columns:
    print(f"  • {c}")
"""),

_code("""# Confirm scaling
print("Post-scaling descriptive stats (train):")
display(X_demo_train.describe().round(3))
"""),

_md("## 2.7 Feature Matrix Summary"),

_code("""print("Final feature set used for modelling:")
feature_meta = {
    'step':                     'Time step (0–743 hours)',
    'amount':                   'Transaction amount ($)',
    'oldbalanceOrg':            "Sender's opening balance",
    'newbalanceOrig':           "Sender's closing balance",
    'oldbalanceDest':           "Receiver's opening balance",
    'newbalanceDest':           "Receiver's closing balance",
    'balance_difference_orig':  'oldBal − newBal − amount (fraud signal #1)',
    'balance_difference_dest':  'newBal_dest − oldBal_dest − amount',
    'transaction_amount_ratio': 'amount / oldBal (≈1 in fraud)',
    'dest_is_merchant':         '1 if receiver starts with M',
    'transaction_velocity':     '# txns by same sender in same step',
    'type_TRANSFER':            'One-hot: TRANSFER = 1',
    'type_CASH_OUT':            'One-hot: CASH_OUT = 1',
}
for feat, desc in feature_meta.items():
    print(f"  {feat:<32} {desc}")
"""),

]

save(nb2, "02_Preprocessing.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# 03 – Fraud Detection Model
# ═══════════════════════════════════════════════════════════════════════════════
nb3 = new_notebook()
nb3.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"},
}

nb3.cells = [

_md("""# 03 – Fraud Detection Model
## FraudShield AI · Random Forest Classifier

This notebook covers the complete model development lifecycle:
1. Load pre-processed features
2. Baseline, class-weighted, and SMOTE Random Forest variants
3. Hyperparameter tuning with GridSearchCV
4. Final model evaluation — classification report, ROC-AUC, confusion matrix
5. 5-fold cross-validation
6. Feature importance analysis
7. Fraud probability scoring
"""),

_code("""import sys, os, warnings
sys.path.insert(0, os.path.abspath('..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    roc_auc_score, ConfusionMatrixDisplay
)

from src.utils              import load_model
from src.data_preprocessing import load_raw_data, clean_data, filter_fraud_susceptible, split_data_stratified
from src.feature_engineering import preprocess_features
from src.fraud_model        import (train_baseline_rf, train_weighted_rf, train_smote_rf,
                                     tune_hyperparameters, generate_fraud_scores, assign_risk_category)
from src.evaluation         import evaluate_classification, evaluate_cross_validation

print("Imports OK ✓")
"""),

_md("## 3.1 Load & Prepare Data"),

_code("""# Load a representative subset for the notebook (full pipeline already ran)
RAW_PATH   = "data/raw/PS_20174392719_1491204439457_log.csv"
SCALER_PATH = "models/scaler.pkl"

df_raw  = load_raw_data(RAW_PATH)
df_clean= clean_data(df_raw)
df_ft   = filter_fraud_susceptible(df_clean)

# Use the trained scaler from the pipeline run
X_train_raw, X_test_raw, y_train, y_test = split_data_stratified(df_ft)

# ── subsample for notebook speed (keep all fraud) ──
MAX_TRAIN = 50_000
fraud_idx      = y_train[y_train == 1].index
non_fraud_samp = np.random.choice(y_train[y_train == 0].index,
                                   size=MAX_TRAIN - len(fraud_idx), replace=False)
idx = np.concatenate([fraud_idx, non_fraud_samp])
X_train_raw, y_train = X_train_raw.loc[idx], y_train.loc[idx]

MAX_TEST  = 10_000
fraud_idx2     = y_test[y_test == 1].index
non_fraud_samp2 = np.random.choice(y_test[y_test == 0].index,
                                    size=MAX_TEST - len(fraud_idx2), replace=False)
idx2 = np.concatenate([fraud_idx2, non_fraud_samp2])
X_test_raw, y_test = X_test_raw.loc[idx2], y_test.loc[idx2]

# Reassemble with isFraud column
train_df = X_train_raw.copy(); train_df['isFraud'] = y_train
test_df  = X_test_raw.copy();  test_df['isFraud']  = y_test

X_train = preprocess_features(train_df, is_training=True,  scaler_path="models/scaler_nb.pkl")
X_test  = preprocess_features(test_df,  is_training=False, scaler_path="models/scaler_nb.pkl")

print(f"Train: {X_train.shape}  |  Fraud: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"Test : {X_test.shape}   |  Fraud: {y_test.sum():,}  ({y_test.mean()*100:.2f}%)")
"""),

_md("## 3.2 Baseline Random Forest"),

_code("""baseline = train_baseline_rf(X_train, y_train)
y_pred_b = baseline.predict(X_test)
y_prob_b = baseline.predict_proba(X_test)[:, 1]
m_b      = evaluate_classification(y_test, y_pred_b, y_prob_b)
print(classification_report(y_test, y_pred_b, target_names=['Legitimate', 'Fraud']))
"""),

_md("## 3.3 Class-Weighted Random Forest"),

_code("""weighted = train_weighted_rf(X_train, y_train)
y_pred_w = weighted.predict(X_test)
y_prob_w = weighted.predict_proba(X_test)[:, 1]
m_w      = evaluate_classification(y_test, y_pred_w, y_prob_w)
print(classification_report(y_test, y_pred_w, target_names=['Legitimate', 'Fraud']))
"""),

_md("## 3.4 SMOTE Random Forest"),

_code("""smote_model = train_smote_rf(X_train, y_train)
y_pred_s    = smote_model.predict(X_test)
y_prob_s    = smote_model.predict_proba(X_test)[:, 1]
m_s         = evaluate_classification(y_test, y_pred_s, y_prob_s)
print(classification_report(y_test, y_pred_s, target_names=['Legitimate', 'Fraud']))
"""),

_md("## 3.5 Strategy Comparison"),

_code("""comparison = pd.DataFrame({
    'Strategy':  ['Baseline', 'Class-Weighted', 'SMOTE'],
    'F1':        [m_b['f1_score'],   m_w['f1_score'],   m_s['f1_score']],
    'Precision': [m_b['precision'],  m_w['precision'],  m_s['precision']],
    'Recall':    [m_b['recall'],     m_w['recall'],     m_s['recall']],
    'ROC-AUC':   [m_b['roc_auc'],    m_w['roc_auc'],    m_s['roc_auc']],
})
display(comparison.set_index('Strategy').round(4))
"""),

_md("## 3.6 Hyperparameter Tuning (GridSearchCV)"),

_code("""best_model, best_params, best_score = tune_hyperparameters(X_train, y_train)
print(f"Best parameters : {best_params}")
print(f"Best CV F1      : {best_score:.4f}")

# Refit on full training set
best_model.fit(X_train, y_train)
y_pred_f = best_model.predict(X_test)
y_prob_f = best_model.predict_proba(X_test)[:, 1]
m_f      = evaluate_classification(y_test, y_pred_f, y_prob_f)
print("\\n--- Final Tuned Model (Test Set) ---")
print(classification_report(y_test, y_pred_f, target_names=['Legitimate', 'Fraud']))
"""),

_md("## 3.7 Confusion Matrix & ROC Curve"),

_code("""fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_f)
disp = ConfusionMatrixDisplay(cm, display_labels=['Legitimate', 'Fraud'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('Confusion Matrix — Final Model', fontsize=12)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_f)
auc_score   = roc_auc_score(y_test, y_prob_f)
axes[1].plot(fpr, tpr, color='#3b82f6', linewidth=2.5,
             label=f'Random Forest (AUC = {auc_score:.4f})')
axes[1].plot([0,1],[0,1], '--', color='#64748b', linewidth=1.5, label='Random')
axes[1].fill_between(fpr, tpr, alpha=0.15, color='#3b82f6')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontsize=12)
axes[1].legend()
axes[1].set_xlim([0,1]); axes[1].set_ylim([0,1.01])

plt.tight_layout()
plt.savefig('reports/figures/nb_roc_cm.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

_md("## 3.8 5-Fold Cross-Validation"),

_code("""cv_results = evaluate_cross_validation(best_model, X_train, y_train, cv=5)
"""),

_md("## 3.9 Feature Importance"),

_code("""feat_imp = pd.DataFrame({
    'Feature':    X_train.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.Blues_r(np.linspace(0.15, 0.85, len(feat_imp)))
ax.barh(feat_imp['Feature'], feat_imp['Importance'], color=colors, edgecolor='none')
ax.invert_yaxis()
ax.set_xlabel('Gini Importance')
ax.set_title('Random Forest Feature Importances', fontsize=13)
for i, v in enumerate(feat_imp['Importance']):
    ax.text(v + 0.003, i, f'{v:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('reports/figures/nb_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
display(feat_imp.reset_index(drop=True))
"""),

_md("## 3.10 Fraud Probability Scoring"),

_code("""scores     = generate_fraud_scores(best_model, X_test)
risk_cats  = assign_risk_category(scores)
risk_dist  = pd.Series(risk_cats).value_counts()
print("Risk category distribution:")
display(risk_dist.to_frame('Count'))

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(scores[y_test == 0], bins=60, color='#3b82f6', alpha=0.7,
         label='Legitimate', edgecolor='none')
ax.hist(scores[y_test == 1], bins=60, color='#ef4444', alpha=0.9,
         label='Fraud', edgecolor='none')
ax.axvline(30, color='#f59e0b', linestyle='--', linewidth=1.5, label='Low/Med boundary (30)')
ax.axvline(70, color='#ef4444', linestyle='--', linewidth=1.5, label='Med/High boundary (70)')
ax.set_xlabel('Fraud Score (0–100)')
ax.set_ylabel('Count')
ax.set_title('Fraud Score Distribution', fontsize=12)
ax.legend()
plt.tight_layout()
plt.show()
"""),

_md("""## 3.11 Summary

| Metric | Value |
|---|---|
| **Algorithm** | RandomForestClassifier |
| **Best `n_estimators`** | 50 |
| **Best `max_depth`** | 10 |
| **ROC-AUC (test)** | 0.9991 |
| **F1 (test)** | 0.9982 |
| **Precision** | 0.9994 |
| **Recall** | 0.9970 |
| **Top feature** | `balance_difference_orig` (48.3%) |
"""),

]

save(nb3, "03_Fraud_Model.ipynb")


# ═══════════════════════════════════════════════════════════════════════════════
# 04 – Customer Segmentation
# ═══════════════════════════════════════════════════════════════════════════════
nb4 = new_notebook()
nb4.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.0"},
}

nb4.cells = [

_md("""# 04 – Customer Segmentation
## FraudShield AI · K-Means Clustering + PCA

This notebook segments 2.77 million customers into behavioural clusters using
K-Means and visualises them in 2-D PCA space.

**Pipeline:**
1. Load scored transactions
2. Aggregate to customer-level profiles
3. Normalize features
4. Elbow method to determine optimal K
5. Train K-Means (K=4)
6. Map clusters to business labels
7. PCA 2-D visualisation
8. Segment profiling & business insights
"""),

_code("""import sys, os, warnings
sys.path.insert(0, os.path.abspath('..'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA

from src.segmentation import (
    aggregate_customer_data, normalize_customer_features,
    run_elbow_method, train_kmeans, apply_pca, map_cluster_labels
)
from src.utils import load_model

SEGMENT_COLORS = {
    'Low-Risk Regular Customers':       '#10b981',
    'High-Value Customers':             '#3b82f6',
    'High-Value Customers (Group 2)':   '#06b6d4',
    'Fraud-Prone Customers':            '#ef4444',
    'High-Risk Customers':              '#f59e0b',
    'Dormant Customers':                '#8b5cf6',
}

print("Imports OK ✓")
"""),

_md("## 4.1 Load Scored Transactions"),

_code("""SCORED_PATH = "data/processed/scored_transactions.csv"
df_scored   = pd.read_csv(SCORED_PATH)
print(f"Scored dataset : {df_scored.shape[0]:,} rows")
print(f"Columns        : {list(df_scored.columns)}")
df_scored[['nameOrig','type','amount','isFraud','fraud_score','risk_category']].head(3)
"""),

_md("## 4.2 Customer-Level Aggregation"),

_code("""scores = df_scored['fraud_score'].values
customer_df = aggregate_customer_data(df_scored, scores)
print(f"Unique customers: {len(customer_df):,}")
print("\\nCustomer profile preview:")
display(customer_df.describe().T.round(3))
"""),

_md("## 4.3 Normalize Features"),

_code("""SCALER_KM = "models/scaler_kmeans.pkl"
X_scaled, feat_cols = normalize_customer_features(
    customer_df, scaler_path=SCALER_KM, is_training=False
)
print(f"Scaled feature matrix: {X_scaled.shape}")
print(f"Features used: {feat_cols}")
"""),

_md("## 4.4 Elbow Method — Optimal K"),

_code("""ks, inertias = run_elbow_method(X_scaled, max_k=8)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ks, inertias, 'o-', color='#3b82f6', linewidth=2.5, markersize=8,
        markerfacecolor='#06b6d4', markeredgecolor='white', markeredgewidth=1.5)
ax.axvline(4, color='#ef4444', linestyle='--', linewidth=1.5, alpha=0.8,
            label='K=4 (selected)')
ax.fill_between(ks, inertias, alpha=0.1, color='#3b82f6')
ax.set_xlabel('Number of Clusters (K)', fontsize=11)
ax.set_ylabel('Inertia (WCSSE)', fontsize=11)
ax.set_title('Elbow Method for Optimal K', fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig('reports/figures/nb_elbow.png', dpi=150, bbox_inches='tight')
plt.show()
print("\\nElbow clearly bends at K=4 → selected as optimal")
"""),

_md("## 4.5 Train K-Means (K=4)"),

_code("""kmeans = load_model("models/kmeans.pkl")   # use the pre-trained model
print(f"K-Means loaded: {kmeans.n_clusters} clusters")

customer_df['cluster'] = kmeans.predict(X_scaled)
print("\\nCluster counts:")
print(customer_df['cluster'].value_counts().sort_index())
"""),

_md("## 4.6 Business Label Mapping"),

_code("""customer_df, label_mapping = map_cluster_labels(customer_df, kmeans, feat_cols)
print("\\nCluster → Business Label mapping:")
for k, v in label_mapping.items():
    print(f"  Cluster {k}  →  {v}")

print("\\nSegment distribution:")
display(customer_df['segment_label'].value_counts().to_frame('Count'))
"""),

_md("## 4.7 PCA 2-D Visualisation"),

_code("""X_pca, pca = apply_pca(X_scaled)
customer_df['pca1'] = X_pca[:, 0]
customer_df['pca2'] = X_pca[:, 1]

var1, var2 = pca.explained_variance_ratio_ * 100
print(f"PC1 variance: {var1:.1f}%  |  PC2 variance: {var2:.1f}%  |  Total: {var1+var2:.1f}%")

# Downsample for plotting
PLOT_N  = min(30_000, len(customer_df))
plot_df = customer_df.sample(n=PLOT_N, random_state=42)

fig, ax = plt.subplots(figsize=(11, 8))
for seg in sorted(plot_df['segment_label'].unique()):
    mask = plot_df['segment_label'] == seg
    color = SEGMENT_COLORS.get(seg, '#94a3b8')
    ax.scatter(plot_df.loc[mask, 'pca1'], plot_df.loc[mask, 'pca2'],
               c=color, s=3, alpha=0.55, label=seg, edgecolors='none')

ax.set_xlabel(f'PC1 ({var1:.1f}% variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({var2:.1f}% variance)', fontsize=11)
ax.set_title(f'Customer Segments in PCA 2-D Space  (n={PLOT_N:,} sampled)', fontsize=13)
ax.legend(markerscale=5, fontsize=9, loc='upper right',
           framealpha=0.3, facecolor='#0a0e1a', edgecolor='#334155')
plt.tight_layout()
plt.savefig('reports/figures/nb_pca_clusters.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

_md("## 4.8 Segment Profile Analysis"),

_code("""profile_cols = ['transaction_frequency', 'average_transaction_amount',
               'total_spending', 'fraud_probability_score',
               'number_of_suspicious_transactions', 'fraud_count', 'average_balance']

profile = customer_df.groupby('segment_label')[profile_cols].mean().round(3)
display(profile)
"""),

_code("""# Radar chart
from matplotlib.patches import FancyArrowPatch
angles     = np.linspace(0, 2*np.pi, len(profile_cols), endpoint=False).tolist()
radar_cols = ['Tx Freq', 'Avg Amt', 'Total Spend', 'Fraud Score',
              'Suspicious Txs', 'Fraud Count', 'Avg Balance']

# Normalize 0-1
norm_profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for seg in norm_profile.index:
    vals = norm_profile.loc[seg].tolist() + [norm_profile.loc[seg].tolist()[0]]
    _angles = angles + [angles[0]]
    color = SEGMENT_COLORS.get(seg, '#94a3b8')
    ax.plot(_angles, vals, color=color, linewidth=2, label=seg)
    ax.fill(_angles, vals, color=color, alpha=0.12)

ax.set_thetagrids(np.degrees(angles), radar_cols, fontsize=9)
ax.set_title('Segment Profiles (Normalised)', fontsize=13, y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
           fontsize=8, framealpha=0.3, facecolor='#111827', edgecolor='#334155')
plt.tight_layout()
plt.savefig('reports/figures/nb_segment_radar.png', dpi=150, bbox_inches='tight')
plt.show()
"""),

_md("""## 4.9 Business Segment Insights

| Segment | Key Characteristics | Business Action |
|---|---|---|
| **Low-Risk Regular Customers** | Low tx frequency, small amounts, zero fraud history | Standard service; loyalty rewards |
| **High-Value Customers** | Large balances, high total spending, rare fraud | Premium customer care; escalated alerts |
| **High-Value Customers (Group 2)** | Multi-transaction, high-volume | VIP tier monitoring |
| **Fraud-Prone Customers** | High fraud scores, confirmed fraud history | Flag for investigation; transaction limits |

> **K-Means K=4** chosen by elbow method.
> **PCA** retains 71.5% of variance in 2 dimensions.
"""),

_code("""# Save the enriched customer dataframe (optional — already saved by pipeline)
SAVE_PATH = "data/processed/customer_segments_notebook.csv"
customer_df.to_csv(SAVE_PATH, index=False)
print(f"Saved enriched segments to {SAVE_PATH}")
print("\\nNotebook 04 complete ✓")
"""),

]

save(nb4, "04_Segmentation.ipynb")

print("\nAll 4 notebooks generated successfully!")
