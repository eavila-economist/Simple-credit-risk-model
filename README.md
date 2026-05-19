# Credit Risk PD Model
### End-to-End Probability of Default Modelling with Regulatory Framework Integration

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Framework](https://img.shields.io/badge/Framework-IFRS%209%20%7C%20Basel%20III%20%7C%20SR%2011--7-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## Overview

This repository presents the complete development and validation of a **Probability of Default (PD) model** for a consumer lending portfolio, implemented to professional credit risk standards. The project covers the full modelling lifecycle — from raw data ingestion and exploratory analysis through feature engineering, model development, regulatory capital calculation and explainability — structured to satisfy the requirements of **IFRS 9**, **Basel III IRB** and **SR 11-7** model governance.

The project is intended as a demonstration of quantitative credit risk modelling at a practitioner level. It is not an academic exercise. Every modelling decision is documented with its regulatory and economic justification, and every output is connected to a concrete business or regulatory consequence.

---

## Key Results

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| AUC-ROC | 0.8584 | 0.8643 |
| Gini Coefficient | 0.7168 | 0.7286 |
| KS Statistic | 0.5697 | 0.5773 |
| Log-loss (calibrated) | 0.1840 | 0.1803 |
| Mean predicted PD | 6.96% | 6.66% |

**Champion-Challenger Decision:** Logistic Regression retained as champion. XGBoost AUC improvement of 0.59 percentage points is statistically significant (DeLong test p < 0.001) but falls below the 2-3 percentage point threshold that justifies replacing an interpretable regulatory model with a black-box alternative in a Basel IRB context.

| Capital Metric | Value |
|---|---|
| Portfolio exposure | $1,275,000,000 |
| SA RWA | $1,020,127,500 |
| SA capital required (10.5%) | $107,113,388 |
| IRB RWA | TBD — Section 06 |
| IRB capital required | TBD — Section 06 |
| Capital relief | TBD — Section 06 |

---

## Regulatory Framework

### IFRS 9 — Financial Instruments
IFRS 9 replaced the IAS 39 incurred loss model with an Expected Credit Loss framework, requiring banks to provision for losses before they occur using forward-looking information. This project implements the three-stage ECL model — 12-month ECL for Stage 1 performing exposures, lifetime ECL for Stage 2 and Stage 3 — using the model's PD estimates as the core input. The SICR assessment framework is operationalised using three complementary triggers: a quantitative PD doubling criterion, an 80% revolving utilisation threshold, and the regulatory 30-day past due backstop.

### Basel III — Internal Ratings Based Approach
Basel III allows banks that receive regulatory approval to use internal PD models to calculate risk-weighted assets rather than the blunt risk weights of the Standardised Approach. This project applies the Basel III ASRF-derived IRB capital formula to each borrower in the portfolio using model-estimated PDs, demonstrating the capital relief achievable through risk-sensitive internal modelling relative to the SA baseline. Foundation IRB regulatory LGD of 75% and EAD equal to outstanding balance are applied for retail unsecured exposures.

### SR 11-7 — Model Risk Management
The Federal Reserve's SR 11-7 guidance on model risk management defines the gold standard for how financial institutions develop, validate and monitor quantitative models. This project is structured to satisfy SR 11-7 validation requirements — with explicit documentation of conceptual soundness, data quality, replication, benchmarking, sensitivity analysis and ongoing monitoring — making it suitable as a model development document for submission to an independent validation team.

---

## Repository Structure
```
credit-risk-pd-model
│
├── README.md
├── requirements.txt
│
├── data/
│   └── README.md                         ← Dataset description and download instructions
│
├── notebooks/
│   ├── 01_EDA_and_Cleaning.ipynb         ← Data quality, distributions, bivariate analysis
│   ├── 02_Feature_Engineering.ipynb      ← WoE transformation, IV calculation, feature selection
│   ├── 03_Model_Development.ipynb        ← Logistic regression, Lasso, C tuning, XGBoost
│   ├── 04_Model_Evaluation.ipynb         ← AUC, Gini, KS, calibration, champion-challenger
│   ├── 05_IFRS9_ECL.ipynb               ← Three-stage ECL calculation, SICR, provisioning
│   ├── 06_Basel_IRB_Capital.ipynb        ← ASRF formula, IRB RWA, SA vs IRB comparison
│   └── 07_SHAP_Explainability.ipynb      ← Global importance, individual explanations
│
├── src/
│   ├── init.py
│   ├── cleaning.py                       ← Data cleaning pipeline functions
│   ├── woe_iv.py                         ← WoE and IV calculation and transformation
│   ├── model.py                          ← Model training, evaluation and calibration
│   ├── ifrs9.py                          ← ECL calculation and staging functions
│   ├── basel.py                          ← IRB capital formula implementation
│   └── monitoring.py                     ← PSI, CSI and ongoing monitoring framework
│
└── docs/
└── model_documentation.md            ← Complete SR 11-7 model development document
```
---

## Methodology

### 1. Exploratory Data Analysis (`01_EDA_and_Cleaning.ipynb`)
Comprehensive three-phase EDA covering data quality assessment, univariate distribution analysis and bivariate relationship analysis with the default indicator. Key findings include sentinel values in delinquency count variables (96, 98 as legacy missing value encodings), a systematic data artifact in DebtRatio driven by zero-income division, extreme outliers in RevolvingUtilizationOfUnsecuredLines and MonthlyIncome, and 19.8% missing rate in MonthlyIncome requiring careful imputation strategy.

### 2. Feature Engineering (`02_Feature_Engineering.ipynb`)
Weight of Evidence transformation applied to all features using OptimalBinning fitted exclusively on the training set. Information Value used for variable selection with IV threshold of 0.02. Key engineering decisions include joint treatment of MonthlyIncome and DebtRatio through an Income_missing indicator and DebtRatio_valid variable — addressing the zero-income artifact contamination problem — and binary flag transformation of severely concentrated delinquency count variables. Final feature set of 11 variables selected from an initial candidate set of 13.

### 3. Model Development (`03_Model_Development.ipynb`)
Logistic regression with Lasso regularisation estimated by maximum likelihood using the SAGA solver. Class imbalance addressed through class_weight='balanced' — chosen over SMOTE and undersampling because it preserves the true population default rate distribution, requiring no post-hoc probability recalibration for IFRS 9 use. Regularisation parameter C selected through 5-fold stratified cross-validation grid search across six candidate values. XGBoost trained as challenger model with Platt scaling probability calibration.

### 4. Model Evaluation (`04_Model_Evaluation.ipynb`)
Discrimination evaluated through AUC-ROC, Gini coefficient and KS statistic on the held-out test set. Calibration evaluated through log-loss and calibration curves. Champion-challenger comparison formalised through the DeLong test for statistical significance of AUC differences. Feature importance compared between logistic regression coefficients and XGBoost gain importance to assess variable consistency across methodologies.

### 5. IFRS 9 ECL Calculation (`05_IFRS9_ECL.ipynb`)
Three-stage Expected Credit Loss model applied to the portfolio using model PD outputs. SICR operationalised through three complementary indicators. Lifetime ECL approximated using a constant hazard rate term structure for Stage 2 and Stage 3 exposures. Provisioning impact demonstrated on a simulated portfolio P&L and balance sheet.

### 6. Basel III IRB Capital (`06_Basel_IRB_Capital.ipynb`)
Asymptotic Single Risk Factor model implemented to calculate loan-level IRB capital requirements. Asset correlation computed using the Basel III retail formula. Capital requirement K derived from the stressed conditional PD at the 99.9th percentile of the systematic risk factor distribution. Portfolio-level IRB RWA compared against SA RWA baseline, demonstrating the capital efficiency of risk-sensitive internal modelling.

### 7. SHAP Explainability (`07_SHAP_Explainability.ipynb`)
TreeExplainer applied to the XGBoost model to generate SHAP values for the test set. Global feature importance visualised through summary plot and mean absolute SHAP values. Individual prediction explanations demonstrated through force plots for representative defaulter and non-defaulter profiles. SHAP dependence plots generated for top features to visualise interaction effects — supporting the individual explanation requirements of IFRS 9 model governance.

---

## Dataset

This project uses the **Give Me Some Credit** dataset from Kaggle — a benchmark consumer lending dataset containing 150,000 observations representative of the US consumer market circa 2007-2008.

**Download instructions:**

1. Create a free account at [kaggle.com](https://www.kaggle.com)
2. Navigate to the [Give Me Some Credit competition](https://www.kaggle.com/c/GiveMeSomeCredit)
3. Download `cs-training.csv`
4. Place the file in the `data/` directory

The dataset cannot be distributed with this repository due to Kaggle's terms of service. See `data/README.md` for the complete variable dictionary and descriptive statistics.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/credit-risk-pd-model.git
cd credit-risk-pd-model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (see Dataset section above)
# Place cs-training.csv in data/ directory

# Launch notebooks
jupyter notebook notebooks/
```

---

## Requirements

```bash
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=1.7.0
optbinning>=0.19.0
shap>=0.42.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
imbalanced-learn>=0.10.0
jupyter>=1.0.0
```

---

## Technical Stack

| Tool | Purpose |
|---|---|
| Python 3.11+ | Core implementation language |
| scikit-learn | Logistic regression, cross-validation, evaluation metrics |
| XGBoost | Gradient boosted tree challenger model |
| OptimalBinning | WoE transformation and IV calculation |
| SHAP | Model explainability and feature attribution |
| SciPy | Statistical tests including DeLong AUC comparison |
| pandas / numpy | Data manipulation and numerical computation |
| matplotlib / seaborn | Visualisation |

---

## Project Background

This project was developed as part of preparation for quantitative risk roles in the financial services industry, with the goal of demonstrating credit risk modelling capability at a level consistent with Analyst and Consultant positions in regulatory and financial risk advisory practices.

The modelling decisions, regulatory framework implementations and documentation structure reflect the standards applied in professional credit risk model development — including the SR 11-7 model development document format used by major financial institutions and their external validators.

---

## Author

**Enrique Ávila Muñoz**
Economics and Finance — University of London (LSE)
Data Analysis Certifications — DataCamp and Google

[LinkedIn](https://linkedin.com/in/enriqueavilam) | [GitHub](https://github.com/eavila-economist)

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgements

Dataset: [Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)

Regulatory references: Basel Committee on Banking Supervision, IASB IFRS 9, Federal Reserve SR 11-7, OSFI E-23