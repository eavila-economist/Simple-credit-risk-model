"""
cleaning.py
===========
Data cleaning pipeline for the Credit Risk PD Model.

Implements all data quality remediation steps documented in Section 2.3
of the model development document, in the mandatory execution sequence.

Steps (must be executed in order):
    1. Replace sentinel values with NaN
    2. Treat RevolvingUtilizationOfUnsecuredLines outliers via Winsorization
    3. Joint treatment of MonthlyIncome and DebtRatio
    4. MonthlyIncome near-zero replacement and log transformation
    5. Age impossible value correction
    6. Missing value imputation with training-set medians
    7. Binary flag creation for concentrated delinquency variables

Key design principle:
    All cleaning parameters (medians, percentile caps) are computed on the
    training set only and stored in a CleaningParams dataclass. These stored
    parameters are then applied identically to the test set, preventing any
    form of data leakage from the test distribution into the cleaning process.

Regulatory relevance:
    SR 11-7 requires that all data transformations be documented and reproducible.
    The CleaningParams object serves as the audit trail for every cleaning decision,
    storing the exact parameter values applied during development.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings


# ---------------------------------------------------------------------------
# Parameters dataclass — stores all training-set derived cleaning parameters
# ---------------------------------------------------------------------------

@dataclass
class CleaningParams:
    """
    Stores all cleaning parameters derived from the training set.

    This object is fitted on X_train and applied to both X_train and X_test,
    ensuring that no information from the test set influences the cleaning
    process applied to either set.

    Attributes
    ----------
    cap_revolving : float
        99th percentile of RevolvingUtilizationOfUnsecuredLines on training set.
        Used to Winsorize impossible utilisation values above this threshold.

    cap_debt_ratio : float
        99th percentile of DebtRatio among observations with valid income on
        training set. Applied only to DebtRatio_valid after separating artifacts.

    income_threshold : float
        Minimum plausible monthly income in USD. Observations with income between
        0 and this threshold are treated as data entry errors and replaced with NaN.
        Default: 100.

    medians : dict
        Median value for each feature computed on the training set.
        Used for missing value imputation across all variables.

    sentinel_values : list
        Integer values encoding missing data in legacy systems.
        Detected empirically by identifying discontinuous jumps in value counts.
        Default: [96, 98].

    delinquency_cols : list
        Columns containing sentinel values requiring replacement.

    binary_flag_cols : list
        Delinquency count columns to be transformed to binary ever/never indicators.
        Selected based on extreme concentration at zero (>90% of observations).
    """

    cap_revolving: float = 0.0
    cap_debt_ratio: float = 0.0
    income_threshold: float = 100.0
    medians: dict = field(default_factory=dict)
    sentinel_values: list = field(default_factory=lambda: [96, 98])
    delinquency_cols: list = field(default_factory=lambda: [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse'
    ])
    binary_flag_cols: list = field(default_factory=lambda: [
        'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse'
    ])


# ---------------------------------------------------------------------------
# Step 1 — Sentinel value replacement
# ---------------------------------------------------------------------------

def replace_sentinel_values(
    X: pd.DataFrame,
    params: CleaningParams
) -> pd.DataFrame:
    """
    Replace sentinel values with NaN in delinquency count variables.

    Sentinel values (96, 98) are data encoding artifacts from legacy systems
    where unknown or unavailable delinquency counts were recorded as extreme
    integers rather than null values. They carry no economic information.

    This step must precede imputation. If imputation were performed first,
    the median calculation would be distorted by the artificially extreme
    sentinel values, producing imputed values that are themselves inflated.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame containing delinquency count variables.
    params : CleaningParams
        Cleaning parameters specifying sentinel values and target columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with sentinel values replaced by NaN.
    """
    X = X.copy()

    for col in params.delinquency_cols:
        if col in X.columns:
            n_before = X[col].isin(params.sentinel_values).sum()
            X[col] = X[col].replace(params.sentinel_values, np.nan)
            if n_before > 0:
                print(f"  [{col}] Replaced {n_before:,} sentinel values with NaN")

    return X


# ---------------------------------------------------------------------------
# Step 2 — Winsorization of RevolvingUtilizationOfUnsecuredLines
# ---------------------------------------------------------------------------

def fit_revolving_cap(X_train: pd.DataFrame, params: CleaningParams) -> CleaningParams:
    """
    Compute the 99th percentile cap for RevolvingUtilizationOfUnsecuredLines
    from the training set.

    The 99th percentile (≈1.09) is consistent with the theoretical upper bound
    of a utilisation ratio bounded between 0 and 1. Values above this threshold
    represent data integrity failures in the source system rather than genuine
    extreme utilisation.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training set feature DataFrame.
    params : CleaningParams
        Cleaning parameters object to update with computed cap.

    Returns
    -------
    CleaningParams
        Updated parameters with cap_revolving set.
    """
    col = 'RevolvingUtilizationOfUnsecuredLines'
    if col in X_train.columns:
        params.cap_revolving = np.nanpercentile(X_train[col], 99)
        print(f"  Revolving utilisation cap (p99): {params.cap_revolving:.4f}")
    return params


def apply_revolving_cap(X: pd.DataFrame, params: CleaningParams) -> pd.DataFrame:
    """
    Apply Winsorization to RevolvingUtilizationOfUnsecuredLines.

    Winsorization replaces values above the cap with the cap value, preserving
    the observation for all other variables while removing the impossible extreme.
    Row deletion is avoided to maintain the full observation set.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame.
    params : CleaningParams
        Cleaning parameters containing the fitted cap value.

    Returns
    -------
    pd.DataFrame
        DataFrame with utilisation values capped at the training p99.
    """
    X = X.copy()
    col = 'RevolvingUtilizationOfUnsecuredLines'
    if col in X.columns and params.cap_revolving > 0:
        n_capped = (X[col] > params.cap_revolving).sum()
        X[col] = X[col].clip(upper=params.cap_revolving)
        if n_capped > 0:
            print(f"  [{col}] Winsorized {n_capped:,} values above {params.cap_revolving:.4f}")
    return X


# ---------------------------------------------------------------------------
# Step 3 — Joint treatment of MonthlyIncome and DebtRatio
# ---------------------------------------------------------------------------

def create_income_validity_flag(
    X: pd.DataFrame,
    df_raw: pd.DataFrame,
    params: CleaningParams
) -> pd.DataFrame:
    """
    Create Income_missing binary indicator from raw income data.

    Income is classified as unreliable when:
    (a) MonthlyIncome is NaN in the raw dataset, or
    (b) MonthlyIncome is present but below the plausibility threshold ($100/month).

    This flag captures the risk associated with income uncertainty independently
    of the leverage signal in DebtRatio, separating two economically distinct
    phenomena that were previously conflated through the DebtRatio variable.

    The Income_missing flag has its own WoE value in the feature engineering
    step, allowing the model to assign appropriate weight to income uncertainty
    as a credit risk signal distinct from high leverage.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame (may already have MonthlyIncome cleaned).
    df_raw : pd.DataFrame
        Original raw DataFrame before any cleaning, used to access
        the original MonthlyIncome values for flag creation.
    params : CleaningParams
        Cleaning parameters containing income_threshold.

    Returns
    -------
    pd.DataFrame
        DataFrame with Income_missing column added.
    """
    X = X.copy()

    if 'MonthlyIncome' in df_raw.columns:
        raw_income = df_raw['MonthlyIncome']
        X['Income_missing'] = (
            raw_income.isna() |
            (raw_income < params.income_threshold)
        ).astype(int)

        n_missing = X['Income_missing'].sum()
        pct_missing = n_missing / len(X) * 100
        print(f"  Income_missing flag: {n_missing:,} observations ({pct_missing:.1f}%)")

    return X


def create_debt_ratio_valid(
    X: pd.DataFrame,
    params: CleaningParams
) -> pd.DataFrame:
    """
    Create DebtRatio_valid — genuine leverage ratio for borrowers with reliable income.

    Standard practice of capping DebtRatio at 1.0 or at the 99th percentile
    produces a material statistical distortion. Three populations are pooled
    at the cap value:

    (a) Genuine high-debt borrowers — elevated but real leverage, high default rate
    (b) Artifact borrowers — extreme ratio from zero income division, average default rate
    (c) Boundary borrowers — at exactly the cap value

    Pooling dilutes the WoE of the high-debt bin, compresses IV, and causes
    the logistic regression to assign an inflated coefficient to compensate
    for the reduced WoE range.

    This function preserves DebtRatio only where income is reliable, setting
    it to NaN where income is missing or implausible. The artifact population
    is captured separately through the Income_missing flag.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame containing DebtRatio and Income_missing.
    params : CleaningParams
        Cleaning parameters (income_threshold used for reference).

    Returns
    -------
    pd.DataFrame
        DataFrame with DebtRatio_valid added and original DebtRatio removed.
    """
    X = X.copy()

    if 'DebtRatio' in X.columns and 'Income_missing' in X.columns:
        # Preserve DebtRatio only where income is reliable
        X['DebtRatio_valid'] = np.where(
            X['Income_missing'] == 0,
            X['DebtRatio'],
            np.nan
        )

        n_artifact = X['Income_missing'].sum()
        print(f"  DebtRatio_valid: {n_artifact:,} observations set to NaN (unreliable income)")

        # Drop original DebtRatio — replaced by DebtRatio_valid and Income_missing
        X = X.drop(columns=['DebtRatio'])

    return X


def fit_debt_ratio_cap(X_train: pd.DataFrame, params: CleaningParams) -> CleaningParams:
    """
    Compute the 99th percentile cap for DebtRatio_valid from training observations
    with reliable income only.

    The cap is computed exclusively on valid income observations to avoid
    contaminating the cap with artifact-driven extreme values. This ensures
    the Winsorization addresses genuine data entry errors in the valid population
    rather than mixing them with zero-income artifacts.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training set with DebtRatio_valid and Income_missing already created.
    params : CleaningParams
        Cleaning parameters to update.

    Returns
    -------
    CleaningParams
        Updated parameters with cap_debt_ratio set.
    """
    if 'DebtRatio_valid' in X_train.columns and 'Income_missing' in X_train.columns:
        valid_mask = X_train['Income_missing'] == 0
        valid_ratios = X_train.loc[valid_mask, 'DebtRatio_valid']
        params.cap_debt_ratio = np.nanpercentile(valid_ratios, 99)
        print(f"  DebtRatio_valid cap (p99 of valid observations): {params.cap_debt_ratio:.4f}")
    return params


def apply_debt_ratio_cap(X: pd.DataFrame, params: CleaningParams) -> pd.DataFrame:
    """
    Apply Winsorization to DebtRatio_valid using training-derived cap.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame with DebtRatio_valid column.
    params : CleaningParams
        Cleaning parameters containing the fitted cap value.

    Returns
    -------
    pd.DataFrame
        DataFrame with DebtRatio_valid capped.
    """
    X = X.copy()
    col = 'DebtRatio_valid'
    if col in X.columns and params.cap_debt_ratio > 0:
        n_capped = (X[col] > params.cap_debt_ratio).sum()
        X[col] = X[col].clip(upper=params.cap_debt_ratio)
        if n_capped > 0:
            print(f"  [{col}] Winsorized {n_capped:,} values above {params.cap_debt_ratio:.4f}")
    return X


# ---------------------------------------------------------------------------
# Step 4 — MonthlyIncome treatment
# ---------------------------------------------------------------------------

def treat_monthly_income(
    X: pd.DataFrame,
    params: CleaningParams,
    is_train: bool = True,
    median_income: Optional[float] = None
) -> tuple:
    """
    Replace implausible near-zero income values, apply log transformation,
    and impute missing values.

    Order of operations within this step is critical:
    (1) Replace implausible values with NaN
    (2) Log transform — median of log-transformed values is more appropriate
        than log of median for a lognormal distribution
    (3) Impute NaN with median of log-transformed training values

    The log transformation serves three purposes:
    - Compresses the right tail of the income distribution
    - Converts the approximately lognormal distribution to approximately normal
    - Preserves zero values safely (log(1+0) = 0)

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame containing MonthlyIncome.
    params : CleaningParams
        Cleaning parameters containing income_threshold.
    is_train : bool
        If True, compute and return the log-median for use on test set.
        If False, use the provided median_income parameter.
    median_income : float, optional
        Pre-computed log-income median from training set.
        Required when is_train=False.

    Returns
    -------
    tuple : (pd.DataFrame, float)
        Cleaned DataFrame and the log-income median (for test set application).
    """
    X = X.copy()
    col = 'MonthlyIncome'

    if col not in X.columns:
        return X, median_income

    # Replace implausible near-zero values with NaN
    implausible_mask = (X[col] > 0) & (X[col] < params.income_threshold)
    n_implausible = implausible_mask.sum()
    X.loc[implausible_mask, col] = np.nan
    if n_implausible > 0:
        print(f"  [{col}] Replaced {n_implausible:,} implausible values (0 < income < {params.income_threshold})")

    # Log transform before imputation
    X[col] = np.log1p(X[col])

    # Compute or apply median
    if is_train:
        median_income = X[col].median()
        print(f"  [{col}] Log-income median (for imputation): {median_income:.4f}")
    elif median_income is None:
        raise ValueError("median_income must be provided when is_train=False")

    # Impute missing values
    n_missing = X[col].isna().sum()
    X[col] = X[col].fillna(median_income)
    if n_missing > 0:
        print(f"  [{col}] Imputed {n_missing:,} missing values with log-median {median_income:.4f}")

    return X, median_income


# ---------------------------------------------------------------------------
# Step 5 — Age impossible value correction
# ---------------------------------------------------------------------------

def correct_age(X: pd.DataFrame, params: CleaningParams) -> pd.DataFrame:
    """
    Replace the single impossible age=0 observation with the training median.

    Age=0 is impossible for any credit borrower. This observation is corrected
    using replace() rather than fillna() because the value is present but wrong —
    fillna() only operates on NaN values and would leave age=0 unchanged.

    A dedicated step is used rather than including age=0 in the general imputation
    step because the origin of this error — a data entry error rather than a missing
    value — warrants explicit documentation separate from the missing value treatment.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame containing age column.
    params : CleaningParams
        Cleaning parameters (medians dict must already contain age median).

    Returns
    -------
    pd.DataFrame
        DataFrame with age=0 replaced by training median.
    """
    X = X.copy()
    col = 'age'

    if col in X.columns and col in params.medians:
        n_impossible = (X[col] == 0).sum()
        if n_impossible > 0:
            X[col] = X[col].replace(0, params.medians[col])
            print(f"  [{col}] Replaced {n_impossible} impossible zero(s) with median {params.medians[col]:.1f}")

    return X


# ---------------------------------------------------------------------------
# Step 6 — Missing value imputation
# ---------------------------------------------------------------------------

def fit_medians(X_train: pd.DataFrame, params: CleaningParams) -> CleaningParams:
    """
    Compute median for each feature from the training set.

    Median imputation is chosen over mean imputation because all features
    exhibit right-skewed distributions. The arithmetic mean is pulled toward
    extreme values by the heavy tails present in credit data, producing
    imputed values that overestimate the typical borrower's risk characteristics.
    The median is the 50th percentile of the ordered distribution and is
    unaffected by extreme values.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training set feature DataFrame. Sentinel values must already
        be replaced with NaN before computing medians.
    params : CleaningParams
        Cleaning parameters object to update with computed medians.

    Returns
    -------
    CleaningParams
        Updated parameters with medians dict populated.
    """
    params.medians = {}
    for col in X_train.columns:
        params.medians[col] = X_train[col].median()
        print(f"  Median [{col}]: {params.medians[col]:.4f}")
    return params


def impute_missing(X: pd.DataFrame, params: CleaningParams) -> pd.DataFrame:
    """
    Impute all remaining missing values using training-set medians.

    This step handles:
    - Original NaN values in MonthlyIncome and NumberOfDependents
    - NaN values introduced by sentinel value replacement (Step 1)
    - NaN values introduced by DebtRatio_valid creation (Step 3)

    Training-set medians are applied to both training and test sets,
    ensuring no information from the test distribution influences imputation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame with all prior cleaning steps applied.
    params : CleaningParams
        Cleaning parameters containing fitted medians.

    Returns
    -------
    pd.DataFrame
        DataFrame with all NaN values replaced.
    """
    X = X.copy()

    for col in X.columns:
        n_missing = X[col].isna().sum()
        if n_missing > 0 and col in params.medians:
            X[col] = X[col].fillna(params.medians[col])
            print(f"  [{col}] Imputed {n_missing:,} missing values")

    # Verify no missing values remain
    remaining = X.isna().sum().sum()
    if remaining > 0:
        warnings.warn(f"  WARNING: {remaining} missing values remain after imputation")
    else:
        print(f"  All missing values successfully imputed")

    return X


# ---------------------------------------------------------------------------
# Step 7 — Binary flag creation for delinquency variables
# ---------------------------------------------------------------------------

def create_delinquency_flags(X: pd.DataFrame, params: CleaningParams) -> pd.DataFrame:
    """
    Transform severely concentrated delinquency count variables to binary flags.

    Bivariate analysis revealed that NumberOfTimes90DaysLate and
    NumberOfTime60-89DaysPastDueNotWorse each have over 95% of observations
    at zero. The default rate shows a dramatic binary structure:
    - Borrowers with zero serious delinquencies: ~7.5% default rate
    - Borrowers with one or more serious delinquencies: 35-50% default rate

    The count beyond one event adds negligible additional discrimination while
    introducing sparsity that destabilises WoE bin estimation. The ever/never
    binary transformation preserves essentially all predictive information while
    producing stable WoE estimates.

    New columns created:
    - Ever90DaysPastDue: 1 if NumberOfTimes90DaysLate >= 1, else 0
    - Ever60_89DaysPastDue: 1 if NumberOfTime60-89DaysPastDueNotWorse >= 1, else 0

    Original columns are dropped after flag creation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame with delinquency count variables.
    params : CleaningParams
        Cleaning parameters specifying which columns to transform.

    Returns
    -------
    pd.DataFrame
        DataFrame with binary flags added and original counts removed.
    """
    X = X.copy()

    flag_mapping = {
        'NumberOfTimes90DaysLate': 'Ever90DaysPastDue',
        'NumberOfTime60-89DaysPastDueNotWorse': 'Ever60_89DaysPastDue'
    }

    cols_to_drop = []

    for original_col, flag_col in flag_mapping.items():
        if original_col in X.columns:
            X[flag_col] = (X[original_col] >= 1).astype(int)
            n_positive = X[flag_col].sum()
            pct_positive = n_positive / len(X) * 100
            print(f"  [{flag_col}] Created: {n_positive:,} positive ({pct_positive:.1f}%)")
            cols_to_drop.append(original_col)

    X = X.drop(columns=cols_to_drop)
    print(f"  Dropped original columns: {cols_to_drop}")

    return X


# ---------------------------------------------------------------------------
# Master pipeline functions
# ---------------------------------------------------------------------------

def fit_cleaning_params(
    X_train: pd.DataFrame,
    df_raw: pd.DataFrame,
    income_threshold: float = 100.0
) -> CleaningParams:
    """
    Fit all cleaning parameters on the training set.

    This function computes all parameters required for the cleaning pipeline
    from the training data only. The returned CleaningParams object is then
    passed to clean_dataset() for application to both training and test sets.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training set feature DataFrame — before any cleaning.
    df_raw : pd.DataFrame
        Full raw dataset — used to access original MonthlyIncome
        values for Income_missing flag creation.
    income_threshold : float
        Minimum plausible monthly income in USD. Default: 100.

    Returns
    -------
    CleaningParams
        Fitted cleaning parameters ready for application to train and test sets.
    """
    print("=" * 60)
    print("FITTING CLEANING PARAMETERS ON TRAINING SET")
    print("=" * 60)

    params = CleaningParams(income_threshold=income_threshold)

    # Step 1 — Replace sentinels first (affects median computation)
    X_temp = replace_sentinel_values(X_train, params)

    # Step 2 — Fit revolving utilisation cap
    print("\nStep 2 — Revolving utilisation cap:")
    params = fit_revolving_cap(X_temp, params)

    # Step 3 — Create Income_missing and DebtRatio_valid to fit debt ratio cap
    print("\nStep 3 — Income and DebtRatio joint treatment:")
    X_temp = create_income_validity_flag(X_temp, df_raw.loc[X_train.index], params)
    X_temp = create_debt_ratio_valid(X_temp, params)
    params = fit_debt_ratio_cap(X_temp, params)

    # Step 6 — Fit medians (after sentinel replacement, before imputation)
    print("\nStep 6 — Computing training medians:")
    params = fit_medians(X_temp, params)

    print("\n" + "=" * 60)
    print("CLEANING PARAMETERS FITTED SUCCESSFULLY")
    print("=" * 60)

    return params


def clean_dataset(
    X: pd.DataFrame,
    df_raw: pd.DataFrame,
    params: CleaningParams,
    is_train: bool = True,
    median_income: Optional[float] = None
) -> tuple:
    """
    Apply the complete cleaning pipeline to a dataset.

    Executes all seven cleaning steps in mandatory sequence using
    parameters derived exclusively from the training set.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame to clean.
    df_raw : pd.DataFrame
        Original raw DataFrame (same index as X) for Income_missing creation.
    params : CleaningParams
        Fitted cleaning parameters from fit_cleaning_params().
    is_train : bool
        True for training set (computes log-income median).
        False for test set (applies provided median_income).
    median_income : float, optional
        Log-income median from training set. Required when is_train=False.

    Returns
    -------
    tuple : (pd.DataFrame, float)
        Cleaned DataFrame and log-income median value.

    Example
    -------
    >>> params = fit_cleaning_params(X_train, df_raw)
    >>> X_train_clean, log_median = clean_dataset(X_train, df_raw, params, is_train=True)
    >>> X_test_clean, _ = clean_dataset(X_test, df_raw, params, is_train=False, median_income=log_median)
    """
    set_label = "TRAINING SET" if is_train else "TEST SET"
    print(f"\n{'=' * 60}")
    print(f"CLEANING PIPELINE — {set_label}")
    print(f"{'=' * 60}")
    print(f"Input shape: {X.shape}")

    # Step 1 — Sentinel values
    print("\nStep 1 — Sentinel value replacement:")
    X = replace_sentinel_values(X, params)

    # Step 2 — Revolving utilisation
    print("\nStep 2 — Revolving utilisation Winsorization:")
    X = apply_revolving_cap(X, params)

    # Step 3 — Joint income and DebtRatio treatment
    print("\nStep 3 — Income and DebtRatio joint treatment:")
    X = create_income_validity_flag(X, df_raw.loc[X.index], params)
    X = create_debt_ratio_valid(X, params)
    X = apply_debt_ratio_cap(X, params)

    # Step 4 — MonthlyIncome treatment
    print("\nStep 4 — MonthlyIncome treatment:")
    X, median_income = treat_monthly_income(X, params, is_train, median_income)

    # Step 5 — Age correction
    print("\nStep 5 — Age impossible value correction:")
    X = correct_age(X, params)

    # Step 6 — Missing value imputation
    print("\nStep 6 — Missing value imputation:")
    X = impute_missing(X, params)

    # Step 7 — Binary delinquency flags
    print("\nStep 7 — Binary delinquency flag creation:")
    X = create_delinquency_flags(X, params)

    print(f"\nOutput shape: {X.shape}")
    print(f"Missing values remaining: {X.isna().sum().sum()}")
    print(f"{'=' * 60}")

    return X, median_income


def cleaning_summary(X_before: pd.DataFrame, X_after: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary table comparing the dataset before and after cleaning.

    Useful for documentation and audit trail purposes consistent with
    SR 11-7 requirements for transparent data transformation documentation.

    Parameters
    ----------
    X_before : pd.DataFrame
        Raw feature DataFrame before cleaning.
    X_after : pd.DataFrame
        Cleaned feature DataFrame after pipeline application.

    Returns
    -------
    pd.DataFrame
        Summary table with before/after statistics for each variable.
    """
    summary_rows = []

    for col in X_before.columns:
        row = {
            'variable': col,
            'missing_before': X_before[col].isna().sum(),
            'missing_after': X_after[col].isna().sum() if col in X_after.columns else 'dropped',
            'mean_before': X_before[col].mean() if X_before[col].dtype != object else 'N/A',
            'mean_after': X_after[col].mean() if col in X_after.columns and X_after[col].dtype != object else 'N/A',
            'max_before': X_before[col].max() if X_before[col].dtype != object else 'N/A',
            'max_after': X_after[col].max() if col in X_after.columns and X_after[col].dtype != object else 'N/A',
            'status': 'kept' if col in X_after.columns else 'dropped/replaced'
        }
        summary_rows.append(row)

    # Add new columns created during cleaning
    new_cols = [c for c in X_after.columns if c not in X_before.columns]
    for col in new_cols:
        row = {
            'variable': col,
            'missing_before': 'N/A (new)',
            'missing_after': X_after[col].isna().sum(),
            'mean_before': 'N/A',
            'mean_after': X_after[col].mean(),
            'max_before': 'N/A',
            'max_after': X_after[col].max(),
            'status': 'created'
        }
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)