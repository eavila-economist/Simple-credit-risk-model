"""
woe_iv.py
=========
Weight of Evidence (WoE) and Information Value (IV) calculation
and transformation for the Credit Risk PD Model.

Mathematical foundation:
    WoE_b = ln(Distr_Good_b / Distr_Bad_b)

    Where:
        Distr_Good_b = n_good_b / N_good  (proportion of non-defaulters in bin b)
        Distr_Bad_b  = n_bad_b  / N_bad   (proportion of defaulters in bin b)

    IV = Σ_b (Distr_Good_b - Distr_Bad_b) × WoE_b

    IV is formally equivalent to the symmetric Kullback-Leibler divergence
    between the good and bad distributions:
        IV = KL(Good || Bad) + KL(Bad || Good)

IV thresholds (empirical rules from credit scoring practice):
    IV < 0.02   : Useless — drop variable
    0.02-0.10   : Weak — keep with caution
    0.10-0.30   : Medium — good predictor
    0.30-0.50   : Strong — very useful
    IV > 0.50   : Very Strong — investigate for data leakage

Regulatory relevance:
    SR 11-7 requires that variable selection decisions be documented and
    economically justified. IV provides a rigorous, information-theoretic
    basis for feature selection that is reproducible and auditable.

    WoE transformation pre-linearizes non-linear relationships, satisfying
    the linearity assumption of logistic regression by construction rather
    than approximation — a key conceptual soundness requirement under SR 11-7.
"""

import numpy as np
import pandas as pd
from typing import Optional
import warnings

try:
    from optbinning import OptimalBinning
    OPTBINNING_AVAILABLE = True
except ImportError:
    OPTBINNING_AVAILABLE = False
    warnings.warn(
        "optbinning not available. Manual WoE calculation will be used. "
        "Install with: pip install optbinning"
    )


# ---------------------------------------------------------------------------
# IV threshold constants
# ---------------------------------------------------------------------------

IV_USELESS = 0.02
IV_WEAK = 0.10
IV_MEDIUM = 0.30
IV_STRONG = 0.50


def classify_iv(iv: float) -> str:
    """
    Classify a variable's predictive power based on its IV value.

    Parameters
    ----------
    iv : float
        Information Value.

    Returns
    -------
    str
        Predictive power classification.
    """
    if iv is None or np.isnan(iv):
        return "Error — check variable"
    elif iv < IV_USELESS:
        return "Useless — drop"
    elif iv < IV_WEAK:
        return "Weak"
    elif iv < IV_MEDIUM:
        return "Medium"
    elif iv < IV_STRONG:
        return "Strong"
    else:
        return "Very Strong — check for leakage"


# ---------------------------------------------------------------------------
# Manual WoE and IV calculation
# ---------------------------------------------------------------------------

def calculate_woe_iv_manual(
    x: np.ndarray,
    y: np.ndarray,
    feature_name: str = "feature",
    smoothing: float = 0.5
) -> tuple:
    """
    Calculate WoE and IV manually for a feature.

    Used as fallback when OptimalBinning is unavailable or fails (e.g.,
    for binary variables with only two unique values where OptimalBinning
    cannot create meaningful bin boundaries).

    For binary variables (0/1) each unique value becomes its own bin.
    For continuous variables with few unique values each value is a bin.

    Smoothing (Laplace smoothing) is applied to prevent infinite WoE
    when a bin contains only one class:
        Distr_Good_b = (n_good_b + smoothing) / (N_good + smoothing × B)

    Parameters
    ----------
    x : np.ndarray
        Feature values (1D array).
    y : np.ndarray
        Binary target (1 = default, 0 = non-default).
    feature_name : str
        Feature name for reporting.
    smoothing : float
        Laplace smoothing constant. Default: 0.5.

    Returns
    -------
    tuple : (pd.DataFrame, float, dict)
        - WoE table with bin-level statistics
        - Total IV for the feature
        - Dictionary mapping bin value to WoE for transformation
    """
    df = pd.DataFrame({'x': x, 'y': y})

    total_good = (df['y'] == 0).sum()
    total_bad = (df['y'] == 1).sum()

    if total_good == 0 or total_bad == 0:
        warnings.warn(f"[{feature_name}] No goods or bads — cannot compute WoE")
        return pd.DataFrame(), 0.0, {}

    # Aggregate by unique value
    grouped = df.groupby('x')['y'].agg(
        total='count',
        bad='sum'
    ).reset_index()
    grouped.columns = ['bin', 'total', 'bad']
    grouped['good'] = grouped['total'] - grouped['bad']

    n_bins = len(grouped)

    # Laplace smoothing applied to counts before distribution calculation
    grouped['dist_good'] = (grouped['good'] + smoothing) / (total_good + smoothing * n_bins)
    grouped['dist_bad'] = (grouped['bad'] + smoothing) / (total_bad + smoothing * n_bins)

    # WoE and IV
    grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
    grouped['iv_bin'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']

    total_iv = grouped['iv_bin'].sum()

    # Default rate per bin
    grouped['default_rate'] = grouped['bad'] / grouped['total']

    # WoE mapping dictionary for transformation
    woe_map = dict(zip(grouped['bin'], grouped['woe']))

    grouped['feature'] = feature_name

    return grouped, total_iv, woe_map


# ---------------------------------------------------------------------------
# OptimalBinning-based WoE and IV calculation
# ---------------------------------------------------------------------------

def calculate_woe_iv_optbinning(
    x: np.ndarray,
    y: np.ndarray,
    feature_name: str = "feature",
    max_n_bins: int = 10,
    min_bin_size: float = 0.05
) -> tuple:
    """
    Calculate WoE and IV using OptimalBinning.

    OptimalBinning solves a constrained optimisation problem to find bin
    boundaries that maximise IV subject to:
    - Minimum bin size of min_bin_size (prevents sparse bins with unreliable WoE)
    - Monotonicity of WoE across bins for continuous variables
    - Maximum of max_n_bins bins per variable

    Falls back to manual calculation if OptimalBinning fails (e.g., for
    binary variables with only two unique values).

    Parameters
    ----------
    x : np.ndarray
        Feature values.
    y : np.ndarray
        Binary target.
    feature_name : str
        Feature name.
    max_n_bins : int
        Maximum number of bins. Default: 10.
    min_bin_size : float
        Minimum fraction of observations per bin. Default: 0.05.

    Returns
    -------
    tuple : (pd.DataFrame, float, OptimalBinning or None)
        - IV summary row as DataFrame
        - Total IV value
        - Fitted OptimalBinning object (for transformation) or None
    """
    if not OPTBINNING_AVAILABLE:
        raise ImportError("optbinning not installed")

    try:
        optb = OptimalBinning(
            name=feature_name,
            dtype='numerical',
            max_n_bins=max_n_bins,
            min_bin_size=min_bin_size
        )

        optb.fit(x, y.astype(int))
        binning_table = optb.binning_table.build()
        iv = binning_table['IV'].iloc[-1]

        result = pd.DataFrame([{
            'feature': feature_name,
            'iv': round(float(iv), 4),
            'predictive_power': classify_iv(float(iv))
        }])

        return result, float(iv), optb

    except Exception as e:
        warnings.warn(f"[{feature_name}] OptimalBinning failed: {e}. Using manual calculation.")
        return None, None, None


# ---------------------------------------------------------------------------
# IV calculation for all features
# ---------------------------------------------------------------------------

def calculate_all_iv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_n_bins: int = 10,
    min_bin_size: float = 0.05,
    smoothing: float = 0.5
) -> tuple:
    """
    Calculate IV for all features and return ranked summary with fitted binners.

    Uses OptimalBinning where available. Falls back to manual calculation
    for binary variables or when OptimalBinning fails.

    All calculations use training data only. The fitted binners (OptimalBinning
    objects or WoE mapping dictionaries) are stored for subsequent transformation
    of both training and test sets.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training set features.
    y_train : pd.Series
        Training set target (1=default, 0=non-default).
    max_n_bins : int
        Maximum bins for OptimalBinning. Default: 10.
    min_bin_size : float
        Minimum bin size fraction. Default: 0.05.
    smoothing : float
        Laplace smoothing for manual calculation. Default: 0.5.

    Returns
    -------
    tuple : (pd.DataFrame, dict)
        - IV summary DataFrame ranked by IV descending
        - Dictionary mapping feature names to fitted binner objects
          (OptimalBinning instance or WoE map dict)
    """
    print("=" * 60)
    print("CALCULATING WoE AND IV — TRAINING SET ONLY")
    print("=" * 60)

    iv_results = []
    fitted_binners = {}

    y_vals = y_train.values.astype(int)

    for feature in X_train.columns:
        x_vals = X_train[feature].values
        n_unique = X_train[feature].nunique()

        # Try OptimalBinning first for non-binary features
        if OPTBINNING_AVAILABLE and n_unique > 2:
            result_df, iv, optb = calculate_woe_iv_optbinning(
                x_vals, y_vals, feature, max_n_bins, min_bin_size
            )
            if iv is not None and not np.isnan(iv):
                iv_results.append({
                    'feature': feature,
                    'iv': round(iv, 4),
                    'n_bins': n_unique if n_unique <= max_n_bins else max_n_bins,
                    'method': 'OptimalBinning',
                    'predictive_power': classify_iv(iv)
                })
                fitted_binners[feature] = {'type': 'optbinning', 'binner': optb}
                print(f"  [{feature}] IV={iv:.4f} | {classify_iv(iv)} | OptimalBinning")
                continue

        # Manual calculation for binary variables or OptimalBinning failures
        _, iv, woe_map = calculate_woe_iv_manual(
            x_vals, y_vals, feature, smoothing
        )

        if iv is not None and not np.isnan(iv):
            iv_results.append({
                'feature': feature,
                'iv': round(iv, 4),
                'n_bins': n_unique,
                'method': 'Manual',
                'predictive_power': classify_iv(iv)
            })
            fitted_binners[feature] = {'type': 'manual', 'woe_map': woe_map}
            print(f"  [{feature}] IV={iv:.4f} | {classify_iv(iv)} | Manual")

    iv_df = pd.DataFrame(iv_results).sort_values('iv', ascending=False).reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print("IV RANKING COMPLETE")
    print(f"{'=' * 60}")
    print(iv_df[['feature', 'iv', 'predictive_power']].to_string(index=False))

    return iv_df, fitted_binners


# ---------------------------------------------------------------------------
# WoE transformation
# ---------------------------------------------------------------------------

def transform_woe(
    X: pd.DataFrame,
    fitted_binners: dict,
    features_to_transform: Optional[list] = None
) -> pd.DataFrame:
    """
    Transform features to their WoE values using fitted binners.

    The WoE mapping learned from the training set is applied to all sets.
    The test set receives the same bin boundaries and WoE values as the
    training set — it does not receive its own WoE estimates.

    Unknown values (observations falling outside bins seen during training)
    receive WoE of 0.0 — the neutral value indicating no discriminatory
    information, a conservative imputation for out-of-sample observations.

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame to transform.
    fitted_binners : dict
        Dictionary of fitted binner objects from calculate_all_iv().
    features_to_transform : list, optional
        Subset of features to transform. If None, transforms all features
        present in fitted_binners.

    Returns
    -------
    pd.DataFrame
        WoE-transformed feature DataFrame.
    """
    X_woe = X.copy()

    if features_to_transform is None:
        features_to_transform = list(fitted_binners.keys())

    for feature in features_to_transform:
        if feature not in X.columns:
            warnings.warn(f"[{feature}] Not found in DataFrame — skipping")
            continue

        if feature not in fitted_binners:
            warnings.warn(f"[{feature}] No fitted binner found — skipping")
            continue

        binner_info = fitted_binners[feature]

        if binner_info['type'] == 'optbinning':
            optb = binner_info['binner']
            try:
                X_woe[feature] = optb.transform(X[feature].values, metric='woe')
            except Exception as e:
                warnings.warn(f"[{feature}] OptimalBinning transform failed: {e}")
                X_woe[feature] = 0.0

        elif binner_info['type'] == 'manual':
            woe_map = binner_info['woe_map']
            # Map values to WoE — unknown values get 0.0 (neutral)
            X_woe[feature] = X[feature].map(woe_map).fillna(0.0)

    return X_woe


def apply_manual_woe_override(
    X_woe: pd.DataFrame,
    X_original: pd.DataFrame,
    feature: str,
    woe_map: dict
) -> pd.DataFrame:
    """
    Apply a manually specified WoE mapping to a feature.

    Used when OptimalBinning fails to produce valid IV estimates — particularly
    for binary indicator variables (Ever90DaysPastDue, Ever60_89DaysPastDue)
    where the binary nature prevents the algorithm from creating meaningful bins.

    WoE values are computed manually from the two-group comparison:
        WoE_0 = ln(Distr_Good_0 / Distr_Bad_0)
        WoE_1 = ln(Distr_Good_1 / Distr_Bad_1)

    Parameters
    ----------
    X_woe : pd.DataFrame
        WoE-transformed DataFrame to update.
    X_original : pd.DataFrame
        Original (pre-WoE) DataFrame for mapping.
    feature : str
        Feature name to override.
    woe_map : dict
        Manual WoE mapping {value: woe_value}.
        Example: {0: 0.271306, 1: -2.051463}

    Returns
    -------
    pd.DataFrame
        Updated WoE DataFrame with manual mapping applied.
    """
    X_woe = X_woe.copy()

    if feature in X_original.columns:
        X_woe[feature] = X_original[feature].map(woe_map).fillna(0.0)
        print(f"  [{feature}] Manual WoE override applied: {woe_map}")

    return X_woe


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features_by_iv(
    iv_df: pd.DataFrame,
    threshold: float = IV_USELESS
) -> list:
    """
    Select features whose IV exceeds the specified threshold.

    Parameters
    ----------
    iv_df : pd.DataFrame
        IV summary DataFrame from calculate_all_iv().
    threshold : float
        Minimum IV for feature retention. Default: 0.02.

    Returns
    -------
    list
        List of feature names to retain.
    """
    selected = iv_df[iv_df['iv'] >= threshold]['feature'].tolist()
    dropped = iv_df[iv_df['iv'] < threshold]['feature'].tolist()

    print(f"Features retained (IV >= {threshold}): {len(selected)}")
    print(f"Features dropped  (IV <  {threshold}): {len(dropped)}")
    if dropped:
        print(f"  Dropped: {dropped}")

    return selected


# ---------------------------------------------------------------------------
# Diagnostic utilities
# ---------------------------------------------------------------------------

def woe_stability_check(
    X_train_woe: pd.DataFrame,
    X_test_woe: pd.DataFrame,
    features: Optional[list] = None
) -> pd.DataFrame:
    """
    Check WoE value stability between training and test sets.

    Large differences in mean WoE between train and test indicate
    distributional shift — a precursor to PSI failure and model degradation.
    This check provides an early warning before formal PSI calculation.

    Parameters
    ----------
    X_train_woe : pd.DataFrame
        WoE-transformed training set.
    X_test_woe : pd.DataFrame
        WoE-transformed test set.
    features : list, optional
        Features to check. If None, checks all common columns.

    Returns
    -------
    pd.DataFrame
        Stability summary with mean WoE for train and test per feature.
    """
    if features is None:
        features = [c for c in X_train_woe.columns if c in X_test_woe.columns]

    rows = []
    for feature in features:
        train_mean = X_train_woe[feature].mean()
        test_mean = X_test_woe[feature].mean()
        diff = abs(train_mean - test_mean)
        rows.append({
            'feature': feature,
            'train_mean_woe': round(train_mean, 4),
            'test_mean_woe': round(test_mean, 4),
            'abs_difference': round(diff, 4),
            'stable': 'Yes' if diff < 0.1 else 'Review'
        })

    return pd.DataFrame(rows).sort_values('abs_difference', ascending=False)