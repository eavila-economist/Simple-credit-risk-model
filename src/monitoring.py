"""
monitoring.py
=============
Post-deployment model performance monitoring framework for the Credit Risk PD Model.

Implements the ongoing monitoring requirements of SR 11-7, which mandates
continuous tracking of model performance after deployment to detect when
models need recalibration, redevelopment, or retirement.

Monitoring dimensions:
    Population stability (PSI/CSI): Has the borrower population shifted
        since model development? Are inputs still distributed as expected?

    Discrimination stability (Gini/AUC): Is the model still ranking
        borrowers correctly over time?

    Calibration stability: Are predicted PDs still aligned with
        observed default rates?

    Backtesting: Do predicted PDs from T-12 months match observed
        defaults over the following 12 months?

Traffic light framework (SR 11-7 monitoring governance):
    GREEN  : No action required — model performing as expected
    AMBER  : Investigate — potential deterioration detected
    RED    : Action required — recalibration or rebuild triggered

Regulatory relevance:
    SR 11-7 requires that models be monitored on an ongoing basis with
    results reported to the Model Risk Committee. The frequency of monitoring
    and the thresholds for escalation must be specified in the model's
    governance documentation and consistently applied.

    IFRS 9 also requires that provisioning models be reviewed for continued
    appropriateness at each reporting date. Significant population shift (PSI)
    or discrimination degradation are indicators that the model may no longer
    produce appropriate ECL estimates.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from typing import Optional
import matplotlib.pyplot as plt
import warnings


# ---------------------------------------------------------------------------
# Traffic light thresholds
# ---------------------------------------------------------------------------

# Population Stability Index
PSI_GREEN = 0.10
PSI_AMBER = 0.25

# Gini coefficient — degradation from development value
GINI_DEGRADATION_AMBER = 0.10   # 10% relative degradation
GINI_DEGRADATION_RED = 0.20     # 20% relative degradation

# Calibration error — mean predicted PD vs observed default rate
CALIB_ERROR_AMBER = 0.20        # 20% relative error
CALIB_ERROR_RED = 0.40          # 40% relative error

# Backtesting — Kupiec test significance level
KUPIEC_ALPHA = 0.05


def traffic_light(value: float, amber_threshold: float, red_threshold: float,
                  higher_is_worse: bool = True) -> str:
    """
    Assign traffic light status based on thresholds.

    Parameters
    ----------
    value : float
        Metric value to classify.
    amber_threshold : float
        Threshold above (or below) which status is AMBER.
    red_threshold : float
        Threshold above (or below) which status is RED.
    higher_is_worse : bool
        If True, higher values are worse (PSI, calibration error).
        If False, lower values are worse (Gini degradation inverted).

    Returns
    -------
    str
        Traffic light status: GREEN, AMBER, or RED.
    """
    if higher_is_worse:
        if value >= red_threshold:
            return "RED"
        elif value >= amber_threshold:
            return "AMBER"
        else:
            return "GREEN"
    else:
        if value <= red_threshold:
            return "RED"
        elif value <= amber_threshold:
            return "AMBER"
        else:
            return "GREEN"


# ---------------------------------------------------------------------------
# Population Stability Index (PSI)
# ---------------------------------------------------------------------------

def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    label: str = "Score",
    plot: bool = False
) -> float:
    """
    Calculate Population Stability Index between development and current distributions.

    PSI measures the divergence between two distributions using a symmetric
    information-theoretic measure:

        PSI = Σ_b (Actual% - Expected%) × ln(Actual% / Expected%)

    This is equivalent to the symmetric KL divergence between the two
    distributions — the same mathematical foundation as IV.

    PSI thresholds (industry standard):
        PSI < 0.10  : GREEN — Stable, no action required
        PSI 0.10-0.25: AMBER — Moderate shift, investigate root cause
        PSI > 0.25  : RED — Significant shift, recalibrate or rebuild

    Parameters
    ----------
    expected : np.ndarray
        Reference distribution (development/training sample scores).
    actual : np.ndarray
        Current distribution (current scoring population scores).
    n_bins : int
        Number of bins for distribution comparison. Default: 10.
    label : str
        Variable label for reporting. Default: 'Score'.
    plot : bool
        Whether to plot distribution comparison. Default: False.

    Returns
    -------
    float
        PSI value.
    """
    # Create bins based on expected distribution percentiles
    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Count observations in each bin
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert to proportions — add small constant to avoid log(0)
    # Laplace smoothing: add 0.0001 to each count before normalising
    smoothing = 0.0001
    expected_pct = (expected_counts + smoothing) / (len(expected) + smoothing * n_bins)
    actual_pct = (actual_counts + smoothing) / (len(actual) + smoothing * n_bins)

    # PSI formula
    psi_bins = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = psi_bins.sum()

    status = traffic_light(psi, PSI_GREEN, PSI_AMBER)

    print(f"PSI [{label}]: {psi:.4f} | {status}")

    if plot:
        _plot_psi(expected_pct, actual_pct, psi_bins, psi, label)

    return psi


def calculate_csi(
    X_dev: pd.DataFrame,
    X_current: pd.DataFrame,
    features: Optional[list] = None,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Calculate Characteristic Stability Index for each input feature.

    CSI applies PSI to each individual input variable, identifying which
    specific characteristics are driving overall score distribution shift.
    When PSI flags instability, CSI provides the diagnostic to determine
    whether the shift is driven by behavioural changes (delinquency variables),
    economic changes (income, leverage), or demographic changes (age, dependents).

    Parameters
    ----------
    X_dev : pd.DataFrame
        Development sample feature values (training set).
    X_current : pd.DataFrame
        Current scoring population feature values.
    features : list, optional
        Features to analyse. If None, uses all common columns.
    n_bins : int
        Number of bins per variable. Default: 10.

    Returns
    -------
    pd.DataFrame
        CSI results sorted by instability (highest PSI first).
    """
    if features is None:
        features = [c for c in X_dev.columns if c in X_current.columns]

    results = []
    for feature in features:
        psi_val = calculate_psi(
            X_dev[feature].values,
            X_current[feature].values,
            n_bins=n_bins,
            label=feature,
            plot=False
        )
        status = traffic_light(psi_val, PSI_GREEN, PSI_AMBER)
        results.append({
            'feature': feature,
            'csi': round(psi_val, 4),
            'status': status
        })

    csi_df = pd.DataFrame(results).sort_values('csi', ascending=False).reset_index(drop=True)

    print(f"\nCSI Summary ({len(features)} features):")
    print(f"  GREEN  (CSI < {PSI_GREEN}): {(csi_df['status']=='GREEN').sum()}")
    print(f"  AMBER  (CSI {PSI_GREEN}-{PSI_AMBER}): {(csi_df['status']=='AMBER').sum()}")
    print(f"  RED    (CSI > {PSI_AMBER}): {(csi_df['status']=='RED').sum()}")

    return csi_df


# ---------------------------------------------------------------------------
# Discrimination monitoring
# ---------------------------------------------------------------------------

def monitor_discrimination(
    gini_development: float,
    gini_current: float,
    period_label: str = "Current period"
) -> dict:
    """
    Monitor Gini coefficient stability against the development benchmark.

    Gini degradation is measured as the relative decline from the development
    value. A 10% relative decline (e.g., Gini from 0.70 to 0.63) triggers
    AMBER status and investigation. A 20% decline triggers RED status and
    mandatory model review.

    Parameters
    ----------
    gini_development : float
        Gini coefficient achieved at model development on test set.
    gini_current : float
        Gini coefficient in the current monitoring period.
    period_label : str
        Label for the current period for reporting.

    Returns
    -------
    dict
        Discrimination monitoring results.
    """
    degradation = (gini_development - gini_current) / gini_development
    status = traffic_light(
        degradation,
        GINI_DEGRADATION_AMBER,
        GINI_DEGRADATION_RED,
        higher_is_worse=True
    )

    results = {
        'period': period_label,
        'gini_development': gini_development,
        'gini_current': gini_current,
        'absolute_change': round(gini_current - gini_development, 4),
        'relative_degradation': round(degradation, 4),
        'status': status
    }

    print(f"Discrimination Monitoring — {period_label}:")
    print(f"  Gini development: {gini_development:.4f}")
    print(f"  Gini current:     {gini_current:.4f}")
    print(f"  Degradation:      {degradation:.1%}")
    print(f"  Status:           {status}")

    return results


# ---------------------------------------------------------------------------
# Calibration monitoring
# ---------------------------------------------------------------------------

def monitor_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pd_bands: Optional[list] = None,
    period_label: str = "Current period"
) -> pd.DataFrame:
    """
    Monitor model calibration by comparing predicted PD to observed default rates.

    Calibration is evaluated both at the portfolio level and within PD bands.
    Portfolio-level calibration checks whether the mean predicted PD approximates
    the overall default rate. Band-level calibration checks whether the model's
    probability estimates are accurate across the full range of predictions —
    a model that is well calibrated on average but systematically over or
    underestimates in specific segments is still a calibration failure.

    Parameters
    ----------
    y_true : np.ndarray
        Observed default indicators (1=default, 0=non-default).
    y_prob : np.ndarray
        Predicted PD estimates from model.
    pd_bands : list, optional
        PD band boundaries for band-level calibration.
        Default: [0, 0.05, 0.10, 0.20, 0.50, 1.0]
    period_label : str
        Current monitoring period label.

    Returns
    -------
    pd.DataFrame
        Band-level calibration results.
    """
    if pd_bands is None:
        pd_bands = [0, 0.05, 0.10, 0.20, 0.50, 1.0]

    # Portfolio-level
    mean_pd = y_prob.mean()
    obs_dr = y_true.mean()
    calib_error = abs(mean_pd - obs_dr) / obs_dr
    portfolio_status = traffic_light(calib_error, CALIB_ERROR_AMBER, CALIB_ERROR_RED)

    print(f"\nCalibration Monitoring — {period_label}:")
    print(f"  Portfolio level:")
    print(f"    Mean predicted PD: {mean_pd:.4f}")
    print(f"    Observed default rate: {obs_dr:.4f}")
    print(f"    Relative error: {calib_error:.1%}")
    print(f"    Status: {portfolio_status}")

    # Band-level calibration
    bands = pd.cut(y_prob, bins=pd_bands, include_lowest=True)
    band_results = []

    for band in bands.cat.categories:
        mask = bands == band
        if mask.sum() == 0:
            continue

        n = mask.sum()
        mean_pred = y_prob[mask].mean()
        obs_rate = y_true[mask].mean()
        rel_error = abs(mean_pred - obs_rate) / max(obs_rate, 1e-6)
        band_status = traffic_light(rel_error, CALIB_ERROR_AMBER, CALIB_ERROR_RED)

        band_results.append({
            'pd_band': str(band),
            'n_exposures': n,
            'mean_predicted_pd': round(mean_pred, 4),
            'observed_default_rate': round(obs_rate, 4),
            'relative_error': round(rel_error, 4),
            'status': band_status
        })

    band_df = pd.DataFrame(band_results)
    print(f"\n  Band-level calibration:")
    print(band_df[['pd_band', 'n_exposures', 'mean_predicted_pd',
                   'observed_default_rate', 'relative_error', 'status']].to_string(index=False))

    return band_df


# ---------------------------------------------------------------------------
# Backtesting — Kupiec test
# ---------------------------------------------------------------------------

def kupiec_test(
    n_observations: int,
    n_exceptions: int,
    pd_estimate: float,
    alpha: float = KUPIEC_ALPHA
) -> dict:
    """
    Kupiec proportion of failures test for PD backtesting.

    Tests H₀: True default rate = PD estimate
    Against H₁: True default rate ≠ PD estimate

    The Kupiec test uses a likelihood ratio statistic that follows a
    chi-squared distribution with 1 degree of freedom under H₀:

        LR = -2 × ln[(1-PD)^(n-x) × PD^x] +
              2 × ln[(1-x/n)^(n-x) × (x/n)^x]

    Where n = number of observations and x = number of exceptions (defaults).

    Backtesting compares PD estimates made 12 months ago against actual
    default experience over the subsequent 12 months. This is the definitive
    calibration test — but requires waiting for outcomes to materialise,
    introducing a 12-month lag in the monitoring cycle.

    Parameters
    ----------
    n_observations : int
        Total number of observations in the backtesting cohort.
    n_exceptions : int
        Number of actual defaults observed.
    pd_estimate : float
        Model's PD estimate for this cohort (used as H₀).
    alpha : float
        Significance level. Default: 0.05.

    Returns
    -------
    dict
        Kupiec test results including LR statistic, p-value, and decision.
    """
    x = n_exceptions
    n = n_observations
    p = pd_estimate

    # Observed default rate
    p_hat = x / n

    # Likelihood ratio statistic
    # Handle edge cases where p_hat = 0 or p_hat = 1
    if p_hat == 0:
        lr = -2 * (x * np.log(p) + (n - x) * np.log(1 - p))
    elif p_hat == 1:
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p))
    else:
        lr = -2 * (
            x * np.log(p) + (n - x) * np.log(1 - p) -
            x * np.log(p_hat) - (n - x) * np.log(1 - p_hat)
        )

    # Chi-squared p-value with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    reject_h0 = p_value < alpha

    status = "RED — Model miscalibrated" if reject_h0 else "GREEN — Model well calibrated"

    results = {
        'n_observations': n,
        'n_exceptions': x,
        'observed_rate': round(p_hat, 4),
        'predicted_pd': round(p, 4),
        'lr_statistic': round(lr, 4),
        'p_value': round(p_value, 4),
        'reject_h0': reject_h0,
        'status': status
    }

    print(f"\nKupiec Backtesting:")
    print(f"  Observations:    {n:,}")
    print(f"  Actual defaults: {x:,} ({p_hat:.2%})")
    print(f"  Predicted PD:    {p:.2%}")
    print(f"  LR statistic:    {lr:.4f}")
    print(f"  P-value:         {p_value:.4f}")
    print(f"  Status:          {status}")

    return results


# ---------------------------------------------------------------------------
# Full monitoring report
# ---------------------------------------------------------------------------

def generate_monitoring_report(
    score_dev: np.ndarray,
    score_current: np.ndarray,
    X_dev: pd.DataFrame,
    X_current: pd.DataFrame,
    y_true_current: np.ndarray,
    y_prob_current: np.ndarray,
    gini_development: float,
    period_label: str = "Current Period",
    n_bins_psi: int = 10
) -> dict:
    """
    Generate a comprehensive quarterly monitoring report.

    Produces all monitoring metrics required for Model Risk Committee
    reporting under SR 11-7, covering the four monitoring dimensions:
    population stability, discrimination, calibration, and a summary
    traffic light dashboard.

    This function is designed to be called quarterly in production,
    with results stored and trended over time to detect gradual
    degradation that might not trigger individual thresholds in any
    single period.

    Parameters
    ----------
    score_dev : np.ndarray
        Model scores from development sample (training set predictions).
    score_current : np.ndarray
        Model scores for current scoring population.
    X_dev : pd.DataFrame
        Development sample feature values.
    X_current : pd.DataFrame
        Current population feature values.
    y_true_current : np.ndarray
        Observed defaults in current period.
    y_prob_current : np.ndarray
        Predicted PDs for current period.
    gini_development : float
        Gini coefficient from model development test set.
    period_label : str
        Label for current monitoring period (e.g., 'Q1-2025').
    n_bins_psi : int
        Number of bins for PSI calculation. Default: 10.

    Returns
    -------
    dict
        Complete monitoring report with all metrics and traffic light statuses.
    """
    print(f"\n{'=' * 60}")
    print(f"QUARTERLY MONITORING REPORT — {period_label}")
    print(f"{'=' * 60}")

    report = {'period': period_label}

    # 1. Score PSI
    print(f"\n1. SCORE POPULATION STABILITY")
    print("-" * 40)
    score_psi = calculate_psi(score_dev, score_current, n_bins_psi, "Score")
    report['score_psi'] = score_psi
    report['score_psi_status'] = traffic_light(score_psi, PSI_GREEN, PSI_AMBER)

    # 2. Characteristic Stability Index
    print(f"\n2. CHARACTERISTIC STABILITY INDEX")
    print("-" * 40)
    csi_df = calculate_csi(X_dev, X_current, n_bins=n_bins_psi)
    report['csi_summary'] = csi_df
    report['max_csi'] = csi_df['csi'].max()
    report['max_csi_feature'] = csi_df.iloc[0]['feature']

    # 3. Discrimination
    print(f"\n3. DISCRIMINATION STABILITY")
    print("-" * 40)
    gini_current = 2 * roc_auc_score(y_true_current, y_prob_current) - 1
    disc_results = monitor_discrimination(gini_development, gini_current, period_label)
    report['gini_current'] = gini_current
    report['gini_degradation'] = disc_results['relative_degradation']
    report['gini_status'] = disc_results['status']

    # 4. Calibration
    print(f"\n4. CALIBRATION MONITORING")
    print("-" * 40)
    calib_df = monitor_calibration(y_true_current, y_prob_current, period_label=period_label)
    mean_pd = y_prob_current.mean()
    obs_dr = y_true_current.mean()
    calib_error = abs(mean_pd - obs_dr) / obs_dr
    report['calibration_error'] = calib_error
    report['calibration_status'] = traffic_light(calib_error, CALIB_ERROR_AMBER, CALIB_ERROR_RED)
    report['calibration_detail'] = calib_df

    # 5. Traffic light summary
    all_statuses = [
        report['score_psi_status'],
        report['gini_status'],
        report['calibration_status']
    ]

    if 'RED' in all_statuses:
        overall_status = 'RED'
    elif 'AMBER' in all_statuses:
        overall_status = 'AMBER'
    else:
        overall_status = 'GREEN'

    report['overall_status'] = overall_status

    _print_traffic_light_summary(report)

    return report


def _print_traffic_light_summary(report: dict) -> None:
    """Print the traffic light dashboard summary."""
    emoji = {'GREEN': '🟢', 'AMBER': '🟡', 'RED': '🔴'}

    print(f"\n{'=' * 60}")
    print(f"TRAFFIC LIGHT SUMMARY — {report['period']}")
    print(f"{'=' * 60}")
    print(f"  Score PSI:       {report['score_psi']:.4f}  |  {emoji.get(report['score_psi_status'], '')} {report['score_psi_status']}")
    print(f"  Gini degradation:{report['gini_degradation']:.1%}   |  {emoji.get(report['gini_status'], '')} {report['gini_status']}")
    print(f"  Calib error:     {report['calibration_error']:.1%}   |  {emoji.get(report['calibration_status'], '')} {report['calibration_status']}")
    print(f"\n  OVERALL STATUS:  {emoji.get(report['overall_status'], '')} {report['overall_status']}")

    actions = {
        'GREEN': 'No action required. Continue regular monitoring.',
        'AMBER': 'Investigate root cause. Prepare recalibration assessment. Report to MRC.',
        'RED': 'Immediate action required. Escalate to Model Risk Committee. Consider model suspension pending recalibration or rebuild.'
    }
    print(f"\n  Recommended action: {actions[report['overall_status']]}")


# ---------------------------------------------------------------------------
# Time series monitoring
# ---------------------------------------------------------------------------

def plot_monitoring_trend(
    periods: list,
    psi_values: list,
    gini_values: list,
    gini_development: float,
    calibration_errors: list
) -> None:
    """
    Plot monitoring metrics over time for trend analysis.

    Trending metrics over multiple periods is more informative than
    single-period snapshots. A gradual drift that does not trigger
    thresholds in any individual period may reveal a concerning trend
    when viewed across multiple quarters.

    Parameters
    ----------
    periods : list
        Period labels (e.g., ['Q1-2024', 'Q2-2024', 'Q3-2024', 'Q4-2024']).
    psi_values : list
        Score PSI values per period.
    gini_values : list
        Gini coefficient values per period.
    gini_development : float
        Development benchmark Gini.
    calibration_errors : list
        Relative calibration errors per period.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # PSI trend
    colors_psi = ['green' if p < PSI_GREEN else 'orange' if p < PSI_AMBER else 'red'
                  for p in psi_values]
    axes[0].bar(periods, psi_values, color=colors_psi, alpha=0.8)
    axes[0].axhline(y=PSI_GREEN, color='orange', linestyle='--', label=f'Amber ({PSI_GREEN})')
    axes[0].axhline(y=PSI_AMBER, color='red', linestyle='--', label=f'Red ({PSI_AMBER})')
    axes[0].set_title('Score PSI Over Time')
    axes[0].set_ylabel('PSI')
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis='x', rotation=45)

    # Gini trend
    axes[1].plot(periods, gini_values, 'o-', color='steelblue', linewidth=2, label='Current Gini')
    axes[1].axhline(y=gini_development, color='green', linestyle='--', label=f'Development ({gini_development:.3f})')
    axes[1].axhline(
        y=gini_development * (1 - GINI_DEGRADATION_AMBER),
        color='orange', linestyle='--',
        label=f'Amber ({GINI_DEGRADATION_AMBER:.0%} degradation)'
    )
    axes[1].axhline(
        y=gini_development * (1 - GINI_DEGRADATION_RED),
        color='red', linestyle='--',
        label=f'Red ({GINI_DEGRADATION_RED:.0%} degradation)'
    )
    axes[1].set_title('Gini Coefficient Over Time')
    axes[1].set_ylabel('Gini')
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis='x', rotation=45)

    # Calibration error trend
    colors_calib = ['green' if e < CALIB_ERROR_AMBER else 'orange' if e < CALIB_ERROR_RED else 'red'
                    for e in calibration_errors]
    axes[2].bar(periods, [e * 100 for e in calibration_errors], color=colors_calib, alpha=0.8)
    axes[2].axhline(y=CALIB_ERROR_AMBER * 100, color='orange', linestyle='--',
                    label=f'Amber ({CALIB_ERROR_AMBER:.0%})')
    axes[2].axhline(y=CALIB_ERROR_RED * 100, color='red', linestyle='--',
                    label=f'Red ({CALIB_ERROR_RED:.0%})')
    axes[2].set_title('Calibration Error Over Time')
    axes[2].set_ylabel('Relative Calibration Error (%)')
    axes[2].legend(fontsize=8)
    axes[2].tick_params(axis='x', rotation=45)

    plt.suptitle('SR 11-7 Ongoing Monitoring Dashboard', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------

def _plot_psi(
    expected_pct: np.ndarray,
    actual_pct: np.ndarray,
    psi_bins: np.ndarray,
    psi: float,
    label: str
) -> None:
    """Plot PSI distribution comparison and bin contributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    bin_labels = [f'B{i+1}' for i in range(len(expected_pct))]

    axes[0].bar(bin_labels, expected_pct, alpha=0.6, color='steelblue', label='Development')
    axes[0].bar(bin_labels, actual_pct, alpha=0.6, color='orange', label='Current')
    axes[0].set_title(f'{label} Distribution Comparison')
    axes[0].set_ylabel('Proportion')
    axes[0].legend()

    bar_colors = ['green' if x < 0.01 else 'orange' if x < 0.025 else 'red'
                  for x in psi_bins]
    axes[1].bar(bin_labels, psi_bins, color=bar_colors, alpha=0.8)
    axes[1].axhline(y=0.025, color='orange', linestyle='--', label='Amber per-bin')
    axes[1].set_title(f'PSI Bin Contributions\nTotal PSI = {psi:.4f}')
    axes[1].set_ylabel('PSI Contribution')
    axes[1].legend()

    plt.tight_layout()
    plt.show()