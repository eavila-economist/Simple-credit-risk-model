"""
ifrs9.py
========
IFRS 9 Expected Credit Loss calculation, staging and SICR assessment
for the Credit Risk PD Model.

IFRS 9 Framework Overview:
    IFRS 9 replaced the IAS 39 incurred loss model with an Expected Credit
    Loss framework, requiring banks to provision for losses before they occur
    using forward-looking information. The standard classifies financial
    instruments into three stages based on credit quality relative to origination.

    ECL formula:
        ECL = PD × LGD × EAD × DF

    Where DF is a discount factor bringing future losses to present value.

Staging model:
    Stage 1 — Performing: 12-month ECL
    Stage 2 — SICR triggered: Lifetime ECL
    Stage 3 — Credit impaired (default): Lifetime ECL

SICR indicators implemented:
    Quantitative: Current PD > 2× origination PD
    Qualitative:  RevolvingUtilizationOfUnsecuredLines > 80%
    Backstop:     30 days past due (rebuttable presumption)

Regulatory relevance:
    This module implements the provisioning side of the dual-framework
    requirement. IFRS 9 provisions reduce CET1 capital directly through
    retained earnings. The interaction with Basel III capital is documented
    in the basel.py module.
"""

import numpy as np
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
import warnings


# ---------------------------------------------------------------------------
# Constants — regulatory defaults for Foundation IRB
# ---------------------------------------------------------------------------

# Foundation IRB prescribed LGD for retail unsecured exposures
LGD_RETAIL_UNSECURED = 0.75

# Effective annual discount rate for present value calculation
DISCOUNT_RATE = 0.05

# SICR quantitative threshold — PD doubling since origination
SICR_PD_MULTIPLIER = 2.0

# SICR qualitative threshold — revolving utilisation
SICR_UTILISATION_THRESHOLD = 0.80

# Backstop — days past due triggering rebuttable presumption
SICR_BACKSTOP_DPD = 30

# Default definition — Basel III and IFRS 9 Stage 3
DEFAULT_DPD = 90


# ---------------------------------------------------------------------------
# SICR Assessment
# ---------------------------------------------------------------------------

def assess_sicr(
    pd_current: np.ndarray,
    pd_origination: np.ndarray,
    utilisation: Optional[np.ndarray] = None,
    days_past_due: Optional[np.ndarray] = None,
    pd_multiplier: float = SICR_PD_MULTIPLIER,
    util_threshold: float = SICR_UTILISATION_THRESHOLD,
    backstop_dpd: int = SICR_BACKSTOP_DPD
) -> pd.DataFrame:
    """
    Assess Significant Increase in Credit Risk for each exposure.

    Three complementary SICR indicators are evaluated. An exposure triggers
    SICR if any indicator fires. This waterfall approach ensures that the
    quantitative indicator captures gradual deterioration, the qualitative
    indicator captures behavioural stress signals, and the backstop captures
    any remaining cases through the regulatory minimum standard.

    SICR indicators:
    ----------------
    Quantitative: pd_current > pd_multiplier × pd_origination
        Fires when current PD exceeds twice the origination PD.
        Calibrated to approximately double the expected lifetime loss —
        a material deterioration threshold consistent with ECB supervisory
        expectations on IFRS 9 implementation.

    Qualitative: utilisation > util_threshold
        Fires when revolving credit utilisation exceeds 80%.
        Borrowers near credit limit exhaustion have minimal financial buffer —
        a leading indicator of imminent financial stress that precedes
        payment delinquency in the data.

    Backstop: days_past_due >= backstop_dpd
        The IFRS 9 rebuttable presumption — 30 days past due.
        Banks may rebut this presumption with evidence that SICR has
        not occurred (e.g., administrative delays), but the burden of
        proof lies with the institution.

    Parameters
    ----------
    pd_current : np.ndarray
        Current point-in-time PD estimates from the model.
    pd_origination : np.ndarray
        PD estimates at origination (stored at loan inception).
    utilisation : np.ndarray, optional
        Current revolving credit utilisation ratio.
    days_past_due : np.ndarray, optional
        Current days past due per exposure.
    pd_multiplier : float
        PD doubling threshold for quantitative SICR. Default: 2.0.
    util_threshold : float
        Utilisation threshold for qualitative SICR. Default: 0.80.
    backstop_dpd : int
        Days past due for backstop trigger. Default: 30.

    Returns
    -------
    pd.DataFrame
        SICR assessment results with indicator flags and combined SICR flag.
    """
    n = len(pd_current)

    results = pd.DataFrame({
        'pd_current': pd_current,
        'pd_origination': pd_origination,
        'pd_ratio': pd_current / np.maximum(pd_origination, 1e-10)
    })

    # Quantitative indicator
    results['sicr_quantitative'] = (
        pd_current > pd_multiplier * pd_origination
    ).astype(int)

    # Qualitative indicator
    if utilisation is not None:
        results['utilisation'] = utilisation
        results['sicr_qualitative'] = (utilisation > util_threshold).astype(int)
    else:
        results['sicr_qualitative'] = 0

    # Backstop indicator
    if days_past_due is not None:
        results['days_past_due'] = days_past_due
        results['sicr_backstop'] = (days_past_due >= backstop_dpd).astype(int)
    else:
        results['sicr_backstop'] = 0

    # Combined SICR — any indicator fires
    results['sicr_flag'] = (
        (results['sicr_quantitative'] == 1) |
        (results['sicr_qualitative'] == 1) |
        (results['sicr_backstop'] == 1)
    ).astype(int)

    # Summary
    print("SICR Assessment Summary:")
    print(f"  Quantitative trigger:  {results['sicr_quantitative'].sum():,} ({results['sicr_quantitative'].mean():.1%})")
    print(f"  Qualitative trigger:   {results['sicr_qualitative'].sum():,} ({results['sicr_qualitative'].mean():.1%})")
    print(f"  Backstop trigger:      {results['sicr_backstop'].sum():,} ({results['sicr_backstop'].mean():.1%})")
    print(f"  Combined SICR flag:    {results['sicr_flag'].sum():,} ({results['sicr_flag'].mean():.1%})")

    return results


# ---------------------------------------------------------------------------
# Staging classification
# ---------------------------------------------------------------------------

def classify_stages(
    pd_current: np.ndarray,
    pd_origination: np.ndarray,
    is_default: np.ndarray,
    utilisation: Optional[np.ndarray] = None,
    days_past_due: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Classify each exposure into IFRS 9 Stage 1, 2, or 3.

    Stage assignment waterfall:
        Stage 3: is_default == 1 (90+ days past due or credit impaired)
        Stage 2: SICR triggered (not yet in default)
        Stage 1: All remaining performing exposures

    Parameters
    ----------
    pd_current : np.ndarray
        Current model PD estimates.
    pd_origination : np.ndarray
        Origination PD estimates.
    is_default : np.ndarray
        Binary default indicator (1 = in default / Stage 3).
    utilisation : np.ndarray, optional
        Current revolving credit utilisation.
    days_past_due : np.ndarray, optional
        Current days past due.

    Returns
    -------
    np.ndarray
        Stage assignments (1, 2, or 3) for each exposure.
    """
    sicr_results = assess_sicr(
        pd_current, pd_origination,
        utilisation, days_past_due
    )

    stages = np.ones(len(pd_current), dtype=int)  # Default Stage 1

    # Stage 2 — SICR triggered (not in default)
    stages[sicr_results['sicr_flag'].values == 1] = 2

    # Stage 3 — In default (overrides Stage 2)
    stages[is_default == 1] = 3

    stage_counts = pd.Series(stages).value_counts().sort_index()
    print("\nStage Distribution:")
    for stage, count in stage_counts.items():
        pct = count / len(stages) * 100
        print(f"  Stage {stage}: {count:,} ({pct:.1f}%)")

    return stages


# ---------------------------------------------------------------------------
# ECL Calculation
# ---------------------------------------------------------------------------

def calculate_12month_ecl(
    pd_12m: np.ndarray,
    lgd: np.ndarray,
    ead: np.ndarray,
    discount_rate: float = DISCOUNT_RATE
) -> np.ndarray:
    """
    Calculate 12-month Expected Credit Loss for Stage 1 exposures.

    ECL_12m = PD_12m × LGD × EAD × DF_6m

    The discount factor uses a 6-month mid-period convention, assuming
    defaults occur on average at the midpoint of the 12-month horizon.

    Parameters
    ----------
    pd_12m : np.ndarray
        12-month probability of default from model.
    lgd : np.ndarray
        Loss given default (proportion of EAD).
    ead : np.ndarray
        Exposure at default in currency units.
    discount_rate : float
        Annual discount rate. Default: 0.05.

    Returns
    -------
    np.ndarray
        12-month ECL per exposure in currency units.
    """
    # Discount factor — mid-period convention (6 months)
    df_6m = 1 / (1 + discount_rate) ** 0.5

    ecl_12m = pd_12m * lgd * ead * df_6m

    return ecl_12m


def calculate_lifetime_ecl(
    pd_12m: np.ndarray,
    lgd: np.ndarray,
    ead: np.ndarray,
    remaining_life_years: np.ndarray,
    discount_rate: float = DISCOUNT_RATE
) -> np.ndarray:
    """
    Calculate lifetime Expected Credit Loss for Stage 2 and Stage 3 exposures.

    Lifetime ECL uses a constant hazard rate term structure:
        PD_lifetime = 1 - (1 - PD_12m)^T

    Where T is the remaining life in years. This assumes the annual
    default hazard rate remains constant over the loan's remaining life —
    a simplification of the full forward PD term structure that a production
    implementation would derive from vintage analysis.

    The present value calculation applies period-specific discount factors:
        ECL_lifetime = Σ_t [PD_t | survival × LGD × EAD × DF_t]

    Where PD_t | survival is the conditional probability of default in year t
    given survival through year t-1.

    Parameters
    ----------
    pd_12m : np.ndarray
        Annual probability of default (constant hazard assumption).
    lgd : np.ndarray
        Loss given default.
    ead : np.ndarray
        Exposure at default.
    remaining_life_years : np.ndarray
        Remaining contractual life in years per exposure.
    discount_rate : float
        Annual discount rate. Default: 0.05.

    Returns
    -------
    np.ndarray
        Lifetime ECL per exposure in currency units.
    """
    max_life = int(np.ceil(remaining_life_years.max()))
    n = len(pd_12m)
    lifetime_ecl = np.zeros(n)

    for i in range(n):
        T = remaining_life_years[i]
        annual_pd = pd_12m[i]
        period_ecl_sum = 0.0
        survival_prob = 1.0

        for t in range(1, int(np.ceil(T)) + 1):
            # Fraction of year for final partial period
            period_fraction = min(t, T) - (t - 1)

            # Conditional PD in this period given survival
            conditional_pd = annual_pd * period_fraction

            # Discount factor — mid-period convention
            df_t = 1 / (1 + discount_rate) ** (t - 0.5)

            # Period ECL contribution
            period_ecl_sum += survival_prob * conditional_pd * lgd[i] * ead[i] * df_t

            # Update survival probability
            survival_prob *= (1 - conditional_pd)

        lifetime_ecl[i] = period_ecl_sum

    return lifetime_ecl


def calculate_portfolio_ecl(
    pd_current: np.ndarray,
    pd_origination: np.ndarray,
    is_default: np.ndarray,
    ead: np.ndarray,
    lgd: Optional[np.ndarray] = None,
    utilisation: Optional[np.ndarray] = None,
    days_past_due: Optional[np.ndarray] = None,
    remaining_life_years: Optional[np.ndarray] = None,
    discount_rate: float = DISCOUNT_RATE,
    average_loan_life_years: float = 3.0
) -> pd.DataFrame:
    """
    Calculate IFRS 9 ECL for the full portfolio with staging.

    This is the master ECL calculation function that orchestrates staging,
    ECL computation by stage, and produces the portfolio-level provisioning
    summary required for financial statement disclosure under IFRS 9.

    Parameters
    ----------
    pd_current : np.ndarray
        Current model PD estimates (point-in-time).
    pd_origination : np.ndarray
        PD at origination. If None, uses pd_current as proxy.
    is_default : np.ndarray
        Binary default indicator (1 = in default).
    ead : np.ndarray
        Exposure at default per loan (outstanding balance).
    lgd : np.ndarray, optional
        Loss given default per loan.
        If None, applies regulatory LGD_RETAIL_UNSECURED = 75%.
    utilisation : np.ndarray, optional
        Current revolving credit utilisation for SICR qualitative trigger.
    days_past_due : np.ndarray, optional
        Current days past due for SICR backstop trigger.
    remaining_life_years : np.ndarray, optional
        Remaining loan life per exposure.
        If None, applies average_loan_life_years uniformly.
    discount_rate : float
        Annual discount rate. Default: 0.05.
    average_loan_life_years : float
        Average remaining loan life when individual lives unavailable.
        Default: 3.0 years.

    Returns
    -------
    pd.DataFrame
        Loan-level results with stage, ECL amount, and coverage ratio.
        Also prints portfolio-level summary.
    """
    n = len(pd_current)

    # Default LGD and remaining life if not provided
    if lgd is None:
        lgd = np.full(n, LGD_RETAIL_UNSECURED)
        print(f"Using regulatory LGD: {LGD_RETAIL_UNSECURED:.0%} (FIRB retail unsecured)")

    if remaining_life_years is None:
        remaining_life_years = np.full(n, average_loan_life_years)
        print(f"Using average remaining life: {average_loan_life_years} years")

    # Stage classification
    print("\nClassifying exposures into IFRS 9 stages...")
    stages = classify_stages(
        pd_current, pd_origination, is_default,
        utilisation, days_past_due
    )

    # Calculate ECL by stage
    print("\nCalculating ECL by stage...")
    ecl = np.zeros(n)

    # Stage 1 — 12-month ECL
    mask_s1 = stages == 1
    if mask_s1.any():
        ecl[mask_s1] = calculate_12month_ecl(
            pd_current[mask_s1],
            lgd[mask_s1],
            ead[mask_s1],
            discount_rate
        )

    # Stage 2 — Lifetime ECL
    mask_s2 = stages == 2
    if mask_s2.any():
        ecl[mask_s2] = calculate_lifetime_ecl(
            pd_current[mask_s2],
            lgd[mask_s2],
            ead[mask_s2],
            remaining_life_years[mask_s2],
            discount_rate
        )

    # Stage 3 — Lifetime ECL (same formula, higher PD)
    mask_s3 = stages == 3
    if mask_s3.any():
        ecl[mask_s3] = calculate_lifetime_ecl(
            pd_current[mask_s3],
            lgd[mask_s3],
            ead[mask_s3],
            remaining_life_years[mask_s3],
            discount_rate
        )

    # Compile results
    results_df = pd.DataFrame({
        'stage': stages,
        'pd_current': pd_current,
        'pd_origination': pd_origination,
        'lgd': lgd,
        'ead': ead,
        'ecl': ecl,
        'coverage_ratio': ecl / np.maximum(ead, 1e-10)
    })

    # Portfolio summary
    _print_ecl_summary(results_df)

    return results_df


def _print_ecl_summary(results_df: pd.DataFrame) -> None:
    """Print portfolio-level IFRS 9 ECL summary."""
    total_ead = results_df['ead'].sum()
    total_ecl = results_df['ecl'].sum()
    coverage = total_ecl / total_ead * 100

    print(f"\n{'=' * 60}")
    print("IFRS 9 ECL — PORTFOLIO SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total portfolio exposure (EAD): ${total_ead:,.0f}")
    print(f"Total provisions (ECL):         ${total_ecl:,.0f}")
    print(f"Overall coverage ratio:         {coverage:.2f}%")

    print(f"\nBy Stage:")
    for stage in [1, 2, 3]:
        mask = results_df['stage'] == stage
        stage_ead = results_df.loc[mask, 'ead'].sum()
        stage_ecl = results_df.loc[mask, 'ecl'].sum()
        stage_count = mask.sum()
        stage_coverage = stage_ecl / max(stage_ead, 1) * 100
        ecl_label = '12m ECL' if stage == 1 else 'Lifetime ECL'
        print(f"  Stage {stage} ({ecl_label}):")
        print(f"    Exposures: {stage_count:,} ({stage_count/len(results_df):.1%})")
        print(f"    EAD:       ${stage_ead:,.0f} ({stage_ead/total_ead:.1%})")
        print(f"    ECL:       ${stage_ecl:,.0f}")
        print(f"    Coverage:  {stage_coverage:.2f}%")


# ---------------------------------------------------------------------------
# P&L and Balance Sheet impact
# ---------------------------------------------------------------------------

def calculate_pnl_impact(
    ecl_current: float,
    ecl_prior: float,
    interest_income: float,
    portfolio_label: str = "Consumer Portfolio"
) -> pd.DataFrame:
    """
    Calculate the IFRS 9 provisioning impact on the income statement.

    The income statement impact of IFRS 9 has two components:
    1. Interest income on the gross carrying amount (all stages)
    2. Impairment charge — the change in ECL provisions period over period

    The net interest margin is compressed when Stage 2 and Stage 3
    borrowers accumulate, because provisions consume a larger share of
    interest income while Stage 3 borrowers only generate interest on
    their net carrying amount.

    Parameters
    ----------
    ecl_current : float
        Total ECL provision at current reporting date.
    ecl_prior : float
        Total ECL provision at prior reporting date.
    interest_income : float
        Gross interest income for the period.
    portfolio_label : str
        Portfolio identifier for reporting.

    Returns
    -------
    pd.DataFrame
        Simplified income statement showing IFRS 9 impact.
    """
    impairment_charge = ecl_current - ecl_prior
    net_income_after_impairment = interest_income - impairment_charge
    coverage_of_interest = impairment_charge / interest_income * 100

    pnl = pd.DataFrame([
        {'Line item': 'Gross interest income', 'Amount': interest_income},
        {'Line item': 'Impairment charge (ΔProvisions)', 'Amount': -impairment_charge},
        {'Line item': 'Net income after impairment', 'Amount': net_income_after_impairment},
        {'Line item': 'Impairment as % of interest income', 'Amount': f"{coverage_of_interest:.1f}%"}
    ])

    print(f"\nIFRS 9 P&L Impact — {portfolio_label}")
    print(pnl.to_string(index=False))

    return pnl


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_ecl_summary(results_df: pd.DataFrame) -> None:
    """
    Visualise IFRS 9 staging and ECL distribution across the portfolio.

    Parameters
    ----------
    results_df : pd.DataFrame
        Loan-level results from calculate_portfolio_ecl().
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = {1: '#22c55e', 2: '#f59e0b', 3: '#ef4444'}
    labels = {1: 'Stage 1\n(12m ECL)', 2: 'Stage 2\n(Lifetime)', 3: 'Stage 3\n(Default)'}

    # Stage distribution by count
    stage_counts = results_df['stage'].value_counts().sort_index()
    axes[0].bar(
        [labels[s] for s in stage_counts.index],
        stage_counts.values,
        color=[colors[s] for s in stage_counts.index]
    )
    axes[0].set_title('Exposure Count by Stage')
    axes[0].set_ylabel('Number of Exposures')

    # Stage distribution by EAD
    stage_ead = results_df.groupby('stage')['ead'].sum()
    axes[1].bar(
        [labels[s] for s in stage_ead.index],
        stage_ead.values / 1e6,
        color=[colors[s] for s in stage_ead.index]
    )
    axes[1].set_title('Exposure (EAD) by Stage')
    axes[1].set_ylabel('EAD ($M)')

    # Coverage ratio by stage
    stage_coverage = results_df.groupby('stage').apply(
        lambda x: x['ecl'].sum() / x['ead'].sum() * 100
    )
    axes[2].bar(
        [labels[s] for s in stage_coverage.index],
        stage_coverage.values,
        color=[colors[s] for s in stage_coverage.index]
    )
    axes[2].set_title('Coverage Ratio by Stage')
    axes[2].set_ylabel('ECL / EAD (%)')

    plt.suptitle('IFRS 9 Portfolio Staging and Provisioning', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()