"""
basel.py
========
Basel III Internal Ratings Based (IRB) capital formula implementation
for the Credit Risk PD Model.

Mathematical foundation — Asymptotic Single Risk Factor (ASRF) Model:
    The Basel III IRB formula is derived from the ASRF model, which decomposes
    each borrower's default risk into two components:

        A_i = √R × Z + √(1-R) × ε_i

    Where:
        Z   = systematic risk factor (state of the economy) ~ N(0,1)
        ε_i = idiosyncratic risk factor (borrower-specific) ~ N(0,1)
        R   = asset correlation (weight of the systematic factor)

    The capital requirement is calibrated to cover losses at the 99.9th
    percentile of the loss distribution — the bank survives all but the
    most extreme 0.1% of economic scenarios.

    Stressed conditional PD at the 99.9th percentile scenario:
        PD_stressed = N(N⁻¹(PD)/√(1-R) + √(R/(1-R)) × N⁻¹(0.999))

    Capital requirement per unit of EAD:
        K = [LGD × PD_stressed - PD × LGD] × Maturity Adjustment

    RWA = K × 12.5 × EAD

    Asset correlation for retail exposures (Basel III formula):
        R = 0.03 × (1-e^(-35×PD))/(1-e^(-35)) +
            0.16 × (1-(1-e^(-35×PD))/(1-e^(-35)))

    Maturity adjustment:
        β = [0.11852 - 0.05478 × log(PD)]²
        MA = (1 + (M - 2.5) × β) / (1 - 1.5 × β)

Regulatory relevance:
    The IRB approach allows banks to use internal PD estimates to calculate
    risk-sensitive RWA rather than the blunt risk weights of the Standardised
    Approach. Capital relief from IRB adoption — demonstrated in notebook 06 —
    provides the business case for investing in model development and validation
    infrastructure under SR 11-7.

    The output floor (72.5% of SA RWA) introduced by Basel III finalisation
    (2017) is implemented to ensure IRB RWA cannot fall below this minimum
    regardless of how low internal PD estimates are.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants — regulatory parameters
# ---------------------------------------------------------------------------

# Capital conservation buffer — total CET1 requirement
CET1_REQUIREMENT = 0.105  # 7% minimum + 2.5% conservation buffer = 10.5%

# Output floor — Basel III finalisation (2017)
# IRB RWA cannot fall below 72.5% of SA RWA
OUTPUT_FLOOR = 0.725

# Foundation IRB prescribed parameters for retail unsecured
LGD_FIRB_RETAIL = 0.75    # 75% LGD for retail unsecured
M_RETAIL = 1.0             # Effective maturity = 1 year for retail (simplified)

# Standardised Approach risk weights
RW_REGULATORY_RETAIL = 0.75   # 75% for qualifying retail
RW_PAST_DUE = 1.50            # 150% for past due exposures (provisions < 20%)
RW_OTHER_RETAIL = 1.00        # 100% for other retail

# Confidence level for capital calculation
CONFIDENCE_LEVEL = 0.999      # 99.9th percentile


# ---------------------------------------------------------------------------
# Asset correlation
# ---------------------------------------------------------------------------

def calculate_asset_correlation(PD: np.ndarray) -> np.ndarray:
    """
    Calculate asset correlation R for retail exposures using the Basel III formula.

    The asset correlation measures the sensitivity of a borrower's asset value
    to the systematic (macroeconomic) risk factor. Higher correlation means
    more of the borrower's risk is driven by common economic conditions rather
    than idiosyncratic factors.

    Basel III retail correlation formula:
        R = 0.03 × (1-e^(-35×PD))/(1-e^(-35)) +
            0.16 × (1-(1-e^(-35×PD))/(1-e^(-35)))

    R ranges from 0.03 (highest PD, most idiosyncratic) to 0.16 (lowest PD,
    most systematic). This inverse relationship reflects that:
    - High-PD borrowers fail primarily due to individual circumstances
      (idiosyncratic risk) — their defaults are more diversifiable
    - Low-PD borrowers fail primarily during severe systemic events
      (systematic risk) — their defaults cluster in economic downturns

    Parameters
    ----------
    PD : np.ndarray
        Probability of Default estimates (between 0 and 1).

    Returns
    -------
    np.ndarray
        Asset correlation R for each exposure.
    """
    # Floor PD at minimum to avoid numerical issues
    PD = np.clip(PD, 1e-6, 1 - 1e-6)

    # Basel III retail asset correlation formula
    factor = (1 - np.exp(-35 * PD)) / (1 - np.exp(-35))
    R = 0.03 * factor + 0.16 * (1 - factor)

    return R


# ---------------------------------------------------------------------------
# Maturity adjustment
# ---------------------------------------------------------------------------

def calculate_maturity_adjustment(PD: np.ndarray, M: float = M_RETAIL) -> np.ndarray:
    """
    Calculate the Basel III maturity adjustment for the IRB capital formula.

    The maturity adjustment scales capital requirements upward for longer
    maturity exposures. Longer maturity means more time for credit quality
    to deteriorate — more risk — hence more capital required.

    The effect is larger for low-PD borrowers (for whom additional time
    creates more opportunity for deterioration) and smaller for high-PD
    borrowers (who are already likely to default soon regardless of maturity).

    Maturity adjustment formula:
        β = [0.11852 - 0.05478 × log(PD)]²
        MA = (1 + (M - 2.5) × β) / (1 - 1.5 × β)

    For retail exposures M = 1.0 year (simplified), giving MA = 1.0 when
    M = 2.5 (the neutral maturity) and MA < 1.0 for M = 1.0. This means
    retail exposures actually receive a maturity discount relative to the
    neutral 2.5-year benchmark.

    Parameters
    ----------
    PD : np.ndarray
        Probability of Default estimates.
    M : float
        Effective maturity in years. Default: 1.0 (retail).

    Returns
    -------
    np.ndarray
        Maturity adjustment factor for each exposure.
    """
    PD = np.clip(PD, 1e-6, 1 - 1e-6)

    beta = (0.11852 - 0.05478 * np.log(PD)) ** 2
    maturity_adj = (1 + (M - 2.5) * beta) / (1 - 1.5 * beta)

    # Floor at zero (can be negative for very short maturities with high PD)
    maturity_adj = np.maximum(maturity_adj, 0)

    return maturity_adj


# ---------------------------------------------------------------------------
# Stressed conditional PD
# ---------------------------------------------------------------------------

def calculate_stressed_pd(
    PD: np.ndarray,
    R: np.ndarray,
    confidence_level: float = CONFIDENCE_LEVEL
) -> np.ndarray:
    """
    Calculate the stressed conditional PD at the specified confidence level.

    The ASRF model gives the conditional default probability when the
    systematic risk factor Z takes its worst-case value at the confidence
    level percentile:

        PD_stressed = N(N⁻¹(PD)/√(1-R) + √(R/(1-R)) × N⁻¹(0.999))

    At the 99.9th percentile scenario (Z* = N⁻¹(0.999) ≈ 3.09 standard
    deviations below mean — extreme recession), the conditional PD for each
    borrower is dramatically higher than the unconditional PD. This stressed
    PD is the basis for the capital requirement calculation.

    Parameters
    ----------
    PD : np.ndarray
        Unconditional (through-the-cycle) PD estimates.
    R : np.ndarray
        Asset correlation for each exposure.
    confidence_level : float
        Confidence level for stress scenario. Default: 0.999 (99.9th percentile).

    Returns
    -------
    np.ndarray
        Stressed conditional PD for each exposure.
    """
    PD = np.clip(PD, 1e-6, 1 - 1e-6)

    # Worst-case systematic factor realisation
    Z_star = norm.ppf(confidence_level)  # ≈ 3.09 for 99.9%

    # Stressed conditional PD
    PD_stressed = norm.cdf(
        norm.ppf(PD) / np.sqrt(1 - R) +
        np.sqrt(R / (1 - R)) * Z_star
    )

    return PD_stressed


# ---------------------------------------------------------------------------
# IRB Capital Formula
# ---------------------------------------------------------------------------

def calculate_irb_capital(
    PD: np.ndarray,
    EAD: np.ndarray,
    LGD: Optional[np.ndarray] = None,
    M: float = M_RETAIL,
    confidence_level: float = CONFIDENCE_LEVEL
) -> pd.DataFrame:
    """
    Calculate Basel III IRB capital requirements using the ASRF formula.

    Full IRB capital formula per exposure:
        R   = asset_correlation(PD)
        MA  = maturity_adjustment(PD, M)
        PD* = stressed_pd(PD, R, 0.999)
        K   = [LGD × PD* - PD × LGD] × MA
        RWA = K × 12.5 × EAD
        Capital = RWA × 10.5%

    The PD × LGD term subtracted within K is the expected loss — already
    covered by IFRS 9 provisions. K captures only the unexpected loss
    beyond expected loss, which is what regulatory capital must absorb.

    Parameters
    ----------
    PD : np.ndarray
        Through-the-cycle PD estimates from the internal model.
    EAD : np.ndarray
        Exposure at default in currency units (outstanding balance).
    LGD : np.ndarray, optional
        Loss given default per exposure.
        If None, applies FIRB regulatory value of 75%.
    M : float
        Effective maturity in years. Default: 1.0 for retail.
    confidence_level : float
        VaR confidence level. Default: 0.999.

    Returns
    -------
    pd.DataFrame
        Loan-level IRB results including R, PD_stressed, K, RWA, and capital.
    """
    n = len(PD)
    PD = np.clip(PD, 1e-6, 1 - 1e-6)

    if LGD is None:
        LGD = np.full(n, LGD_FIRB_RETAIL)

    # Step 1 — Asset correlation
    R = calculate_asset_correlation(PD)

    # Step 2 — Maturity adjustment
    MA = calculate_maturity_adjustment(PD, M)

    # Step 3 — Stressed conditional PD
    PD_stressed = calculate_stressed_pd(PD, R, confidence_level)

    # Step 4 — Capital requirement per unit of EAD
    # Unexpected loss = stressed EL - expected EL
    unexpected_loss = LGD * PD_stressed - PD * LGD
    K = np.maximum(unexpected_loss * MA, 0)  # Floor at zero

    # Step 5 — RWA and capital
    RWA = K * 12.5 * EAD
    Capital = RWA * CET1_REQUIREMENT

    results = pd.DataFrame({
        'pd': PD,
        'lgd': LGD,
        'ead': EAD,
        'asset_correlation_R': R.round(4),
        'maturity_adjustment': MA.round(4),
        'pd_stressed': PD_stressed.round(4),
        'expected_loss': (PD * LGD * EAD).round(2),
        'K': K.round(6),
        'rwa': RWA.round(2),
        'capital_required': Capital.round(2),
        'el_rate': (PD * LGD).round(4),
        'ul_rate': K.round(6)
    })

    return results


# ---------------------------------------------------------------------------
# Standardised Approach
# ---------------------------------------------------------------------------

def calculate_sa_capital(
    EAD: np.ndarray,
    is_past_due: np.ndarray,
    provision_rate: float = 0.0
) -> pd.DataFrame:
    """
    Calculate Basel III Standardised Approach RWA and capital.

    SA risk weights for retail consumer exposures:
        Qualifying regulatory retail: 75%
        Past due (>90 DPD, provisions < 20%): 150%
        Past due (>90 DPD, provisions >= 20%): 100%

    The SA applies uniform risk weights regardless of individual borrower PD.
    This is the fundamental inefficiency the IRB approach addresses — a
    borrower with 0.1% PD receives the same 75% risk weight as one with 45%.

    Parameters
    ----------
    EAD : np.ndarray
        Exposure at default per loan.
    is_past_due : np.ndarray
        Binary indicator — 1 if exposure is 90+ days past due.
    provision_rate : float
        Current provision as fraction of EAD for past due loans.
        Determines whether 100% or 150% weight applies.

    Returns
    -------
    pd.DataFrame
        Loan-level SA results with risk weights, RWA and capital.
    """
    n = len(EAD)

    # Assign risk weights
    risk_weights = np.where(
        is_past_due == 1,
        np.where(provision_rate >= 0.20, RW_PAST_DUE - 0.50, RW_PAST_DUE),
        RW_REGULATORY_RETAIL
    )

    RWA = risk_weights * EAD
    Capital = RWA * CET1_REQUIREMENT

    results = pd.DataFrame({
        'ead': EAD,
        'is_past_due': is_past_due,
        'risk_weight': risk_weights,
        'rwa': RWA.round(2),
        'capital_required': Capital.round(2)
    })

    return results


# ---------------------------------------------------------------------------
# SA vs IRB comparison
# ---------------------------------------------------------------------------

def compare_sa_irb(
    irb_results: pd.DataFrame,
    sa_results: pd.DataFrame,
    apply_output_floor: bool = True
) -> dict:
    """
    Compare IRB and SA capital requirements and calculate capital relief.

    The output floor (72.5% of SA RWA) prevents IRB models from producing
    RWA below this minimum, ensuring a floor on capital requirements regardless
    of how low internal PD estimates are. This prevents gaming of internal
    models while preserving the risk sensitivity benefits of IRB.

    Parameters
    ----------
    irb_results : pd.DataFrame
        Loan-level IRB results from calculate_irb_capital().
    sa_results : pd.DataFrame
        Loan-level SA results from calculate_sa_capital().
    apply_output_floor : bool
        Whether to apply the 72.5% output floor. Default: True.

    Returns
    -------
    dict
        Summary statistics comparing IRB and SA approaches.
    """
    total_ead = sa_results['ead'].sum()
    total_rwa_sa = sa_results['rwa'].sum()
    total_capital_sa = sa_results['capital_required'].sum()

    total_rwa_irb_raw = irb_results['rwa'].sum()
    total_capital_irb_raw = irb_results['capital_required'].sum()

    # Apply output floor
    floor_rwa = OUTPUT_FLOOR * total_rwa_sa
    if apply_output_floor and total_rwa_irb_raw < floor_rwa:
        total_rwa_irb = floor_rwa
        floor_binding = True
    else:
        total_rwa_irb = total_rwa_irb_raw
        floor_binding = False

    total_capital_irb = total_rwa_irb * CET1_REQUIREMENT

    capital_relief = total_capital_sa - total_capital_irb
    capital_relief_pct = capital_relief / total_capital_sa * 100

    rwa_density_sa = total_rwa_sa / total_ead * 100
    rwa_density_irb = total_rwa_irb / total_ead * 100

    summary = {
        'total_ead': total_ead,
        'rwa_sa': total_rwa_sa,
        'rwa_irb_pre_floor': total_rwa_irb_raw,
        'rwa_irb_post_floor': total_rwa_irb,
        'output_floor_binding': floor_binding,
        'capital_sa': total_capital_sa,
        'capital_irb': total_capital_irb,
        'capital_relief': capital_relief,
        'capital_relief_pct': capital_relief_pct,
        'rwa_density_sa_pct': rwa_density_sa,
        'rwa_density_irb_pct': rwa_density_irb
    }

    print(f"\n{'=' * 60}")
    print("BASEL III — SA vs IRB CAPITAL COMPARISON")
    print(f"{'=' * 60}")
    print(f"Total portfolio EAD:         ${total_ead:>15,.0f}")
    print(f"\nSTANDARDISED APPROACH:")
    print(f"  Total RWA:                 ${total_rwa_sa:>15,.0f}")
    print(f"  RWA density:               {rwa_density_sa:>14.1f}%")
    print(f"  Capital required (10.5%):  ${total_capital_sa:>15,.0f}")
    print(f"\nINTERNAL RATINGS BASED:")
    print(f"  Total RWA (pre-floor):     ${total_rwa_irb_raw:>15,.0f}")
    print(f"  Output floor (72.5% SA):   ${floor_rwa:>15,.0f}")
    print(f"  Floor binding:             {'Yes — IRB RWA floored' if floor_binding else 'No — IRB below floor'}")
    print(f"  Total RWA (post-floor):    ${total_rwa_irb:>15,.0f}")
    print(f"  RWA density:               {rwa_density_irb:>14.1f}%")
    print(f"  Capital required (10.5%):  ${total_capital_irb:>15,.0f}")
    print(f"\nCAPITAL RELIEF FROM IRB ADOPTION:")
    print(f"  Capital relief:            ${capital_relief:>15,.0f}")
    print(f"  Relief as % of SA capital: {capital_relief_pct:>14.1f}%")

    return summary


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_irb_analysis(
    irb_results: pd.DataFrame,
    sa_comparison: dict,
    figsize: tuple = (16, 5)
) -> None:
    """
    Visualise IRB capital analysis results.

    Three panels:
    1. PD vs Capital requirement K — demonstrates non-linearity
    2. PD vs Asset correlation R — shows inverse relationship
    3. SA vs IRB capital comparison bar chart

    Parameters
    ----------
    irb_results : pd.DataFrame
        Loan-level IRB results.
    sa_comparison : dict
        Portfolio-level SA vs IRB comparison from compare_sa_irb().
    figsize : tuple
        Figure size.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panel 1 — PD vs K (capital requirement rate)
    pd_range = np.linspace(0.001, 0.50, 500)
    R_range = calculate_asset_correlation(pd_range)
    MA_range = calculate_maturity_adjustment(pd_range, M_RETAIL)
    PD_stressed_range = calculate_stressed_pd(pd_range, R_range)
    K_range = np.maximum(LGD_FIRB_RETAIL * PD_stressed_range - pd_range * LGD_FIRB_RETAIL, 0) * MA_range

    axes[0].plot(pd_range * 100, K_range * 100, color='steelblue', linewidth=2)
    axes[0].set_xlabel('PD (%)')
    axes[0].set_ylabel('Capital Requirement K (%)')
    axes[0].set_title('IRB Capital Requirement vs PD\n(Non-linear ASRF relationship)')
    axes[0].grid(True, alpha=0.3)

    # Mark portfolio average
    avg_pd = irb_results['pd'].mean()
    avg_K = irb_results['K'].mean()
    axes[0].axvline(x=avg_pd * 100, color='red', linestyle='--', alpha=0.7,
                    label=f'Portfolio avg PD={avg_pd:.1%}')
    axes[0].legend(fontsize=9)

    # Panel 2 — PD vs Asset Correlation R
    axes[1].plot(pd_range * 100, R_range, color='#7c3aed', linewidth=2)
    axes[1].set_xlabel('PD (%)')
    axes[1].set_ylabel('Asset Correlation R')
    axes[1].set_title('Asset Correlation vs PD\n(Inverse relationship — retail)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0.03, color='gray', linestyle=':', alpha=0.7, label='Min R = 0.03')
    axes[1].axhline(y=0.16, color='gray', linestyle='--', alpha=0.7, label='Max R = 0.16')
    axes[1].legend(fontsize=9)

    # Panel 3 — SA vs IRB capital comparison
    labels = ['SA Capital', 'IRB Capital\n(post-floor)', 'Capital Relief']
    values = [
        sa_comparison['capital_sa'] / 1e6,
        sa_comparison['capital_irb'] / 1e6,
        sa_comparison['capital_relief'] / 1e6
    ]
    bar_colors = ['#ef4444', '#3b82f6', '#22c55e']
    bars = axes[2].bar(labels, values, color=bar_colors, alpha=0.8)
    axes[2].set_ylabel('Capital Required ($M)')
    axes[2].set_title(f"SA vs IRB Capital\nRelief: {sa_comparison['capital_relief_pct']:.1f}%")

    for bar, val in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'${val:.1f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.suptitle('Basel III IRB Capital Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_capital_by_pd_bucket(irb_results: pd.DataFrame) -> None:
    """
    Show capital requirements and RWA density by PD bucket.

    Demonstrates how IRB allocates more capital to higher-PD borrowers
    and less to lower-PD borrowers — the risk sensitivity that motivates
    IRB adoption over the Standardised Approach.

    Parameters
    ----------
    irb_results : pd.DataFrame
        Loan-level IRB results from calculate_irb_capital().
    """
    # Create PD buckets
    irb_results = irb_results.copy()
    irb_results['pd_bucket'] = pd.cut(
        irb_results['pd'],
        bins=[0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0],
        labels=['0-2%', '2-5%', '5-10%', '10-20%', '20-50%', '50%+']
    )

    bucket_summary = irb_results.groupby('pd_bucket').agg(
        n=('pd', 'count'),
        avg_pd=('pd', 'mean'),
        total_ead=('ead', 'sum'),
        total_rwa=('rwa', 'sum'),
        total_capital=('capital_required', 'sum')
    ).reset_index()

    bucket_summary['rwa_density'] = bucket_summary['total_rwa'] / bucket_summary['total_ead'] * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(
        bucket_summary['pd_bucket'].astype(str),
        bucket_summary['rwa_density'],
        color='steelblue', alpha=0.8
    )
    axes[0].axhline(y=75, color='red', linestyle='--', label='SA retail weight (75%)')
    axes[0].set_xlabel('PD Bucket')
    axes[0].set_ylabel('RWA Density (%)')
    axes[0].set_title('IRB RWA Density by PD Bucket\n(vs 75% flat SA weight)')
    axes[0].legend()

    axes[1].bar(
        bucket_summary['pd_bucket'].astype(str),
        bucket_summary['n'],
        color='#7c3aed', alpha=0.8
    )
    axes[1].set_xlabel('PD Bucket')
    axes[1].set_ylabel('Number of Exposures')
    axes[1].set_title('Portfolio Distribution by PD Bucket')

    plt.suptitle('IRB Capital Sensitivity Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()