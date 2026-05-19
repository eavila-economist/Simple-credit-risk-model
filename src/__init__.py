"""
Credit Risk PD Model — Source Package
======================================
End-to-end Probability of Default modelling with regulatory framework integration.

Modules:
    cleaning    : Data cleaning pipeline — outliers, sentinels, imputation, transformation
    woe_iv      : Weight of Evidence and Information Value calculation and transformation
    model       : Model training, cross-validation, evaluation and calibration
    ifrs9       : IFRS 9 ECL calculation, staging and SICR assessment
    basel       : Basel III IRB capital formula — ASRF model implementation
    monitoring  : PSI, CSI and ongoing model performance monitoring

Regulatory frameworks implemented:
    IFRS 9      : Three-stage Expected Credit Loss model
    Basel III   : Asymptotic Single Risk Factor IRB capital formula
    SR 11-7     : Model development and validation documentation standard
"""

from . import cleaning
from . import woe_iv
from . import model
from . import ifrs9
from . import basel
from . import monitoring

__version__ = "1.0.0"
__author__ = "Enrique Ávila Muñoz"