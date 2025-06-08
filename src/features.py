"""Feature engineering utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from pandas.api.types import CategoricalDtype

__all__ = ['FeatureEngineer']


class FeatureEngineer:
  """Encapsulates feature engineering logic."""

  MARKET_APR = 0.090

  def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    """Return engineered feature DataFrame."""
    df_fe = df.copy(deep=True)
    df_fe.columns = (
      df_fe.columns
      .str.strip()
      .str.lower()
      .str.replace(r"[ \t\-/]+", '', regex=True)
      .str.replace(r"[^\w]", '', regex=True)
    )

    def _zeros() -> pd.Series:
      return pd.Series(0.0, index=df_fe.index)

    if 'income_annum' in df_fe.columns:
      df_fe['total_income_month'] = df_fe['income_annum'].fillna(0) / 12.0
    elif 'incomeannum' in df_fe.columns:
      df_fe['total_income_month'] = df_fe['incomeannum'].fillna(0) / 12.0
    else:
      warnings.warn('No `income_annum` column – filling income with zeros.')
      df_fe['total_income_month'] = _zeros()

    asset_cols = [
      'residential_assets_value',
      'commercial_assets_value',
      'luxury_assets_value',
      'bank_asset_value',
    ]
    for c in asset_cols:
      if c not in df_fe.columns:
        warnings.warn(f'Asset column `{c}` missing – created 0-filled.')
        df_fe[c] = 0.0
    df_fe['total_assets'] = df_fe[asset_cols].sum(axis=1)
    df_fe['net_worth'] = df_fe['total_assets'] - df_fe['loan_amount']

    loan_med = df_fe['loan_amount'].median()
    term_med = df_fe['loan_term'].replace(0, np.nan).median()
    if not np.isfinite(term_med) or term_med == 0:
      warnings.warn('loan_term median 0/NaN – defaulting to 120 months.')
      term_med = 120

    df_fe['emi_simple'] = (
      df_fe['loan_amount'].fillna(loan_med) /
      df_fe['loan_term'].replace(0, np.nan).fillna(term_med)
    )

    r = self.MARKET_APR / 12.0
    n = df_fe['loan_term'].replace(0, np.nan).fillna(term_med)
    P = df_fe['loan_amount'].fillna(loan_med)
    df_fe['emi_amortised'] = (
      P * r * (1 + r) ** n / ((1 + r) ** n - 1 + 1e-6)
    )

    df_fe['debt_to_income_ratio'] = (
      df_fe['loan_amount'] / (df_fe['total_income_month'] * 12 + 1e-6)
    )
    df_fe['dscr'] = df_fe['total_income_month'] / (df_fe['emi_amortised'] + 1e-6)

    df_fe['log_loan_amount'] = np.log1p(df_fe['loan_amount'])
    df_fe['log_total_income_month'] = np.log1p(df_fe['total_income_month'])
    df_fe['log_total_assets'] = np.log1p(df_fe['total_assets'])

    df_fe['cibil_score_sq'] = df_fe['cibil_score'] ** 2
    cibil_cat = CategoricalDtype(
      ['poor', 'fair', 'good', 'verygood', 'excellent'], ordered=True
    )
    df_fe['cibil_score_bin'] = pd.cut(
      df_fe['cibil_score'],
      bins=[-np.inf, 579, 679, 779, 850, np.inf],
      labels=cibil_cat.categories
    ).astype(cibil_cat)

    df_fe['loan_term_bin'] = pd.cut(
      df_fe['loan_term'],
      bins=[0, 9, 12, 18, 24, np.inf],
      labels=['≤9 m', '10–12 m', '13–18 m', '19–24 m', '>24 m']
    )

    df_fe['number_of_dependents'] = (
      pd.to_numeric(df_fe.get('no_of_dependents', _zeros()), errors='coerce')
        .fillna(0)
        .astype(int)
    )
    df_fe['many_dependents_flag'] = (df_fe['number_of_dependents'] >= 3).astype(int)
    df_fe['income_per_dependent'] = (
      df_fe['total_income_month'] /
      (df_fe['number_of_dependents'] + 1)
    )

    df_fe['luxury_asset_ratio'] = (
      df_fe['luxury_assets_value'] / (df_fe['total_assets'] + 1e-6)
    )
    liquid_assets = (
      df_fe['bank_asset_value'].fillna(0) +
      df_fe['residential_assets_value'].fillna(0)
    )
    df_fe['liquid_asset_ratio'] = liquid_assets / (df_fe['total_assets'] + 1e-6)
    df_fe['asset_diversity_count'] = (df_fe[asset_cols] > 0).astype(int).sum(axis=1)

    df_fe['graduate_flag'] = (
      df_fe['education'].str.lower().str.contains('graduate').astype(int)
    )
    df_fe['income_times_graduate'] = (
      df_fe['total_income_month'] * df_fe['graduate_flag']
    )
    df_fe['self_employed_flag'] = (
      df_fe['self_employed'].astype(str).str.lower().isin(['yes', 'y', '1']).astype(int)
    )
    df_fe['selfemp_loan_to_income'] = (
      df_fe['self_employed_flag'] *
      (df_fe['loan_amount'] / (df_fe['total_income_month'] * 12 + 1e-6))
    )

    df_fe['highrisk_combo_flag'] = (
      (df_fe['cibil_score'] < 600) &
      (df_fe['loan_amount'] / (df_fe['total_assets'] + 1e-6) > 0.80)
    ).astype(int)

    cat_cols = [
      c
      for c in [
        'gender',
        'married',
        'self_employed',
        'property_area',
      ]
      if c in df_fe.columns
    ]
    if cat_cols:
      df_fe = pd.get_dummies(df_fe, columns=cat_cols, prefix_sep='=', drop_first=True)
    bool_cols = df_fe.select_dtypes('bool').columns
    df_fe[bool_cols] = df_fe[bool_cols].astype('uint8')

    n_missing = int(df_fe.isna().sum().sum())
    if n_missing:
      warnings.warn(f'Feature matrix has {n_missing} missing values.')

    return df_fe
