"""Feature engineering utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from pandas.api.types import CategoricalDtype

__all__ = ["FeatureEngineer"]


class FeatureEngineer:
    """Encapsulates feature engineering logic."""

    MARKET_APR = 0.090
    ASSET_COLS = [
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value",
    ]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return engineered feature DataFrame."""
        df_fe = self._standardise_columns(df)
        df_fe = self._derive_income(df_fe)
        df_fe = self._aggregate_assets(df_fe)
        df_fe = self._compute_emi(df_fe)
        df_fe = self._derive_ratios(df_fe)
        df_fe = self._encode_scores(df_fe)
        df_fe = self._dependents_features(df_fe)
        df_fe = self._asset_ratios(df_fe)
        df_fe = self._education_employment_features(df_fe)
        df_fe = self._flag_highrisk(df_fe)
        df_fe = self._encode_categories(df_fe)
        self._warn_missing(df_fe)
        return df_fe

    def _standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalise column names."""
        df = df.copy(deep=True)
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(r"[ \t\-/]+", "", regex=True)
            .str.replace(r"[^\w]", "", regex=True)
        )
        return df

    def _derive_income(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create monthly income feature."""
        if "income_annum" in df.columns:
            df["total_income_month"] = df["income_annum"].fillna(0) / 12.0
        elif "incomeannum" in df.columns:
            df["total_income_month"] = df["incomeannum"].fillna(0) / 12.0
        else:
            warnings.warn("No `income_annum` column – filling income with zeros.")
            df["total_income_month"] = pd.Series(0.0, index=df.index)
        return df

    def _aggregate_assets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure asset columns and compute totals."""
        for c in self.ASSET_COLS:
            if c not in df.columns:
                warnings.warn(f"Asset column `{c}` missing – created 0-filled.")
                df[c] = 0.0
        df["total_assets"] = df[self.ASSET_COLS].sum(axis=1)
        df["net_worth"] = df["total_assets"] - df["loan_amount"]
        return df

    def _compute_emi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate simple and amortised EMIs."""
        loan_med = df["loan_amount"].median()
        term_med = df["loan_term"].replace(0, np.nan).median()
        if not np.isfinite(term_med) or term_med == 0:
            warnings.warn("loan_term median 0/NaN – defaulting to 120 months.")
            term_med = 120

        df["emi_simple"] = df["loan_amount"].fillna(loan_med) / df["loan_term"].replace(
            0, np.nan
        ).fillna(term_med)

        r = self.MARKET_APR / 12.0
        n = df["loan_term"].replace(0, np.nan).fillna(term_med)
        P = df["loan_amount"].fillna(loan_med)
        df["emi_amortised"] = P * r * (1 + r) ** n / ((1 + r) ** n - 1 + 1e-6)
        return df

    def _derive_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive financial ratios and logs."""
        df["debt_to_income_ratio"] = df["loan_amount"] / (
            df["total_income_month"] * 12 + 1e-6
        )
        df["dscr"] = df["total_income_month"] / (df["emi_amortised"] + 1e-6)
        df["log_loan_amount"] = np.log1p(df["loan_amount"])
        df["log_total_income_month"] = np.log1p(df["total_income_month"])
        df["log_total_assets"] = np.log1p(df["total_assets"])
        return df

    def _encode_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode credit scores and term bins."""
        df["cibil_score_sq"] = df["cibil_score"] ** 2
        cibil_cat = CategoricalDtype(
            ["poor", "fair", "good", "verygood", "excellent"], ordered=True
        )
        df["cibil_score_bin"] = pd.cut(
            df["cibil_score"],
            bins=[-np.inf, 579, 679, 779, 850, np.inf],
            labels=cibil_cat.categories,
        ).astype(cibil_cat)
        df["loan_term_bin"] = pd.cut(
            df["loan_term"],
            bins=[0, 9, 12, 18, 24, np.inf],
            labels=["≤9 m", "10–12 m", "13–18 m", "19–24 m", ">24 m"],
        )
        return df

    def _dependents_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add dependent count features."""
        dep = (
            pd.to_numeric(
                df.get("no_of_dependents", pd.Series(0, index=df.index)),
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )
        df["number_of_dependents"] = dep
        df["many_dependents_flag"] = (df["number_of_dependents"] >= 3).astype(int)
        df["income_per_dependent"] = df["total_income_month"] / (
            df["number_of_dependents"] + 1
        )
        return df

    def _asset_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create asset ratio features."""
        df["luxury_asset_ratio"] = df["luxury_assets_value"] / (
            df["total_assets"] + 1e-6
        )
        liquid_assets = df["bank_asset_value"].fillna(0) + df[
            "residential_assets_value"
        ].fillna(0)
        df["liquid_asset_ratio"] = liquid_assets / (df["total_assets"] + 1e-6)
        df["asset_diversity_count"] = (df[self.ASSET_COLS] > 0).astype(int).sum(axis=1)
        return df

    def _education_employment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add education and employment flags."""
        df["graduate_flag"] = (
            df["education"].str.lower().str.contains("graduate").astype(int)
        )
        df["income_times_graduate"] = df["total_income_month"] * df["graduate_flag"]
        df["self_employed_flag"] = (
            df["self_employed"].astype(str).str.lower().isin(["yes", "y", "1"])
        ).astype(int)
        df["selfemp_loan_to_income"] = df["self_employed_flag"] * (
            df["loan_amount"] / (df["total_income_month"] * 12 + 1e-6)
        )
        return df

    def _flag_highrisk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify risky borrower combinations."""
        df["highrisk_combo_flag"] = (
            (df["cibil_score"] < 600)
            & (df["loan_amount"] / (df["total_assets"] + 1e-6) > 0.80)
        ).astype(int)
        return df

    def _encode_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode common categorical features."""
        cat_cols = [
            c
            for c in ["gender", "married", "self_employed", "property_area"]
            if c in df.columns
        ]
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, prefix_sep="=", drop_first=True)
        bool_cols = df.select_dtypes("bool").columns
        df[bool_cols] = df[bool_cols].astype("uint8")
        return df

    def _warn_missing(self, df: pd.DataFrame) -> None:
        """Warn if the resulting DataFrame contains NaNs."""
        n_missing = int(df.isna().sum().sum())
        if n_missing:
            warnings.warn(f"Feature matrix has {n_missing} missing values.")
