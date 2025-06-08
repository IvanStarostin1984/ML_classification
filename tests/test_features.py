from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.features import FeatureEngineer


def test_feature_engineer_basic():
  df = pd.DataFrame({
    'income_annum': [120000.0, 240000.0],
    'loan_amount': [100000.0, 200000.0],
    'loan_term': [12, 24],
    'cibil_score': [650, 700],
    'education': ['Graduate', 'Not Graduate'],
    'self_employed': ['No', 'Yes'],
    'residential_assets_value': [50000, 100000],
    'commercial_assets_value': [0, 0],
    'luxury_assets_value': [0, 0],
    'bank_asset_value': [0, 0],
    'gender': ['M', 'F'],
    'married': ['Yes', 'No'],
    'property_area': ['Urban', 'Rural'],
    'no_of_dependents': [0, 1],
  })
  fe = FeatureEngineer()
  out = fe.transform(df)
  assert 'emi_simple' in out.columns
  assert 'cibil_score_bin' in out.columns
