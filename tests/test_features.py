import pandas as pd
import pytest
from src.features import FeatureEngineer


def test_feature_engineer_basic():
    df = pd.DataFrame(
        {
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
        }
    )
    fe = FeatureEngineer()
    out = fe.transform(df)
    assert 'emi_simple' in out.columns
    assert 'cibil_score_bin' in out.columns


def test_standardise_columns():
    df = pd.DataFrame(
        {
            ' Loan-Amount ': [1],
            'Income/Annum': [2],
            'Commercial Assets Value$': [3],
        }
    )
    fe = FeatureEngineer()
    out = fe._standardise_columns(df)
    assert out.columns.tolist() == [
        'loanamount',
        'incomeannum',
        'commercialassetsvalue',
    ]


def test_aggregate_assets_and_ratios_and_flag():
    df = pd.DataFrame(
        {
            'loan_amount': [90, 70],
            'residential_assets_value': [50, 30],
            'luxury_assets_value': [10, 0],
            # commercial_assets_value and bank_asset_value missing
            'cibil_score': [580, 650],
        }
    )
    fe = FeatureEngineer()
    with pytest.warns(UserWarning) as rec:
        df = fe._aggregate_assets(df)
    # two missing asset columns should trigger warnings
    assert len(rec) == 2
    assert {'commercial_assets_value', 'bank_asset_value'} <= set(df.columns)
    # totals
    assert df.loc[0, 'total_assets'] == 60
    assert df.loc[0, 'net_worth'] == -30
    df = fe._asset_ratios(df)
    assert pytest.approx(df.loc[0, 'luxury_asset_ratio']) == 10 / 60
    assert pytest.approx(df.loc[0, 'liquid_asset_ratio']) == 50 / 60
    assert df.loc[0, 'asset_diversity_count'] == 2
    df = fe._flag_highrisk(df)
    assert df.loc[0, 'highrisk_combo_flag'] == 1
    assert df.loc[1, 'highrisk_combo_flag'] == 0
