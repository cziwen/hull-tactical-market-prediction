"""
Compute all 200+ factors without Qlib dependency.

This module provides manual factor computation for use in notebooks
when Qlib is not available or you want direct control.
"""

import pandas as pd
import numpy as np
from typing import List


def compute_all_factors(df: pd.DataFrame, base_features: List[str] = None) -> pd.DataFrame:
    """
    Compute comprehensive factor set manually (without Qlib).

    Args:
        df: DataFrame with raw features
        base_features: List of base feature names to use (default: M1, V1, P1, S1, E1, I1)

    Returns:
        DataFrame with original columns + computed factors
    """
    if base_features is None:
        # Use important features that exist in data
        base_features = ['M1', 'V1', 'P1', 'S1']
        base_features = [f for f in base_features if f in df.columns]

    df_feat = df.copy()
    factor_count = 0

    print(f"Computing factors for {len(base_features)} base features...")

    # 1. LAG FEATURES
    print("  - Lag features...")
    for feat in base_features:
        for lag in [1, 2, 5, 10, 20, 60]:
            df_feat[f'LAG_{feat}_T{lag}'] = df[feat].shift(lag)
            factor_count += 1

    # 2. ROLLING STATISTICS
    print("  - Rolling statistics...")
    for feat in base_features:
        for window in [5, 10, 20, 60, 120]:
            # Mean
            df_feat[f'ROLL_MEAN_{feat}_W{window}'] = df[feat].rolling(window).mean()
            factor_count += 1

            # Std
            df_feat[f'ROLL_STD_{feat}_W{window}'] = df[feat].rolling(window).std()
            factor_count += 1

            # Max
            df_feat[f'ROLL_MAX_{feat}_W{window}'] = df[feat].rolling(window).max()
            factor_count += 1

            # Min
            df_feat[f'ROLL_MIN_{feat}_W{window}'] = df[feat].rolling(window).min()
            factor_count += 1

            # Median
            df_feat[f'ROLL_MEDIAN_{feat}_W{window}'] = df[feat].rolling(window).median()
            factor_count += 1

    # 3. MOMENTUM FEATURES
    print("  - Momentum features...")
    for feat in base_features:
        for period in [5, 10, 20, 60]:
            df_feat[f'MOM_{feat}_T{period}'] = (
                (df[feat] - df[feat].shift(period)) / df[feat].shift(period)
            )
            factor_count += 1

    # 4. VOLATILITY FEATURES
    print("  - Volatility features...")
    for feat in base_features:
        for window in [5, 10, 20, 60]:
            # Coefficient of variation
            mean = df[feat].rolling(window).mean()
            std = df[feat].rolling(window).std()
            df_feat[f'VOL_CV_{feat}_W{window}'] = std / mean
            factor_count += 1

            # Range (max-min) / mean
            max_val = df[feat].rolling(window).max()
            min_val = df[feat].rolling(window).min()
            df_feat[f'VOL_RANGE_{feat}_W{window}'] = (max_val - min_val) / mean
            factor_count += 1

    # 5. Z-SCORE FEATURES
    print("  - Z-score features...")
    for feat in base_features:
        for window in [20, 60, 120]:
            mean = df[feat].rolling(window).mean()
            std = df[feat].rolling(window).std()
            df_feat[f'ZSCORE_{feat}_W{window}'] = (df[feat] - mean) / std
            factor_count += 1

    # 6. RATIO/CROSS FEATURES
    print("  - Ratio features...")
    if 'M1' in df.columns and 'M2' in df.columns:
        df_feat['RATIO_M1_DIV_M2'] = df['M1'] / df['M2']
        df_feat['DIFF_M1_SUB_M2'] = df['M1'] - df['M2']
        factor_count += 2

    if 'M1' in df.columns and 'M3' in df.columns:
        df_feat['RATIO_M1_DIV_M3'] = df['M1'] / df['M3']
        df_feat['DIFF_M1_SUB_M3'] = df['M1'] - df['M3']
        factor_count += 2

    if 'V1' in df.columns and 'V2' in df.columns:
        df_feat['RATIO_V1_DIV_V2'] = df['V1'] / df['V2']
        df_feat['DIFF_V1_SUB_V2'] = df['V1'] - df['V2']
        factor_count += 2

    if 'P1' in df.columns and 'P2' in df.columns:
        df_feat['RATIO_P1_DIV_P2'] = df['P1'] / df['P2']
        df_feat['DIFF_P1_SUB_P2'] = df['P1'] - df['P2']
        factor_count += 2

    # 7. TECHNICAL INDICATORS
    print("  - Technical indicators...")
    for feat in base_features:
        # EMA approximation (using exponential weighted mean)
        if feat in df.columns:
            ema_12 = df[feat].ewm(span=12, adjust=False).mean()
            ema_26 = df[feat].ewm(span=26, adjust=False).mean()

            # MACD-style
            df_feat[f'TECH_MACD_{feat}'] = (ema_12 - ema_26) / ema_26
            factor_count += 1

            # Bollinger Band position
            mean_20 = df[feat].rolling(20).mean()
            std_20 = df[feat].rolling(20).std()
            df_feat[f'TECH_BB_POSITION_{feat}'] = (df[feat] - mean_20) / (2 * std_20)
            factor_count += 1

            # Rate of change (10 days)
            df_feat[f'TECH_ROC_{feat}_10'] = (df[feat] - df[feat].shift(10)) / df[feat].shift(10)
            factor_count += 1

    # 8. QUANTILE FEATURES
    print("  - Quantile features...")
    for feat in base_features:
        for window in [20, 60]:
            for q in [0.25, 0.5, 0.75]:
                df_feat[f'QUANTILE_{feat}_W{window}_Q{int(q*100)}'] = (
                    df[feat].rolling(window).quantile(q)
                )
                factor_count += 1

    print(f"\n✓ Computed {factor_count} factors")
    print(f"  DataFrame shape: {df.shape} → {df_feat.shape}")

    return df_feat


def get_computed_factor_names(df_original: pd.DataFrame, df_with_factors: pd.DataFrame) -> List[str]:
    """Get list of computed factor column names."""
    original_cols = set(df_original.columns)
    factor_cols = [c for c in df_with_factors.columns if c not in original_cols]
    return sorted(factor_cols)


if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    df = pd.read_csv('../data/processed/SP500.csv', parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    print(f"Original data shape: {df.shape}")

    # Compute all factors
    df_with_factors = compute_all_factors(df)

    # Get factor names
    factor_cols = get_computed_factor_names(df, df_with_factors)

    print(f"\nComputed {len(factor_cols)} factors")
    print(f"\nFirst 10 factors:")
    for i, factor in enumerate(factor_cols[:10], 1):
        print(f"  {i:2d}. {factor}")
