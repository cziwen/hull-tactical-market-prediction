"""
Compute factors (NON-ROLLING) without Qlib dependency.

This mirrors the original structure & function names in compute_all_factors.py,
but strips ALL rolling/windowed families:
- NO: rolling Mean/Std/Max/Min/Median, EMA/MACD/BB, Quantiles, Z-score, CV/Range
- YES: Lags, Momentum (ROC), Deltas, Ratios/Diffs

Public API unchanged:
- compute_all_factors(df, base_features=None) -> pd.DataFrame
- get_computed_factor_names(df_original, df_with_factors) -> List[str]
"""

import pandas as pd
import numpy as np
from typing import List


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)


def compute_all_factors(df: pd.DataFrame, base_features: List[str] = None) -> pd.DataFrame:
    """
    Compute comprehensive NON-ROLLING factor set (lags/ROC/deltas/ratios).

    Args:
        df: DataFrame with raw features
        base_features: List of base feature names to use (default: M1, V1, P1, S1)

    Returns:
        DataFrame with original columns + computed factors
    """
    if base_features is None:
        base_features = ['M1', 'V1', 'P1', 'S1']
        base_features = [f for f in base_features if f in df.columns]

    df_feat = df.copy()
    factor_count = 0

    print(f"Computing factors for {len(base_features)} base features...")

    # 1) LAG FEATURES  (kept)
    print("  - Lag features...")
    for feat in base_features:
        for lag in [1, 2, 5, 10, 20, 60]:
            df_feat[f"LAG_{feat}_T{lag}"] = df[feat].shift(lag)
            factor_count += 1

    # 2) ROLLING STATISTICS  (removed)
    print("  - Rolling statistics... [skipped: non-rolling mode]")

    # 3) MOMENTUM FEATURES (ROC)  (kept)
    print("  - Momentum features...")
    for feat in base_features:
        for period in [5, 10, 20, 60]:
            prev = df[feat].shift(period)
            df_feat[f"MOM_{feat}_T{period}"] = _safe_div(df[feat] - prev, prev)
            factor_count += 1

    # 4) VOLATILITY FEATURES  (removed)
    print("  - Volatility features... [skipped: non-rolling mode]")

    # 5) Z-SCORE FEATURES  (removed)
    print("  - Z-score features... [skipped: non-rolling mode]")

    # 6) RATIO/CROSS FEATURES  (kept; same pairs)
    print("  - Ratio features...")
    if 'M1' in df.columns and 'M2' in df.columns:
        df_feat['RATIO_M1_DIV_M2'] = _safe_div(df['M1'], df['M2'])
        df_feat['DIFF_M1_SUB_M2'] = df['M1'] - df['M2']
        factor_count += 2

    if 'M1' in df.columns and 'M3' in df.columns:
        df_feat['RATIO_M1_DIV_M3'] = _safe_div(df['M1'], df['M3'])
        df_feat['DIFF_M1_SUB_M3'] = df['M1'] - df['M3']
        factor_count += 2

    if 'V1' in df.columns and 'V2' in df.columns:
        df_feat['RATIO_V1_DIV_V2'] = _safe_div(df['V1'], df['V2'])
        df_feat['DIFF_V1_SUB_V2'] = df['V1'] - df['V2']
        factor_count += 2

    if 'P1' in df.columns and 'P2' in df.columns:
        df_feat['RATIO_P1_DIV_P2'] = _safe_div(df['P1'], df['P2'])
        df_feat['DIFF_P1_SUB_P2'] = df['P1'] - df['P2']
        factor_count += 2

    # 7) TECHNICAL INDICATORS  (remove EMA/MACD/BB; keep simple ROC(10))
    print("  - Technical indicators (ROC only)...")
    for feat in base_features:
        df_feat[f"TECH_ROC_{feat}_10"] = _safe_div(df[feat] - df[feat].shift(10), df[feat].shift(10))
        factor_count += 1

    # 8) QUANTILE FEATURES  (removed)
    print("  - Quantile features... [skipped: non-rolling mode]")

    print(f"\n✓ Computed {factor_count} factors")
    print(f"  DataFrame shape: {df.shape} → {df_feat.shape}")
    return df_feat


def get_computed_factor_names(df_original: pd.DataFrame, df_with_factors: pd.DataFrame) -> List[str]:
    """Get list of computed factor column names (same helper as original)."""
    original_cols = set(df_original.columns)
    factor_cols = [c for c in df_with_factors.columns if c not in original_cols]
    return sorted(factor_cols)


if __name__ == "__main__":
    # Optional demo (same style as original)
    print("This is the NON-ROLLING variant; no demo data loaded here.")
