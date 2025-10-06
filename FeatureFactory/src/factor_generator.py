"""
Factor generator using Qlib's expression engine.

Generates alpha factors from base features using various transformations:
- Temporal: lag, rolling statistics, delta
- Statistical: z-score, quantile, rank
- Technical: momentum, volatility, trend
- Compositional: ratios, products, differences
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from itertools import combinations
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactorGenerator:
    """Generate alpha factors using Qlib expression syntax."""

    def __init__(self, feature_prefixes: Optional[List[str]] = None):
        """
        Args:
            feature_prefixes: List of feature prefixes (e.g., ['M', 'V', 'P'])
                             If None, uses all Hull Tactical prefixes
        """
        if feature_prefixes is None:
            self.feature_prefixes = {
                'D': 9,   # D1-D9 (dummy/binary)
                'E': 20,  # E1-E20 (economic)
                'I': 9,   # I1-I9 (interest rate)
                'M': 18,  # M1-M18 (market dynamics)
                'P': 13,  # P1-P13 (price/valuation)
                'S': 12,  # S1-S12 (sentiment)
                'V': 13,  # V1-V13 (volatility)
            }
        else:
            self.feature_prefixes = feature_prefixes

        self.factors = []

    def get_all_base_features(self) -> List[str]:
        """Get list of all base feature names."""
        features = []
        for prefix, count in self.feature_prefixes.items():
            for i in range(1, count + 1):
                features.append(f"{prefix}{i}")
        return features

    def generate_lag_factors(
        self,
        features: Optional[List[str]] = None,
        lags: List[int] = [1, 2, 5, 10, 20, 60]
    ) -> List[Tuple[str, str]]:
        """
        Generate lag features.

        Args:
            features: List of feature names (e.g., ['M1', 'V1'])
            lags: List of lag periods

        Returns:
            List of (factor_name, factor_expression) tuples
        """
        if features is None:
            features = ['M1', 'V1', 'P1', 'S1', 'E1', 'I1']  # Important features

        factors = []
        for feature in features:
            for lag in lags:
                name = f"LAG_{feature}_T{lag}"
                expr = f"Ref(${feature}, {lag})"
                factors.append((name, expr))

        logger.info(f"Generated {len(factors)} lag factors")
        return factors

    def generate_rolling_stats(
        self,
        features: Optional[List[str]] = None,
        windows: List[int] = [5, 10, 20, 60, 120],
        stats: List[str] = ['mean', 'std', 'max', 'min', 'median']
    ) -> List[Tuple[str, str]]:
        """
        Generate rolling statistics factors.

        Args:
            features: List of feature names
            windows: Rolling window sizes
            stats: Statistics to compute (mean, std, max, min, median, quantile)

        Returns:
            List of (factor_name, factor_expression) tuples
        """
        if features is None:
            features = ['M1', 'V1', 'P1', 'S1']

        factors = []
        stat_map = {
            'mean': 'Mean',
            'std': 'Std',
            'max': 'Max',
            'min': 'Min',
            'median': 'Median',
            'sum': 'Sum',
        }

        for feature in features:
            for window in windows:
                for stat in stats:
                    if stat in stat_map:
                        name = f"ROLL_{stat.upper()}_{feature}_W{window}"
                        expr = f"{stat_map[stat]}(${feature}, {window})"
                        factors.append((name, expr))

        logger.info(f"Generated {len(factors)} rolling statistics factors")
        return factors

    def generate_momentum_factors(
        self,
        features: Optional[List[str]] = None,
        periods: List[int] = [5, 10, 20, 60]
    ) -> List[Tuple[str, str]]:
        """
        Generate momentum factors (return over period).

        Args:
            features: List of feature names
            periods: Lookback periods

        Returns:
            List of (factor_name, factor_expression) tuples
        """
        if features is None:
            features = ['M1', 'P1', 'V1']

        factors = []
        for feature in features:
            for period in periods:
                # Simple return
                name = f"MOM_{feature}_T{period}"
                expr = f"(${feature} - Ref(${feature}, {period})) / Ref(${feature}, {period})"
                factors.append((name, expr))

        logger.info(f"Generated {len(factors)} momentum factors")
        return factors

    def generate_volatility_factors(
        self,
        features: Optional[List[str]] = None,
        windows: List[int] = [5, 10, 20, 60]
    ) -> List[Tuple[str, str]]:
        """
        Generate volatility-related factors.

        Args:
            features: List of feature names
            windows: Window sizes

        Returns:
            List of (factor_name, factor_expression) tuples
        """
        if features is None:
            features = ['M1', 'V1', 'P1']

        factors = []
        for feature in features:
            for window in windows:
                # Coefficient of variation
                name = f"VOL_CV_{feature}_W{window}"
                expr = f"Std(${feature}, {window}) / Mean(${feature}, {window})"
                factors.append((name, expr))

                # Range (max - min)
                name = f"VOL_RANGE_{feature}_W{window}"
                expr = f"(Max(${feature}, {window}) - Min(${feature}, {window})) / Mean(${feature}, {window})"
                factors.append((name, expr))

        logger.info(f"Generated {len(factors)} volatility factors")
        return factors

    def generate_zscore_factors(
        self,
        features: Optional[List[str]] = None,
        windows: List[int] = [20, 60, 120]
    ) -> List[Tuple[str, str]]:
        """
        Generate z-score normalization factors.

        Args:
            features: List of feature names
            windows: Window sizes for mean/std calculation

        Returns:
            List of (factor_name, factor_expression) tuples
        """
        if features is None:
            features = ['M1', 'V1', 'P1', 'S1']

        factors = []
        for feature in features:
            for window in windows:
                name = f"ZSCORE_{feature}_W{window}"
                expr = f"(${feature} - Mean(${feature}, {window})) / Std(${feature}, {window})"
                factors.append((name, expr))

        logger.info(f"Generated {len(factors)} z-score factors")
        return factors

    def generate_ratio_factors(
        self,
        feature_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> List[Tuple[str, str]]:
        """
        Generate ratio factors between feature pairs.

        Args:
            feature_pairs: List of (feature1, feature2) tuples
                          If None, generates pairs from same prefix

        Returns:
            List of (factor_name, factor_expression) tuples
        """
        if feature_pairs is None:
            # Generate pairs within same category
            feature_pairs = [
                ('M1', 'M2'), ('M1', 'M3'), ('M2', 'M3'),
                ('V1', 'V2'), ('V1', 'V3'),
                ('P1', 'P2'), ('P1', 'P3'),
                ('S1', 'S2'),
            ]

        factors = []
        for feat1, feat2 in feature_pairs:
            # Ratio
            name = f"RATIO_{feat1}_DIV_{feat2}"
            expr = f"${feat1} / ${feat2}"
            factors.append((name, expr))

            # Difference
            name = f"DIFF_{feat1}_SUB_{feat2}"
            expr = f"${feat1} - ${feat2}"
            factors.append((name, expr))

        logger.info(f"Generated {len(factors)} ratio/difference factors")
        return factors

    def generate_technical_indicators(
        self,
        features: Optional[List[str]] = None
    ) -> List[Tuple[str, str]]:
        """
        Generate technical indicator-style factors.

        Includes:
        - RSI-like indicators
        - MACD-like indicators
        - Bollinger Band-like indicators

        Args:
            features: List of feature names

        Returns:
            List of (factor_name, factor_expression) tuples
        """
        if features is None:
            features = ['M1', 'P1', 'V1']

        factors = []

        for feature in features:
            # MACD-style (fast EMA - slow EMA)
            name = f"TECH_MACD_{feature}"
            expr = f"(EMA(${feature}, 12) - EMA(${feature}, 26)) / EMA(${feature}, 26)"
            factors.append((name, expr))

            # Bollinger Band position
            name = f"TECH_BB_POSITION_{feature}"
            expr = f"(${feature} - Mean(${feature}, 20)) / (2 * Std(${feature}, 20))"
            factors.append((name, expr))

            # Rate of change
            name = f"TECH_ROC_{feature}_10"
            expr = f"(${feature} - Ref(${feature}, 10)) / Ref(${feature}, 10)"
            factors.append((name, expr))

        logger.info(f"Generated {len(factors)} technical indicator factors")
        return factors

    def generate_quantile_factors(
        self,
        features: Optional[List[str]] = None,
        windows: List[int] = [20, 60],
        quantiles: List[float] = [0.25, 0.5, 0.75]
    ) -> List[Tuple[str, str]]:
        """
        Generate quantile-based factors.

        Args:
            features: List of feature names
            windows: Window sizes
            quantiles: Quantile levels (0-1)

        Returns:
            List of (factor_name, factor_expression) tuples
        """
        if features is None:
            features = ['M1', 'V1', 'P1']

        factors = []
        for feature in features:
            for window in windows:
                for q in quantiles:
                    name = f"QUANTILE_{feature}_W{window}_Q{int(q*100)}"
                    expr = f"Quantile(${feature}, {window}, {q})"
                    factors.append((name, expr))

        logger.info(f"Generated {len(factors)} quantile factors")
        return factors

    def generate_all_factors(
        self,
        factor_types: Optional[List[str]] = None
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Generate all factor types.

        Args:
            factor_types: List of factor types to generate
                         Options: ['lag', 'rolling', 'momentum', 'volatility',
                                  'zscore', 'ratio', 'technical', 'quantile']
                         If None, generates all types

        Returns:
            Dict mapping factor type to list of (name, expression) tuples
        """
        if factor_types is None:
            factor_types = [
                'lag', 'rolling', 'momentum', 'volatility',
                'zscore', 'ratio', 'technical', 'quantile'
            ]

        all_factors = {}

        if 'lag' in factor_types:
            all_factors['lag'] = self.generate_lag_factors()

        if 'rolling' in factor_types:
            all_factors['rolling'] = self.generate_rolling_stats()

        if 'momentum' in factor_types:
            all_factors['momentum'] = self.generate_momentum_factors()

        if 'volatility' in factor_types:
            all_factors['volatility'] = self.generate_volatility_factors()

        if 'zscore' in factor_types:
            all_factors['zscore'] = self.generate_zscore_factors()

        if 'ratio' in factor_types:
            all_factors['ratio'] = self.generate_ratio_factors()

        if 'technical' in factor_types:
            all_factors['technical'] = self.generate_technical_indicators()

        if 'quantile' in factor_types:
            all_factors['quantile'] = self.generate_quantile_factors()

        total = sum(len(factors) for factors in all_factors.values())
        logger.info(f"Generated {total} total factors across {len(all_factors)} types")

        return all_factors

    def export_factor_list(
        self,
        factors: Dict[str, List[Tuple[str, str]]],
        output_path: str
    ) -> pd.DataFrame:
        """
        Export factor definitions to CSV.

        Args:
            factors: Dict from generate_all_factors()
            output_path: Path to save CSV

        Returns:
            DataFrame with columns: factor_name, expression, category
        """
        rows = []
        for category, factor_list in factors.items():
            for name, expr in factor_list:
                rows.append({
                    'factor_name': name,
                    'expression': expr,
                    'category': category
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} factors to {output_path}")

        return df


def main():
    """Example usage."""
    generator = FactorGenerator()

    # Generate all factor types
    all_factors = generator.generate_all_factors()

    # Print summary
    print("\nFactor Generation Summary:")
    print("-" * 50)
    for category, factors in all_factors.items():
        print(f"{category.capitalize():15s}: {len(factors):4d} factors")
    print("-" * 50)
    print(f"{'Total':15s}: {sum(len(f) for f in all_factors.values()):4d} factors")

    # Show examples
    print("\n\nExample factors:")
    for category, factors in all_factors.items():
        print(f"\n{category.upper()}:")
        for name, expr in factors[:3]:  # Show first 3
            print(f"  {name:40s} = {expr}")

    # Export
    df = generator.export_factor_list(all_factors, "../outputs/factor_library.csv")
    print(f"\nExported to factor_library.csv")


if __name__ == "__main__":
    main()
