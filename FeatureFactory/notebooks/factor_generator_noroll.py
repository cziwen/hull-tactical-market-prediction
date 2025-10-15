"""
Factor generator using Qlib expression engine (NON-ROLLING).

Removes all windowed ops (Mean/Std/Max/Min/Median/Quantile, EMA/MACD/BB),
volatility, and z-score. Keeps: lags, momentum (ROC), ratios/diffs, simple ROC(10).
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from itertools import combinations
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactorGenerator:
    """Generate non-rolling alpha factor expressions in Qlib syntax."""

    def __init__(self, feature_prefixes: Optional[Dict[str, int]] = None):
        if feature_prefixes is None:
            self.feature_prefixes = {
                'D': 9, 'E': 20, 'I': 9, 'M': 18, 'P': 13, 'S': 12, 'V': 13,
            }
        else:
            self.feature_prefixes = feature_prefixes
        self.factors = []

    def get_all_base_features(self) -> List[str]:
        feats = []
        for prefix, count in self.feature_prefixes.items():
            feats.extend([f"{prefix}{i}" for i in range(1, count + 1)])
        return feats

    # ---------- KEPT FAMILIES (non-rolling) ----------

    def generate_lag_factors(
        self,
        features: Optional[List[str]] = None,
        lags: List[int] = [1, 2, 5, 10, 20, 60],
    ) -> List[Tuple[str, str]]:
        if features is None:
            features = ['M1', 'V1', 'P1', 'S1', 'E1', 'I1']
        out = []
        for f in features:
            for t in lags:
                out.append((f"LAG_{f}_T{t}", f"Ref(${f}, {t})"))
        logger.info(f"Generated {len(out)} lag factors")
        return out

    def generate_momentum_factors(
        self,
        features: Optional[List[str]] = None,
        periods: List[int] = [5, 10, 20, 60],
    ) -> List[Tuple[str, str]]:
        if features is None:
            features = ['M1', 'P1', 'V1']
        out = []
        for f in features:
            for p in periods:
                out.append((f"MOM_{f}_T{p}", f"(${f} - Ref(${f}, {p})) / Ref(${f}, {p})"))
        logger.info(f"Generated {len(out)} momentum factors")
        return out

    def generate_ratio_factors(
        self,
        feature_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> List[Tuple[str, str]]:
        if feature_pairs is None:
            feature_pairs = [
                ('M1','M2'), ('M1','M3'), ('M2','M3'),
                ('V1','V2'), ('V1','V3'),
                ('P1','P2'), ('P1','P3'),
                ('S1','S2'),
            ]
        out = []
        for a, b in feature_pairs:
            out.append((f"RATIO_{a}_DIV_{b}", f"${a} / ${b}"))
            out.append((f"DIFF_{a}_SUB_{b}",  f"${a} - ${b}"))
        logger.info(f"Generated {len(out)} ratio/diff factors")
        return out

    def generate_technical_indicators(
        self,
        features: Optional[List[str]] = None
    ) -> List[Tuple[str, str]]:
        # Keep only simple rate-of-change (ROC); remove EMA/MACD/BB
        if features is None:
            features = ['M1', 'P1', 'V1']
        out = []
        for f in features:
            out.append((f"TECH_ROC_{f}_10", f"(${f} - Ref(${f}, 10)) / Ref(${f}, 10)"))
        logger.info(f"Generated {len(out)} technical ROC factors")
        return out

    # ---------- REMOVED FAMILIES (rolling/windowed) ----------
    # generate_volatility_factors  -> removed
    # generate_zscore_factors      -> removed
    # generate_quantile_factors    -> removed
    # generate_rolling_stats       -> removed

    def generate_all_factors(
        self,
        factor_types: Optional[List[str]] = None
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Generate selected non-rolling factor types.

        Valid options now: ['lag', 'momentum', 'ratio', 'technical']
        """
        if factor_types is None:
            factor_types = ['lag', 'momentum', 'ratio', 'technical']

        all_factors: Dict[str, List[Tuple[str, str]]] = {}

        if 'lag' in factor_types:
            all_factors['lag'] = self.generate_lag_factors()
        if 'momentum' in factor_types:
            all_factors['momentum'] = self.generate_momentum_factors()
        if 'ratio' in factor_types:
            all_factors['ratio'] = self.generate_ratio_factors()
        if 'technical' in factor_types:
            all_factors['technical'] = self.generate_technical_indicators()

        total = sum(len(v) for v in all_factors.values())
        logger.info(f"Generated {total} total non-rolling factors across {len(all_factors)} types")
        return all_factors

    def export_factor_list(
        self,
        factors: Dict[str, List[Tuple[str, str]]],
        output_path: str
    ) -> pd.DataFrame:
        rows = []
        for cat, flist in factors.items():
            for name, expr in flist:
                rows.append({'factor_name': name, 'expression': expr, 'category': cat})
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} factors to {output_path}")
        return df


def main():
    gen = FactorGenerator()
    all_fac = gen.generate_all_factors()
    print("\nFactor Generation Summary:")
    print("-" * 60)
    for cat, lst in all_fac.items():
        print(f"{cat.capitalize():15s}: {len(lst):5d} factors")
    print("-" * 60)
    print(f"{'Total':15s}: {sum(len(v) for v in all_fac.values()):5d} factors")

    # Demo export (optional)
    # gen.export_factor_list(all_fac, "../outputs/factor_library_noroll.csv")


if __name__ == "__main__":
    main()
