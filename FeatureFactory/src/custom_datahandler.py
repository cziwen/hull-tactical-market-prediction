"""
Custom DataHandler for Hull Tactical data using Qlib framework.

This handler extends Qlib's capabilities to work with the Hull Tactical dataset,
supporting custom feature engineering with Qlib's expression engine.
"""

from typing import List, Union, Optional
import pandas as pd
import numpy as np

try:
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.processor import (
        Processor,
        ProcessorFactory,
    )
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    print("Warning: Qlib not installed. Install with: pip install pyqlib")


class HullTacticalHandler(DataHandlerLP if QLIB_AVAILABLE else object):
    """
    Custom DataHandler for Hull Tactical Market Prediction.

    Inherits from Qlib's DataHandlerLP and provides:
    1. Feature engineering via expression engine
    2. Custom processors for missing data
    3. Factor library similar to Alpha158
    """

    def __init__(
        self,
        instruments="SP500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processors=None,
        **kwargs
    ):
        """
        Args:
            instruments: Stock pool (default: SP500)
            start_time: Data start time
            end_time: Data end time
            freq: Data frequency (default: day)
            infer_processors: Processors for inference
            learn_processors: Processors for training
        """
        if not QLIB_AVAILABLE:
            raise ImportError("Qlib is required. Install with: pip install pyqlib")

        # Initialize processors if not provided
        if not infer_processors:
            infer_processors = self._get_default_infer_processors()
        if not learn_processors:
            learn_processors = self._get_default_learn_processors()

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,
            filter_pipe=filter_pipe,
            inst_processors=inst_processors,
            **kwargs
        )

    @staticmethod
    def _get_default_infer_processors():
        """Default processors for inference phase."""
        return [
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
        ]

    @staticmethod
    def _get_default_learn_processors():
        """Default processors for training phase."""
        return [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ]

    def get_feature_config(self) -> List[str]:
        """
        Get feature expressions using Qlib's expression engine.

        This method defines features similar to Alpha158 but adapted for Hull Tactical data.
        Returns list of feature expressions.
        """
        fields = []

        # Original features (94 features: D, E, I, M, P, S, V)
        feature_prefixes = ['D', 'E', 'I', 'M', 'P', 'S', 'V']
        for prefix in feature_prefixes:
            # Add raw features (will be determined from data)
            # This will be populated dynamically
            pass

        # Time-series features using Qlib operators
        # Example: Create lag features, rolling statistics, momentum indicators

        # 1. Lag features (t-1, t-5, t-10)
        for lag in [1, 5, 10]:
            fields.append(f"Ref($M1, {lag})")  # Example: Market feature lag

        # 2. Rolling mean features
        for window in [5, 10, 20, 60]:
            fields.append(f"Mean($V1, {window})")  # Volatility rolling mean
            fields.append(f"Mean($M1, {window})")  # Market rolling mean

        # 3. Rolling std features
        for window in [5, 10, 20, 60]:
            fields.append(f"Std($V1, {window})")
            fields.append(f"Std($M1, {window})")

        # 4. Momentum features
        for window in [5, 10, 20]:
            fields.append(f"($P1 - Ref($P1, {window})) / Ref($P1, {window})")  # Price momentum

        # 5. Max/Min features
        for window in [5, 10, 20]:
            fields.append(f"Max($V1, {window})")
            fields.append(f"Min($V1, {window})")

        # 6. Quantile features
        for window in [10, 20]:
            fields.append(f"Quantile($M1, {window}, 0.25)")
            fields.append(f"Quantile($M1, {window}, 0.75)")

        # 7. Delta features (difference)
        fields.append("$V1 - Ref($V1, 1)")
        fields.append("$M1 - Ref($M1, 1)")

        # 8. Ratio features
        fields.append("$M1 / $M2")
        fields.append("$V1 / $V2")

        # 9. EMA (Exponential Moving Average)
        for window in [5, 10, 20]:
            fields.append(f"EMA($M1, {window})")

        # 10. MACD-style indicators
        fields.append("(EMA($M1, 12) - EMA($M1, 26)) / EMA($M1, 26)")

        return fields

    def get_label_config(self) -> List[Union[str, tuple]]:
        """
        Get label configuration.

        For Hull Tactical, the target is market_forward_excess_returns.
        """
        return [
            ("market_forward_excess_returns", "Ref($market_forward_excess_returns, -1)")
            # Negative ref means future value (t+1)
        ]


class HullTacticalHandlerV2(DataHandlerLP if QLIB_AVAILABLE else object):
    """
    Enhanced version with comprehensive factor library.

    Generates 200+ features covering:
    - Raw features
    - Lag features
    - Rolling statistics
    - Technical indicators
    - Cross-feature interactions
    """

    def get_feature_config(self) -> List[str]:
        """Generate comprehensive feature set."""
        fields = []

        # Define base features from each category
        base_features = {
            'D': list(range(1, 10)),      # D1-D9 (dummy/binary)
            'E': list(range(1, 21)),      # E1-E20 (economic)
            'I': list(range(1, 10)),      # I1-I9 (interest rate)
            'M': list(range(1, 19)),      # M1-M18 (market dynamics)
            'P': list(range(1, 14)),      # P1-P13 (price/valuation)
            'S': list(range(1, 13)),      # S1-S12 (sentiment)
            'V': list(range(1, 14)),      # V1-V13 (volatility)
        }

        # 1. Raw features
        for prefix, nums in base_features.items():
            for num in nums:
                fields.append(f"${prefix}{num}")

        # 2. Lag features (selected important features)
        important_features = ['M1', 'V1', 'P1', 'S1', 'E1', 'I1']
        for feature in important_features:
            for lag in [1, 5, 10, 20]:
                fields.append(f"Ref(${feature}, {lag})")

        # 3. Rolling statistics
        windows = [5, 10, 20, 60]
        for feature in ['M1', 'V1', 'P1']:
            for w in windows:
                fields.append(f"Mean(${feature}, {w})")
                fields.append(f"Std(${feature}, {w})")
                fields.append(f"Max(${feature}, {w})")
                fields.append(f"Min(${feature}, {w})")

        # 4. Momentum indicators
        for feature in ['M1', 'P1']:
            for period in [5, 10, 20]:
                fields.append(f"(${feature} - Ref(${feature}, {period})) / Ref(${feature}, {period})")

        # 5. Volatility measures
        for period in [5, 10, 20]:
            fields.append(f"Std($M1, {period}) / Mean($M1, {period})")  # Coefficient of variation

        # 6. Cross-feature ratios
        fields.append("$M1 / $M2")
        fields.append("$V1 / $V2")
        fields.append("$P1 / $P2")

        # 7. Z-score features
        for feature in ['M1', 'V1']:
            for window in [20, 60]:
                fields.append(f"(${feature} - Mean(${feature}, {window})) / Std(${feature}, {window})")

        # 8. Quantile features
        for feature in ['M1', 'V1']:
            for window in [20, 60]:
                fields.append(f"Quantile(${feature}, {window}, 0.25)")
                fields.append(f"Quantile(${feature}, {window}, 0.75)")

        return fields


def create_hull_handler(config: dict):
    """
    Factory function to create HullTacticalHandler from config.

    Args:
        config: Dict with keys: start_time, end_time, fit_start_time, fit_end_time

    Returns:
        HullTacticalHandler instance
    """
    if not QLIB_AVAILABLE:
        raise ImportError("Qlib is required")

    return HullTacticalHandler(
        instruments=config.get('instruments', 'SP500'),
        start_time=config.get('start_time'),
        end_time=config.get('end_time'),
        fit_start_time=config.get('fit_start_time'),
        fit_end_time=config.get('fit_end_time'),
    )
