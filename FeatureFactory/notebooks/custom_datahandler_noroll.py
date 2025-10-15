"""
Custom DataHandler (NON-ROLLING) for Hull Tactical using Qlib.

Structure mirrors the original custom_datahandler.py, but get_feature_config()
returns ONLY non-rolling expressions: lags, ROC, deltas, ratios; NO windowed ops.
"""

from typing import List, Union, Optional
import pandas as pd
import numpy as np

try:
    from qlib.data.dataset.handler import DataHandlerLP
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    print("Warning: Qlib not installed. Install with: pip install pyqlib")


class HullTacticalHandler(DataHandlerLP if QLIB_AVAILABLE else object):
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
        if not QLIB_AVAILABLE:
            raise ImportError("Qlib is required. Install with: pip install pyqlib")

        if not infer_processors:
            infer_processors = [
                {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ]
        if not learn_processors:
            learn_processors = [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ]

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

    def get_feature_config(self) -> List[str]:
        """
        Non-rolling features in Qlib expression syntax.
        """
        fields: List[str] = []

        # 1) Lags (example on M1)
        for lag in [1, 5, 10]:
            fields.append(f"Ref($M1, {lag})")

        # 2) Momentum (ROC) (example on P1)
        for p in [5, 10, 20]:
            fields.append(f"($P1 - Ref($P1, {p})) / Ref($P1, {p})")

        # 3) Simple deltas (example M1/V1)
        fields.append("$M1 - Ref($M1, 1)")
        fields.append("$V1 - Ref($V1, 1)")

        # 4) Ratios (example pairs)
        fields.append("$M1 / $M2")
        fields.append("$V1 / $V2")

        # NO: Mean/Std/Max/Min/Median, Quantile, EMA/MACD/BB
        return fields

    def get_label_config(self) -> List[Union[str, tuple]]:
        # Same as original: forward (t+1) target
        return [("market_forward_excess_returns", "Ref($market_forward_excess_returns, -1)")]


class HullTacticalHandlerV2(DataHandlerLP if QLIB_AVAILABLE else object):
    """
    Enhanced non-rolling version: broader raw set + lags/ROC/deltas/ratios.
    """
    def get_feature_config(self) -> List[str]:
        fields: List[str] = []

        base_features = {
            'D': list(range(1, 10)),
            'E': list(range(1, 21)),
            'I': list(range(1, 10)),
            'M': list(range(1, 19)),
            'P': list(range(1, 14)),
            'S': list(range(1, 13)),
            'V': list(range(1, 14)),
        }

        # 1) Raw
        for prefix, nums in base_features.items():
            for num in nums:
                fields.append(f"${prefix}{num}")

        # 2) Lags (selected)
        for f in ['M1', 'V1', 'P1', 'S1', 'E1', 'I1']:
            for t in [1, 5, 10, 20]:
                fields.append(f"Ref(${f}, {t})")

        # 3) Momentum (ROC)
        for f in ['M1', 'P1']:
            for p in [5, 10, 20]:
                fields.append(f"(${f} - Ref(${f}, {p})) / Ref(${f}, {p})")

        # 4) Deltas
        for f in ['M1', 'V1', 'P1']:
            fields.append(f"${f} - Ref(${f}, 1)")

        # 5) Ratios
        fields += ["$M1 / $M2", "$V1 / $V2", "$P1 / $P2"]

        # NO rolling/vol/zscore/quantile/EMA
        return fields


def create_hull_handler(config: dict):
    if not QLIB_AVAILABLE:
        raise ImportError("Qlib is required")
    return HullTacticalHandler(
        instruments=config.get('instruments', 'SP500'),
        start_time=config.get('start_time'),
        end_time=config.get('end_time'),
        fit_start_time=config.get('fit_start_time'),
        fit_end_time=config.get('fit_end_time'),
    )
