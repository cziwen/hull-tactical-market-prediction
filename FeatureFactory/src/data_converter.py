"""
Convert Hull Tactical CSV data to Qlib binary format.

Qlib expects stock data with columns: date, instrument, [features...]
For single-asset case (S&P 500), we create a synthetic instrument column.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HullTacticalDataConverter:
    """Convert Hull Tactical CSV to Qlib-compatible format."""

    def __init__(self, data_path: str, output_path: str, instrument_name: str = "SP500"):
        """
        Args:
            data_path: Path to train.csv or test.csv
            output_path: Directory to save Qlib-compatible CSV
            instrument_name: Symbol name for the single asset (default: SP500)
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.instrument_name = instrument_name
        self.output_path.mkdir(parents=True, exist_ok=True)

    def convert(self, start_date: str = "1990-01-01") -> pd.DataFrame:
        """
        Convert CSV to Qlib format.

        Args:
            start_date: Starting date for date_id=0 (arbitrary reference point)

        Returns:
            DataFrame in Qlib format with columns: date, instrument, [features...]
        """
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)

        # Create date column from date_id (assuming daily frequency)
        base_date = pd.to_datetime(start_date)
        df['date'] = df['date_id'].apply(lambda x: base_date + timedelta(days=int(x)))

        # Add instrument column
        df['instrument'] = self.instrument_name

        # Reorder columns: date, instrument, features, targets
        feature_cols = [col for col in df.columns if col not in ['date_id', 'date', 'instrument']]
        ordered_cols = ['date', 'instrument'] + feature_cols
        df_qlib = df[ordered_cols]

        # Sort by date
        df_qlib = df_qlib.sort_values('date').reset_index(drop=True)

        logger.info(f"Converted {len(df_qlib)} rows with {len(feature_cols)} features")
        logger.info(f"Date range: {df_qlib['date'].min()} to {df_qlib['date'].max()}")
        logger.info(f"Missing value ratio: {df_qlib.isnull().sum().sum() / (df_qlib.shape[0] * df_qlib.shape[1]):.2%}")

        return df_qlib

    def save_for_qlib_dump(self, df: pd.DataFrame, filename: str = "SP500.csv") -> Path:
        """
        Save in format ready for Qlib's dump_bin.py script.

        Qlib expects one CSV per instrument with columns: date, [features...]
        """
        output_file = self.output_path / filename

        # Drop instrument column for single-stock CSV
        df_save = df.drop(columns=['instrument']).set_index('date')

        df_save.to_csv(output_file)
        logger.info(f"Saved Qlib-compatible CSV to {output_file}")

        return output_file

    def get_feature_config(self, df: pd.DataFrame) -> dict:
        """
        Generate feature configuration for Qlib DataHandler.

        Returns dict with feature categories.
        """
        all_cols = df.columns.tolist()

        feature_groups = {
            'D_features': [c for c in all_cols if c.startswith('D') and c[1:].isdigit()],
            'E_features': [c for c in all_cols if c.startswith('E') and c[1:].isdigit()],
            'I_features': [c for c in all_cols if c.startswith('I') and c[1:].isdigit()],
            'M_features': [c for c in all_cols if c.startswith('M') and c[1:].isdigit()],
            'P_features': [c for c in all_cols if c.startswith('P') and c[1:].isdigit()],
            'S_features': [c for c in all_cols if c.startswith('S') and c[1:].isdigit()],
            'V_features': [c for c in all_cols if c.startswith('V') and c[1:].isdigit()],
        }

        # Target variables
        targets = [c for c in all_cols if 'return' in c.lower() or 'rate' in c.lower()]

        config = {
            'features': feature_groups,
            'targets': targets,
            'all_features': sum(feature_groups.values(), [])
        }

        return config


def main():
    """Example usage."""
    converter = HullTacticalDataConverter(
        data_path="../data/raw/train.csv",
        output_path="../data/processed"
    )

    # Convert
    df_qlib = converter.convert(start_date="1990-01-01")

    # Save for Qlib
    output_file = converter.save_for_qlib_dump(df_qlib)

    # Get feature config
    config = converter.get_feature_config(df_qlib)
    print(f"\nFeature groups found:")
    for group, features in config['features'].items():
        print(f"  {group}: {len(features)} features")
    print(f"  Targets: {config['targets']}")


if __name__ == "__main__":
    main()
