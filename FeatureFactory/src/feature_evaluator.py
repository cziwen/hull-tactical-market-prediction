"""
Feature evaluator for alpha factors.

Evaluates factor quality using:
- IC (Information Coefficient): Correlation with future returns
- ICIR (IC Information Ratio): IC mean / IC std
- Auto-correlation: Factor stability over time
- Feature importance: Via tree-based models
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEvaluator:
    """Evaluate alpha factor quality."""

    def __init__(self, target_col: str = 'market_forward_excess_returns'):
        """
        Args:
            target_col: Name of target variable column
        """
        self.target_col = target_col

    def calculate_ic(
        self,
        factor_values: pd.Series,
        target_values: pd.Series,
        method: str = 'spearman'
    ) -> float:
        """
        Calculate Information Coefficient (IC).

        IC measures the correlation between factor values and future returns.

        Args:
            factor_values: Factor values
            target_values: Target returns (aligned with factors)
            method: 'spearman' or 'pearson'

        Returns:
            IC value (correlation coefficient)
        """
        # Remove NaN values
        valid_mask = ~(factor_values.isna() | target_values.isna())
        factor_clean = factor_values[valid_mask]
        target_clean = target_values[valid_mask]

        if len(factor_clean) < 2:
            return np.nan

        if method == 'spearman':
            ic, _ = spearmanr(factor_clean, target_clean)
        elif method == 'pearson':
            ic, _ = pearsonr(factor_clean, target_clean)
        else:
            raise ValueError(f"Unknown method: {method}")

        return ic

    def calculate_rolling_ic(
        self,
        df: pd.DataFrame,
        factor_col: str,
        window: int = 20,
        method: str = 'spearman'
    ) -> pd.Series:
        """
        Calculate rolling IC over time.

        Args:
            df: DataFrame with factor and target columns
            factor_col: Name of factor column
            window: Rolling window size
            method: 'spearman' or 'pearson'

        Returns:
            Series of rolling IC values
        """
        rolling_ic = []

        for i in range(window, len(df)):
            window_data = df.iloc[i-window:i]
            ic = self.calculate_ic(
                window_data[factor_col],
                window_data[self.target_col],
                method=method
            )
            rolling_ic.append(ic)

        # Pad with NaN for first window
        rolling_ic = [np.nan] * window + rolling_ic

        return pd.Series(rolling_ic, index=df.index)

    def calculate_icir(
        self,
        df: pd.DataFrame,
        factor_col: str,
        window: int = 20,
        method: str = 'spearman'
    ) -> float:
        """
        Calculate IC Information Ratio (ICIR).

        ICIR = mean(IC) / std(IC)
        Measures consistency of factor performance.

        Args:
            df: DataFrame with factor and target columns
            factor_col: Name of factor column
            window: Window for rolling IC calculation
            method: 'spearman' or 'pearson'

        Returns:
            ICIR value
        """
        rolling_ic = self.calculate_rolling_ic(df, factor_col, window, method)
        rolling_ic_clean = rolling_ic.dropna()

        if len(rolling_ic_clean) < 2:
            return np.nan

        ic_mean = rolling_ic_clean.mean()
        ic_std = rolling_ic_clean.std()

        if ic_std == 0:
            return np.nan

        return ic_mean / ic_std

    def calculate_autocorr(
        self,
        factor_values: pd.Series,
        lag: int = 1
    ) -> float:
        """
        Calculate autocorrelation of factor.

        High autocorrelation indicates stable factor values.

        Args:
            factor_values: Factor values
            lag: Lag period

        Returns:
            Autocorrelation coefficient
        """
        factor_clean = factor_values.dropna()

        if len(factor_clean) < lag + 2:
            return np.nan

        return factor_clean.autocorr(lag=lag)

    def evaluate_factor(
        self,
        df: pd.DataFrame,
        factor_col: str,
        ic_window: int = 20,
        ic_method: str = 'spearman'
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of a single factor.

        Args:
            df: DataFrame with factor and target columns
            factor_col: Name of factor column
            ic_window: Window for rolling IC calculation
            ic_method: 'spearman' or 'pearson'

        Returns:
            Dict with evaluation metrics
        """
        metrics = {}

        # Overall IC
        metrics['ic'] = self.calculate_ic(
            df[factor_col],
            df[self.target_col],
            method=ic_method
        )

        # ICIR
        metrics['icir'] = self.calculate_icir(
            df,
            factor_col,
            window=ic_window,
            method=ic_method
        )

        # Autocorrelation
        metrics['autocorr_1'] = self.calculate_autocorr(df[factor_col], lag=1)
        metrics['autocorr_5'] = self.calculate_autocorr(df[factor_col], lag=5)

        # Missing value ratio
        metrics['missing_ratio'] = df[factor_col].isna().mean()

        # Value range
        factor_clean = df[factor_col].dropna()
        if len(factor_clean) > 0:
            metrics['mean'] = factor_clean.mean()
            metrics['std'] = factor_clean.std()
            metrics['min'] = factor_clean.min()
            metrics['max'] = factor_clean.max()
        else:
            metrics['mean'] = np.nan
            metrics['std'] = np.nan
            metrics['min'] = np.nan
            metrics['max'] = np.nan

        return metrics

    def evaluate_factor_batch(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        ic_window: int = 20,
        ic_method: str = 'spearman',
        top_n: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Evaluate multiple factors and rank by quality.

        Args:
            df: DataFrame with factor and target columns
            factor_cols: List of factor column names
            ic_window: Window for rolling IC calculation
            ic_method: 'spearman' or 'pearson'
            top_n: If specified, return only top N factors by |IC|

        Returns:
            DataFrame with evaluation results, sorted by |IC|
        """
        results = []

        for factor_col in factor_cols:
            if factor_col not in df.columns:
                logger.warning(f"Factor {factor_col} not found in DataFrame")
                continue

            metrics = self.evaluate_factor(df, factor_col, ic_window, ic_method)
            metrics['factor'] = factor_col
            results.append(metrics)

        df_results = pd.DataFrame(results)

        # Sort by absolute IC (descending)
        df_results['abs_ic'] = df_results['ic'].abs()
        df_results = df_results.sort_values('abs_ic', ascending=False)

        # Select top N
        if top_n is not None:
            df_results = df_results.head(top_n)

        # Reorder columns
        cols = ['factor', 'ic', 'icir', 'autocorr_1', 'autocorr_5',
                'missing_ratio', 'mean', 'std', 'min', 'max', 'abs_ic']
        df_results = df_results[cols]

        logger.info(f"Evaluated {len(df_results)} factors")

        return df_results

    def calculate_feature_correlation(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        method: str = 'spearman'
    ) -> pd.DataFrame:
        """
        Calculate pairwise correlation between factors.

        Useful for identifying redundant features.

        Args:
            df: DataFrame with factor columns
            factor_cols: List of factor column names
            method: 'spearman' or 'pearson'

        Returns:
            Correlation matrix
        """
        df_factors = df[factor_cols]

        if method == 'spearman':
            corr_matrix = df_factors.corr(method='spearman')
        elif method == 'pearson':
            corr_matrix = df_factors.corr(method='pearson')
        else:
            raise ValueError(f"Unknown method: {method}")

        return corr_matrix

    def select_low_correlation_features(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        threshold: float = 0.8,
        method: str = 'spearman'
    ) -> List[str]:
        """
        Select features with low inter-correlation.

        Iteratively removes features with high correlation.

        Args:
            df: DataFrame with factor columns
            factor_cols: List of factor column names
            threshold: Correlation threshold (0-1)
            method: 'spearman' or 'pearson'

        Returns:
            List of selected feature names
        """
        corr_matrix = self.calculate_feature_correlation(df, factor_cols, method)

        # Find pairs with high correlation
        high_corr_pairs = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append((
                        corr_matrix.index[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))

        # Remove features with high correlation
        # Strategy: Keep feature with higher IC
        removed = set()
        for feat1, feat2, _ in high_corr_pairs:
            if feat1 in removed or feat2 in removed:
                continue

            # Compare IC
            ic1 = abs(self.calculate_ic(df[feat1], df[self.target_col], method))
            ic2 = abs(self.calculate_ic(df[feat2], df[self.target_col], method))

            if ic1 < ic2:
                removed.add(feat1)
            else:
                removed.add(feat2)

        selected = [f for f in factor_cols if f not in removed]

        logger.info(f"Selected {len(selected)}/{len(factor_cols)} features "
                   f"(removed {len(removed)} highly correlated)")

        return selected

    def generate_evaluation_report(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive evaluation report.

        Args:
            df: DataFrame with factors and target
            factor_cols: List of factor column names
            output_path: Path to save report (optional)

        Returns:
            Dict with report data
        """
        logger.info("Generating evaluation report...")

        # Evaluate all factors
        df_eval = self.evaluate_factor_batch(df, factor_cols)

        # Calculate correlation matrix
        corr_matrix = self.calculate_feature_correlation(df, factor_cols)

        # Summary statistics
        summary = {
            'total_factors': len(factor_cols),
            'mean_ic': df_eval['ic'].mean(),
            'median_ic': df_eval['ic'].median(),
            'mean_abs_ic': df_eval['abs_ic'].mean(),
            'max_abs_ic': df_eval['abs_ic'].max(),
            'mean_icir': df_eval['icir'].mean(),
            'positive_ic_ratio': (df_eval['ic'] > 0).mean(),
        }

        report = {
            'summary': summary,
            'factor_evaluation': df_eval,
            'correlation_matrix': corr_matrix,
        }

        # Save if path provided
        if output_path:
            df_eval.to_csv(output_path.replace('.csv', '_factors.csv'), index=False)
            corr_matrix.to_csv(output_path.replace('.csv', '_correlation.csv'))

            # Save summary
            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(output_path.replace('.csv', '_summary.csv'), index=False)

            logger.info(f"Report saved to {output_path}")

        return report


def main():
    """Example usage."""
    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    # Create factors and target
    data = {
        'target': np.random.randn(n_samples),
    }

    # Factor with high IC
    data['good_factor'] = data['target'] + np.random.randn(n_samples) * 0.5

    # Factor with low IC
    data['bad_factor'] = np.random.randn(n_samples)

    # Factor with moderate IC
    data['medium_factor'] = data['target'] * 0.3 + np.random.randn(n_samples) * 0.7

    df = pd.DataFrame(data)

    # Evaluate
    evaluator = FeatureEvaluator(target_col='target')

    factor_cols = ['good_factor', 'bad_factor', 'medium_factor']
    results = evaluator.evaluate_factor_batch(df, factor_cols)

    print("\nFactor Evaluation Results:")
    print(results[['factor', 'ic', 'icir', 'autocorr_1']])

    # Generate report
    report = evaluator.generate_evaluation_report(df, factor_cols)
    print("\nSummary:")
    for key, value in report['summary'].items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
