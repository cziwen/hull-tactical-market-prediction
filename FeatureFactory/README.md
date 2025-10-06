# FeatureFactory

Qlib-powered factor engineering for Hull Tactical Market Prediction.

## Overview

FeatureFactory leverages [Microsoft Qlib](https://github.com/microsoft/qlib), an AI-oriented quantitative investment platform, to generate and evaluate alpha factors from Hull Tactical market data.

**Key Features:**
- 🏭 **Factor Generation**: 300+ candidate factors using Qlib's expression engine
- 📊 **Factor Evaluation**: IC/ICIR metrics for factor quality assessment
- 🔧 **Custom DataHandler**: Extends Qlib for single-asset time series
- 📈 **Feature Selection**: Automated selection of high-quality, low-correlation features

## Project Structure

```
FeatureFactory/
├── environment.yml              # Conda environment
├── data/
│   ├── raw/                    # Symbolic links to train.csv/test.csv
│   ├── qlib_format/            # Qlib binary format (generated)
│   └── processed/              # Processed CSV files
├── configs/
│   ├── data_config.yaml        # Data processing config (optional)
│   └── factor_config.yaml      # Factor generation config (optional)
├── notebooks/
│   ├── 01_data_preparation.ipynb       # Convert CSV to Qlib format
│   └── 02_factor_exploration.ipynb     # Generate and evaluate factors
├── src/
│   ├── data_converter.py       # CSV → Qlib format converter
│   ├── custom_datahandler.py   # Qlib DataHandler for Hull Tactical
│   ├── factor_generator.py     # Factor generation engine
│   └── feature_evaluator.py    # Factor evaluation (IC/ICIR)
└── outputs/
    ├── features/               # Exported feature files
    └── reports/                # Evaluation reports
```

## Quick Start

### 1. Setup Environment

```bash
cd FeatureFactory

# Create conda environment
conda env create -f environment.yml
conda activate qlib_env

# Verify installation
python -c "import qlib; print(qlib.__version__)"
```

### 2. Data Preparation

Run the data preparation notebook:

```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```

This will:
- Load raw CSV data
- Convert `date_id` to datetime
- Add `instrument` column (SP500)
- Analyze data quality
- Save in Qlib-compatible format

**Output:** `data/processed/SP500.csv`

### 3. Factor Engineering

Run the factor exploration notebook:

```bash
jupyter notebook notebooks/02_factor_exploration.ipynb
```

This will:
- Generate 300+ candidate factors using Qlib expressions
- Compute factors from base features
- Evaluate factors using IC/ICIR metrics
- Select top-performing, uncorrelated features
- Export features for model training

**Outputs:**
- `outputs/factor_library.csv` - Full factor catalog
- `outputs/features/selected_features.csv` - Selected features
- `outputs/reports/factor_evaluation_report.csv` - Evaluation metrics

### 4. Use Features in Model Training

Import the selected features in your main training notebook:

```python
import pandas as pd

# Load engineered features
df_features = pd.read_csv('FeatureFactory/outputs/features/selected_features.csv')

# Load feature list
with open('FeatureFactory/outputs/features/feature_list.txt') as f:
    feature_cols = [line.strip() for line in f]

# Use in your model
X = df_features[feature_cols]
y = df_features['market_forward_excess_returns']
```

## Factor Categories

FeatureFactory generates 8 types of factors:

| Category | Description | Examples | Count |
|----------|-------------|----------|-------|
| **Lag** | Historical values | `Ref($M1, 5)` - M1 value 5 days ago | 36 |
| **Rolling** | Window statistics | `Mean($V1, 20)` - 20-day mean volatility | 80 |
| **Momentum** | Rate of change | `($P1 - Ref($P1, 20)) / Ref($P1, 20)` | 24 |
| **Volatility** | Dispersion measures | `Std($M1, 20) / Mean($M1, 20)` - Coefficient of variation | 24 |
| **Z-score** | Normalized values | `($M1 - Mean($M1, 60)) / Std($M1, 60)` | 24 |
| **Ratio** | Cross-feature | `$M1 / $M2` - Ratio of market features | 16 |
| **Technical** | TA indicators | `(EMA($M1, 12) - EMA($M1, 26)) / EMA($M1, 26)` - MACD-style | 9 |
| **Quantile** | Percentile-based | `Quantile($M1, 20, 0.75)` - 75th percentile | 18 |

**Total:** 231 base factors (before custom combinations)

## Qlib Expression Engine

Qlib's expression engine allows creating factors using mathematical expressions:

### Supported Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `Ref($feature, N)` | Lag N periods | `Ref($close, 1)` |
| `Mean($feature, N)` | N-period mean | `Mean($M1, 20)` |
| `Std($feature, N)` | N-period std | `Std($V1, 10)` |
| `Max/Min($feature, N)` | N-period max/min | `Max($P1, 5)` |
| `Delta($feature, N)` | Difference over N periods | `Delta($M1, 1)` |
| `EMA($feature, N)` | Exponential MA | `EMA($M1, 12)` |
| `Quantile($feature, N, q)` | N-period quantile | `Quantile($M1, 20, 0.25)` |
| `Rank($feature)` | Cross-sectional rank | `Rank($close)` |

### Example Factors

```python
# Momentum
"($M1 - Ref($M1, 20)) / Ref($M1, 20)"

# MACD-style
"(EMA($M1, 12) - EMA($M1, 26)) / EMA($M1, 26)"

# Bollinger Band position
"($M1 - Mean($M1, 20)) / (2 * Std($M1, 20))"

# Z-score
"($V1 - Mean($V1, 60)) / Std($V1, 60)"

# Ratio
"$M1 / $M2"
```

## Evaluation Metrics

### IC (Information Coefficient)

Spearman/Pearson correlation between factor values and future returns:

```python
IC = correlation(factor[t], target[t+1])
```

**Interpretation:**
- `|IC| > 0.05`: Strong factor
- `|IC| > 0.02`: Moderate factor
- `|IC| < 0.02`: Weak factor

### ICIR (IC Information Ratio)

Measures consistency of factor performance:

```python
ICIR = mean(rolling_IC) / std(rolling_IC)
```

**Interpretation:**
- `ICIR > 1`: Highly consistent
- `ICIR > 0.5`: Moderately consistent
- `ICIR < 0`: Unstable

### Autocorrelation

Measures factor stability over time:

```python
autocorr = correlation(factor[t], factor[t-1])
```

## Customization

### Add Custom Factors

Edit `src/factor_generator.py`:

```python
def generate_custom_factors(self):
    """Add your custom factor logic."""
    factors = []

    # Example: Custom ratio
    factors.append(("MY_RATIO", "$E1 / $I1"))

    # Example: Complex indicator
    factors.append(("MY_INDICATOR",
                   "(Mean($M1, 10) - Mean($M1, 30)) / Std($M1, 20)"))

    return factors
```

### Modify DataHandler

Edit `src/custom_datahandler.py` to:
- Change feature preprocessing
- Add custom processors
- Modify feature expressions

### Adjust Evaluation Criteria

Edit `notebooks/02_factor_exploration.ipynb`:

```python
# Change IC threshold
top_factors = df_eval[df_eval['abs_ic'] > 0.03]

# Change correlation threshold
selected = evaluator.select_low_correlation_features(
    df, factors, threshold=0.7  # Lower = more strict
)
```

## Advanced Usage

### Using Qlib DataHandler (Full Integration)

For production use with Qlib's full pipeline:

```python
import qlib
from qlib.data.dataset import DatasetH
from src.custom_datahandler import HullTacticalHandlerV2

# Initialize Qlib
qlib.init(provider_uri='~/.qlib/qlib_data/hull_tactical')

# Create DataHandler
handler = HullTacticalHandlerV2(
    instruments='SP500',
    start_time='2000-01-01',
    end_time='2020-12-31',
)

# Get features
df_features = handler.fetch()
```

### Batch Factor Evaluation

Evaluate 1000+ factors efficiently:

```python
from src.factor_generator import FactorGenerator
from src.feature_evaluator import FeatureEvaluator

generator = FactorGenerator()
all_factors = generator.generate_all_factors()

# Flatten factor list
factor_list = []
for category, factors in all_factors.items():
    factor_list.extend([expr for name, expr in factors])

# Evaluate
evaluator = FeatureEvaluator()
results = evaluator.evaluate_factor_batch(df, factor_list, top_n=100)
```

## Performance Tips

1. **Missing Values**: Early data has extensive missing values. Consider:
   - Using only recent data (`date_id > 5000`)
   - Forward-fill with `Fillna` processor
   - Using robust statistics (median, MAD)

2. **Factor Computation**: For large factor sets:
   - Compute in batches
   - Cache intermediate results
   - Use Polars for faster processing

3. **Feature Selection**:
   - Start with high |IC| factors (top 50-100)
   - Remove correlations > 0.8
   - Use L1 regularization in models

## Integration with Main Project

### Option 1: Direct Feature Import

```python
# In main training notebook
df_features = pd.read_csv('FeatureFactory/outputs/features/selected_features.csv')
X_train = df_features[feature_cols]
```

### Option 2: On-the-Fly Computation

```python
# In submission.ipynb
from FeatureFactory.src.factor_generator import FactorGenerator

def predict(test: pl.DataFrame) -> float:
    # Compute factors in real-time
    # (Requires implementing factor computation without Qlib)
    pass
```

## Troubleshooting

### Qlib Installation Issues

```bash
# If pyqlib fails, try:
pip install numpy cython
pip install pyqlib --no-build-isolation

# Or build from source:
git clone https://github.com/microsoft/qlib.git
cd qlib
pip install -e .
```

### Memory Issues

For large factor sets:

```python
# Process in chunks
chunk_size = 50
for i in range(0, len(factor_cols), chunk_size):
    chunk = factor_cols[i:i+chunk_size]
    results = evaluator.evaluate_factor_batch(df, chunk)
```

### Date Conversion Issues

Ensure consistent datetime format:

```python
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
```

## References

- [Qlib Documentation](https://qlib.readthedocs.io/)
- [Qlib GitHub](https://github.com/microsoft/qlib)
- [Alpha158 Factor Library](https://qlib.readthedocs.io/en/latest/component/data.html#alpha158)
- [Qlib Paper](https://arxiv.org/abs/2009.11189)

## License

This project uses Qlib under MIT License. See Qlib repository for details.

## Contributing

To add new features:

1. Create custom factor generators in `src/factor_generator.py`
2. Add evaluation logic in `src/feature_evaluator.py`
3. Document in notebooks
4. Export selected features to `outputs/features/`

---

**Next Steps:**
1. ✅ Setup environment
2. ✅ Run data preparation notebook
3. ✅ Run factor exploration notebook
4. 📊 Integrate features into model training
5. 🚀 Deploy with submission.ipynb
