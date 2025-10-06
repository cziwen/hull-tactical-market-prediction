# FeatureFactory

基于Qlib的Hull Tactical市场预测因子工程系统

## 概述

FeatureFactory利用[微软Qlib](https://github.com/microsoft/qlib)(AI驱动的量化投资平台)从Hull Tactical市场数据中生成和评估alpha因子。

**核心功能:**
- 🏭 **因子生成**: 使用Qlib表达式引擎生成300+候选因子
- 📊 **因子评估**: 基于IC/ICIR指标评估因子质量
- 🔧 **自定义DataHandler**: 扩展Qlib以支持单资产时间序列
- 📈 **特征选择**: 自动选择高质量、低相关性的特征

## 项目结构

```
FeatureFactory/
├── environment.yml              # Conda环境配置
├── data/
│   ├── raw/                    # 原始数据的软链接(train.csv/test.csv)
│   ├── qlib_format/            # Qlib二进制格式(生成后)
│   └── processed/              # 处理后的CSV文件
├── configs/
│   ├── data_config.yaml        # 数据处理配置
│   └── factor_config.yaml      # 因子生成配置
├── notebooks/
│   ├── 01_data_preparation.ipynb       # 数据准备:CSV转Qlib格式
│   └── 02_factor_exploration.ipynb     # 因子生成与评估
├── src/
│   ├── data_converter.py       # CSV到Qlib格式转换器
│   ├── custom_datahandler.py   # 自定义Qlib DataHandler
│   ├── factor_generator.py     # 因子生成引擎
│   └── feature_evaluator.py    # 因子评估器(IC/ICIR)
└── outputs/
    ├── features/               # 导出的特征文件
    └── reports/                # 评估报告
```

## 快速开始

### 1. 环境搭建

```bash
cd FeatureFactory

# 创建conda环境
conda env create -f environment.yml
conda activate qlib_env

# 验证安装
python test_setup.py
```

### 2. 数据准备

运行数据准备notebook:

```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```

这个notebook会:
- 加载原始CSV数据
- 将`date_id`转换为datetime格式
- 添加`instrument`列(SP500)
- 分析数据质量
- 保存为Qlib兼容格式

**输出:** `data/processed/SP500.csv`

### 3. 因子工程

运行因子探索notebook:

```bash
jupyter notebook notebooks/02_factor_exploration.ipynb
```

这个notebook会:
- 使用Qlib表达式生成300+候选因子
- 从基础特征计算衍生因子
- 使用IC/ICIR指标评估因子质量
- 选择高性能、低相关性的因子
- 导出特征用于模型训练

**输出:**
- `outputs/factor_library.csv` - 完整因子目录
- `outputs/features/selected_features.csv` - 精选特征
- `outputs/reports/factor_evaluation_report.csv` - 评估指标

### 4. 在模型训练中使用特征

在主训练notebook中导入精选特征:

```python
import pandas as pd

# 加载工程化特征
df_features = pd.read_csv('FeatureFactory/outputs/features/selected_features.csv')

# 加载特征列表
with open('FeatureFactory/outputs/features/feature_list.txt') as f:
    feature_cols = [line.strip() for line in f]

# 用于模型训练
X = df_features[feature_cols]
y = df_features['market_forward_excess_returns']
```

## 因子类型

FeatureFactory生成8大类因子:

| 类别 | 描述 | 示例 | 数量 |
|----------|-------------|----------|-------|
| **滞后特征** | 历史值 | `Ref($M1, 5)` - M1的5天前值 | 36 |
| **滚动统计** | 窗口统计量 | `Mean($V1, 20)` - V1的20天均值 | 80 |
| **动量因子** | 变化率 | `($P1 - Ref($P1, 20)) / Ref($P1, 20)` - 20天收益率 | 24 |
| **波动率因子** | 离散度指标 | `Std($M1, 20) / Mean($M1, 20)` - 变异系数 | 24 |
| **Z-score** | 标准化值 | `($M1 - Mean($M1, 60)) / Std($M1, 60)` - 60天z-score | 24 |
| **比率因子** | 跨特征比率 | `$M1 / $M2` - 市场特征比率 | 16 |
| **技术指标** | 技术分析指标 | `(EMA($M1, 12) - EMA($M1, 26)) / EMA($M1, 26)` - MACD风格 | 9 |
| **分位数** | 百分位数 | `Quantile($M1, 20, 0.75)` - 20天75分位数 | 18 |

**总计:** 231个基础因子(可自定义组合生成更多)

## Qlib表达式引擎

Qlib的表达式引擎允许用数学公式创建因子:

### 支持的算子

| 算子 | 描述 | 示例 |
|----------|-------------|---------|
| `Ref($feature, N)` | 滞后N期 | `Ref($close, 1)` - 前一天收盘价 |
| `Mean($feature, N)` | N期均值 | `Mean($M1, 20)` - 20天均值 |
| `Std($feature, N)` | N期标准差 | `Std($V1, 10)` - 10天标准差 |
| `Max/Min($feature, N)` | N期最大/最小值 | `Max($P1, 5)` - 5天最大值 |
| `Delta($feature, N)` | N期差分 | `Delta($M1, 1)` - 日变化 |
| `EMA($feature, N)` | 指数移动平均 | `EMA($M1, 12)` - 12天EMA |
| `Quantile($feature, N, q)` | N期分位数 | `Quantile($M1, 20, 0.25)` - 20天25分位数 |
| `Rank($feature)` | 横截面排名 | `Rank($close)` - 排名 |

### 因子示例

```python
# 动量因子
"($M1 - Ref($M1, 20)) / Ref($M1, 20)"  # 20天收益率

# MACD风格指标
"(EMA($M1, 12) - EMA($M1, 26)) / EMA($M1, 26)"  # 快慢线差

# 布林带位置
"($M1 - Mean($M1, 20)) / (2 * Std($M1, 20))"  # 相对布林带位置

# Z-score标准化
"($V1 - Mean($V1, 60)) / Std($V1, 60)"  # 60天z-score

# 特征比率
"$M1 / $M2"  # 市场特征M1与M2的比率
```

## 评估指标

### IC (信息系数)

因子值与未来收益的Spearman/Pearson相关系数:

```python
IC = correlation(factor[t], target[t+1])
```

**解读:**
- `|IC| > 0.05`: 强因子
- `|IC| > 0.02`: 中等因子
- `|IC| < 0.02`: 弱因子

### ICIR (IC信息比率)

衡量因子表现的一致性:

```python
ICIR = mean(rolling_IC) / std(rolling_IC)
```

**解读:**
- `ICIR > 1`: 高度一致
- `ICIR > 0.5`: 中度一致
- `ICIR < 0`: 不稳定

### 自相关性

衡量因子随时间的稳定性:

```python
autocorr = correlation(factor[t], factor[t-1])
```

高自相关性表明因子值稳定。

## 自定义配置

### 添加自定义因子

编辑 `src/factor_generator.py`:

```python
def generate_custom_factors(self):
    """添加你的自定义因子逻辑"""
    factors = []

    # 示例:自定义比率
    factors.append(("MY_RATIO", "$E1 / $I1"))

    # 示例:复杂指标
    factors.append(("MY_INDICATOR",
                   "(Mean($M1, 10) - Mean($M1, 30)) / Std($M1, 20)"))

    return factors
```

### 修改DataHandler

编辑 `src/custom_datahandler.py` 以:
- 改变特征预处理方式
- 添加自定义处理器
- 修改特征表达式

### 调整评估标准

编辑 `notebooks/02_factor_exploration.ipynb`:

```python
# 改变IC阈值
top_factors = df_eval[df_eval['abs_ic'] > 0.03]

# 改变相关性阈值
selected = evaluator.select_low_correlation_features(
    df, factors, threshold=0.7  # 更低=更严格
)
```

### 修改配置文件

编辑 `configs/factor_config.yaml`:

```yaml
# 调整滞后期
lag:
  periods: [1, 3, 5, 10, 20]  # 自定义滞后期

# 调整窗口大小
rolling:
  windows: [10, 20, 40, 80]  # 自定义窗口

# 调整选择标准
evaluation:
  min_abs_ic: 0.03    # 提高IC阈值
  top_n: 30           # 选择Top 30因子
  correlation_threshold: 0.7  # 降低相关性阈值
```

## 高级用法

### 使用Qlib DataHandler(完整集成)

生产环境使用Qlib完整流程:

```python
import qlib
from qlib.data.dataset import DatasetH
from src.custom_datahandler import HullTacticalHandlerV2

# 初始化Qlib
qlib.init(provider_uri='~/.qlib/qlib_data/hull_tactical')

# 创建DataHandler
handler = HullTacticalHandlerV2(
    instruments='SP500',
    start_time='2000-01-01',
    end_time='2020-12-31',
)

# 获取特征
df_features = handler.fetch()
```

### 批量因子评估

高效评估1000+因子:

```python
from src.factor_generator import FactorGenerator
from src.feature_evaluator import FeatureEvaluator

generator = FactorGenerator()
all_factors = generator.generate_all_factors()

# 展平因子列表
factor_list = []
for category, factors in all_factors.items():
    factor_list.extend([expr for name, expr in factors])

# 评估
evaluator = FeatureEvaluator()
results = evaluator.evaluate_factor_batch(df, factor_list, top_n=100)
```

## 性能优化技巧

1. **缺失值处理**: 早期数据缺失严重,建议:
   - 只使用近期数据(`date_id > 5000`)
   - 使用`Fillna`处理器前向填充
   - 使用稳健统计量(中位数、MAD)

2. **因子计算**: 对于大规模因子集:
   - 分批计算
   - 缓存中间结果
   - 使用Polars加速处理

3. **特征选择**:
   - 从高|IC|因子开始(top 50-100)
   - 剔除相关性 > 0.8的因子
   - 在模型中使用L1正则化

## 与主项目集成

### 方案1: 直接导入特征

```python
# 在主训练notebook中
df_features = pd.read_csv('FeatureFactory/outputs/features/selected_features.csv')
X_train = df_features[feature_cols]
```

### 方案2: 实时计算

```python
# 在submission.ipynb中
from FeatureFactory.src.factor_generator import FactorGenerator

def predict(test: pl.DataFrame) -> float:
    # 实时计算因子
    # (需要实现不依赖Qlib的因子计算)
    pass
```

## 常见问题

### Qlib安装失败

```bash
# 方法1: 使用--no-build-isolation
pip install pyqlib --no-build-isolation

# 方法2: 从源码安装
git clone https://github.com/microsoft/qlib.git
cd qlib
pip install -e .
```

### 内存问题

对于大规模因子集:

```python
# 分块处理
chunk_size = 50
for i in range(0, len(factor_cols), chunk_size):
    chunk = factor_cols[i:i+chunk_size]
    results = evaluator.evaluate_factor_batch(df, chunk)
```

### 日期转换问题

确保日期格式一致:

```python
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
```

### 数据稀疏性问题

早期数据(date_id < 5000)缺失严重:

```python
# 方案1: 只使用近期数据
df_filtered = df[df['date_id'] > 5000]

# 方案2: 在配置中设置
# configs/data_config.yaml
quality:
  use_date_id_from: 5000
```

## 参考资料

- [Qlib官方文档](https://qlib.readthedocs.io/)
- [Qlib GitHub仓库](https://github.com/microsoft/qlib)
- [Alpha158因子库](https://qlib.readthedocs.io/en/latest/component/data.html#alpha158)
- [Qlib论文](https://arxiv.org/abs/2009.11189)
- [因子挖掘最佳实践](https://qlib.readthedocs.io/en/latest/advanced/alpha.html)

## 工作流程示例

### 完整因子工程流程

```bash
# 1. 激活环境
conda activate qlib_env

# 2. 数据准备
jupyter notebook notebooks/01_data_preparation.ipynb
# 输出: data/processed/SP500.csv

# 3. 因子生成与评估
jupyter notebook notebooks/02_factor_exploration.ipynb
# 输出: outputs/features/selected_features.csv

# 4. 集成到训练
# 在主项目的训练notebook中导入特征

# 5. 模型训练与验证
# 使用工程化特征训练LightGBM/XGBoost

# 6. 提交
# 在submission.ipynb中使用训练好的模型
```

## 项目特点

### ✅ 优点

1. **表达式引擎**: 用公式而非代码定义因子,快速迭代
2. **批量生成**: 一键生成300+候选因子
3. **自动评估**: IC/ICIR指标自动计算排序
4. **去相关**: 自动剔除高相关冗余因子
5. **独立环境**: 不影响主项目,专注因子研究
6. **可扩展**: 易于添加自定义因子类型

### ⚠️ 注意事项

1. **Qlib是可选的**: 即使不安装Qlib,基础功能也能运行
2. **单资产适配**: Qlib主要为多股票设计,已适配单资产场景
3. **数据稀疏**: 早期数据缺失严重,建议过滤
4. **计算成本**: 300+因子需要一定计算时间和内存

## 贡献指南

添加新功能:

1. 在`src/factor_generator.py`中创建自定义因子生成器
2. 在`src/feature_evaluator.py`中添加评估逻辑
3. 在notebooks中添加文档说明
4. 导出精选特征到`outputs/features/`

## 版本历史

- **v0.1.0** (2025-10-06): 初始版本
  - 8种因子类型
  - IC/ICIR评估
  - 自定义DataHandler
  - 完整文档和示例

## 许可证

本项目使用Qlib(MIT许可证)。详见Qlib仓库。

## 致谢

- [Microsoft Qlib](https://github.com/microsoft/qlib) - 核心量化框架
- [Hull Tactical](https://www.hulltactical.com/) - 数据来源

---

## 下一步

1. ✅ 环境搭建
2. ✅ 运行数据准备notebook
3. ✅ 运行因子探索notebook
4. 📊 集成特征到模型训练
5. 🚀 在submission.ipynb中部署

**准备开始?**

```bash
cd FeatureFactory
conda activate qlib_env
python test_setup.py
jupyter notebook
```

祝你因子挖掘顺利! 🚀📈
