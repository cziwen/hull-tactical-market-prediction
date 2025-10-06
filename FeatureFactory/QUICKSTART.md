# FeatureFactory Quick Start Guide

快速上手指南 - 10分钟开始因子工程

## 📋 前置要求

- Conda或Miniconda已安装
- Hull Tactical数据文件([train.csv](../train.csv), [test.csv](../test.csv))已存在

## 🚀 5步开始

### 步骤 1: 创建环境

```bash
cd FeatureFactory

# 创建conda环境(首次运行,约2-3分钟)
conda env create -f environment.yml

# 激活环境
conda activate qlib_env
```

### 步骤 2: 验证安装

```bash
# 运行测试脚本
python test_setup.py
```

预期输出:
```
✓ Python version OK
✓ pandas
✓ numpy
...
Setup Test Complete!
```

### 步骤 3: 数据准备

打开Jupyter:
```bash
jupyter notebook
```

运行 `notebooks/01_data_preparation.ipynb`:
- 加载CSV数据
- 转换为Qlib格式
- 分析数据质量
- **输出**: `data/processed/SP500.csv`

**时间**: ~2分钟

### 步骤 4: 因子工程

运行 `notebooks/02_factor_exploration.ipynb`:
- 生成300+候选因子
- 计算IC/ICIR指标
- 选择Top因子
- **输出**: `outputs/features/selected_features.csv`

**时间**: ~5-10分钟(取决于数据量)

### 步骤 5: 集成到模型

在主训练notebook中:
```python
import pandas as pd

# 加载工程化特征
df_features = pd.read_csv('FeatureFactory/outputs/features/selected_features.csv')

# 加载特征列表
with open('FeatureFactory/outputs/features/feature_list.txt') as f:
    feature_cols = [line.strip() for line in f]

# 使用特征
X = df_features[feature_cols]
y = df_features['market_forward_excess_returns']
```

## 📊 期望输出

完成后你将获得:

1. **因子库** (`outputs/factor_library.csv`)
   - 300+候选因子定义
   - 包含类别和Qlib表达式

2. **精选特征** (`outputs/features/selected_features.csv`)
   - Top 50高质量因子
   - 低相关性(< 0.8)
   - 时间序列完整

3. **评估报告** (`outputs/reports/factor_evaluation_report.csv`)
   - IC/ICIR指标
   - 自相关性
   - 缺失值比例

## 🎯 常见因子示例

生成的因子包括:

```python
# 滞后特征
LAG_M1_T5     # M1的5天前值

# 滚动统计
ROLL_MEAN_V1_W20   # V1的20天均值
ROLL_STD_M1_W10    # M1的10天标准差

# 动量因子
MOM_P1_T20    # P1的20天动量 (return)

# 波动率因子
VOL_CV_M1_W60      # M1的60天变异系数

# Z-score标准化
ZSCORE_V1_W60      # V1的60天z-score

# 比率因子
RATIO_M1_DIV_M2    # M1/M2比率

# 技术指标
TECH_MACD_M1       # MACD风格指标
TECH_BB_POSITION_V1 # 布林带位置
```

## 🔧 自定义配置

### 修改因子生成参数

编辑 `configs/factor_config.yaml`:

```yaml
# 调整滞后期
lag:
  periods: [1, 3, 5, 10, 20]  # 增加或减少滞后期

# 调整窗口大小
rolling:
  windows: [10, 20, 40, 80]  # 自定义窗口

# 调整选择标准
evaluation:
  min_abs_ic: 0.03    # 提高IC阈值(更严格)
  top_n: 30           # 选择Top 30因子
```

### 添加自定义因子

编辑 `src/factor_generator.py`:

```python
def generate_custom_factors(self):
    factors = []

    # 你的自定义因子逻辑
    factors.append(("MY_FACTOR", "($M1 + $M2) / 2"))

    return factors
```

## ⚠️ 注意事项

### 数据稀疏性

早期数据(date_id < 5000)缺失值较多:
- 建议使用 `date_id > 5000` 的数据
- 或在配置中设置: `use_date_id_from: 5000`

### 内存使用

生成300+因子需要内存:
- 推荐: 8GB+ RAM
- 如内存不足,减少因子数量或分批处理

### 计算时间

因子评估时间取决于:
- 数据量: 8990行 → ~5分钟
- 因子数量: 300因子 → ~10分钟
- 使用Polars可加速

## 🐛 故障排除

### Qlib安装失败

```bash
# 方法1: 使用--no-build-isolation
pip install pyqlib --no-build-isolation

# 方法2: 从源码安装
git clone https://github.com/microsoft/qlib.git
cd qlib
pip install -e .
```

### 数据文件未找到

```bash
# 检查软链接
ls -la data/raw/

# 如果失败,手动创建链接
cd data/raw
ln -s ../../train.csv train.csv
ln -s ../../test.csv test.csv
```

### Jupyter kernel未找到

```bash
# 安装kernel
python -m ipykernel install --user --name=qlib_env

# 在Jupyter中选择 qlib_env kernel
```

## 📚 下一步

1. **实验不同因子组合**
   - 尝试不同的窗口大小
   - 测试不同的统计量
   - 添加领域知识因子

2. **优化因子选择**
   - 调整IC阈值
   - 使用集成学习特征重要性
   - A/B测试不同因子集

3. **生产化部署**
   - 在submission.ipynb中实时计算因子
   - 缓存中间结果
   - 优化计算性能

## 🎓 学习资源

- [Qlib文档](https://qlib.readthedocs.io/)
- [Alpha158因子库](https://qlib.readthedocs.io/en/latest/component/data.html#alpha158)
- [因子挖掘论文](https://arxiv.org/abs/2009.11189)

## 💡 技巧

1. **快速迭代**: 先用少量因子(~50)测试流程
2. **可视化**: 使用notebook可视化IC时间序列
3. **版本控制**: 保存不同版本的因子配置
4. **文档化**: 记录表现好的因子及其业务逻辑

---

**Ready to start?**

```bash
conda activate qlib_env
jupyter notebook
# Open notebooks/01_data_preparation.ipynb
```

祝你因子挖掘顺利! 🚀
