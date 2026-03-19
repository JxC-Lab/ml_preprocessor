# 🧹 ml_preprocessor

**A configurable, modular data preprocessing framework for Machine Learning pipelines.**

Define your entire preprocessing workflow in a single YAML file and run it from the command line — no code changes required between datasets.

---

## ✨ Features

| Module | Capabilities |
|---|---|
| `MissingValueHandler` | mean, median, most_frequent, constant, drop_rows, drop_cols |
| `CategoricalEncoder` | One-Hot, Label, Ordinal, Frequency, Target Encoding |
| `FeatureScaler` | Standard, MinMax, Robust, MaxAbs, Log, Quantile |
| `FeatureEngineer` | Interactions, Binning, Date extraction, Group aggregations |

- **CLI-first**: drive everything from `config.yaml`
- **Optional sklearn integration**: wrap any transformer with `.to_sklearn()`
- **Train/test split aware**: fit on train, transform test separately
- **Pipeline persistence**: save/load fitted pipelines with `.pkl`
- **Supports**: CSV, TSV, Parquet, JSON, Excel

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/ml-preprocessor.git
cd ml-preprocessor
pip install -r requirements.txt
```

---

## ⚡ Quick Start

### 1. Define your pipeline in YAML

```yaml
# config.yaml
pipeline:
  - step: missing
    columns: [age, salary]
    strategy: median

  - step: encoding
    columns: [city]
    method: onehot

  - step: scaling
    method: standard
```

### 2. Run from CLI

```bash
# Fit + transform train data
python -m ml_preprocessor run \
  --config config.yaml \
  --input data/train.csv \
  --output data/train_clean.csv

# Fit on train, transform test separately
python -m ml_preprocessor run \
  --config config.yaml \
  --input data/train.csv \
  --test-input data/test.csv \
  --output data/train_clean.csv \
  --save-pipeline pipeline.pkl
```

### 3. Or use the Python API

```python
from ml_preprocessor import PreprocessingPipeline, MissingValueHandler, FeatureScaler

pipeline = PreprocessingPipeline([
    MissingValueHandler(columns=["age", "salary"], strategy="median"),
    FeatureScaler(method="standard"),
])

df_train_clean = pipeline.fit_transform(df_train)
df_test_clean  = pipeline.transform(df_test)

pipeline.save("pipeline.pkl")
```

---

## 🖥️ CLI Reference

```
python -m ml_preprocessor <command> [options]

Commands:
  run        Run pipeline on data
  validate   Validate a config file without running it
  inspect    Show pipeline summary from config
  schema     Print full config schema reference
```

```bash
# Validate your config before running
python -m ml_preprocessor validate --config config.yaml

# Inspect pipeline structure
python -m ml_preprocessor inspect --config config.yaml

# Print config schema / reference
python -m ml_preprocessor schema
```

---

## 📄 Config Schema

```yaml
pipeline:
  - step: missing
    columns: [col1, col2]           # optional; null = all columns
    strategy: mean                  # mean | median | most_frequent | constant | drop_rows | drop_cols
    fill_value: 0                   # required if strategy=constant
    drop_threshold: 0.5             # for drop_cols

  - step: encoding
    columns: [col1]
    method: onehot                  # onehot | label | ordinal | frequency | target
    drop_first: true                # for onehot
    ordinal_order:                  # required for ordinal
      col1: [low, mid, high]

  - step: scaling
    columns: [col1, col2]
    method: standard                # standard | minmax | robust | maxabs | log | quantile
    feature_range: [0, 1]           # for minmax

  - step: features
    interactions:
      - [col_a, col_b]              # creates col_a * col_b
    binning:
      - column: age
        bins: [0, 18, 35, 60, 100]
        labels: [minor, young, adult, senior]
        drop_original: false
    dates:
      - column: created_at
        extract: [year, month, dayofweek, is_weekend]
        drop_original: true
    aggregations:
      - group_by: category
        agg_col: price
        func: [mean, std]
```

---

## 🔗 Optional sklearn integration

```python
from ml_preprocessor import FeatureScaler

scaler = FeatureScaler(method="robust")
sklearn_scaler = scaler.to_sklearn()  # returns sklearn-compatible transformer

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

sklearn_pipeline = Pipeline([
    ("scaler", sklearn_scaler),
    ("model", LogisticRegression()),
])
```

---

## 📁 Project Structure

```
ml_preprocessor/
├── __init__.py
├── __main__.py          # python -m ml_preprocessor entry point
├── cli.py               # CLI commands (run, validate, inspect, schema)
├── pipeline.py          # PreprocessingPipeline orchestrator
├── config.py            # YAML/JSON config loader & pipeline builder
├── transformers/
│   ├── base.py          # BaseTransformer + sklearn wrapper
│   ├── missing.py       # MissingValueHandler
│   ├── encoding.py      # CategoricalEncoder
│   ├── scaling.py       # FeatureScaler
│   └── features.py      # FeatureEngineer
└── utils/
    └── logger.py
```

---

## 📦 Requirements

```
pandas>=1.5
numpy>=1.23
pyyaml>=6.0
openpyxl>=3.0        # for Excel support
pyarrow>=10.0        # for Parquet support
scikit-learn>=1.2    # optional, only needed for .to_sklearn()
scipy>=1.10          # optional, only needed for quantile scaling with normal output
```

---

## 📝 License

MIT
