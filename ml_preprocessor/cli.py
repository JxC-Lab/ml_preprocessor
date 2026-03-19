"""
ml_preprocessor CLI — entry point.

Commands:
  run      Run a pipeline from a config file on input data
  validate Validate a config file without running it
  inspect  Show the pipeline summary from a config
  schema   Print the config schema / reference

Examples:
  python -m ml_preprocessor run --config config.yaml --input data.csv --output out.csv
  python -m ml_preprocessor run --config config.yaml --input data.csv --output out.csv --save-pipeline pipeline.pkl
  python -m ml_preprocessor run --config config.yaml --input data.csv --test-input test.csv --output out.csv
  python -m ml_preprocessor validate --config config.yaml
  python -m ml_preprocessor inspect --config config.yaml
  python -m ml_preprocessor schema
"""

import argparse
import sys
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    loaders = {
        ".csv":     lambda: pd.read_csv(p),
        ".tsv":     lambda: pd.read_csv(p, sep="\t"),
        ".parquet": lambda: pd.read_parquet(p),
        ".json":    lambda: pd.read_json(p),
        ".xlsx":    lambda: pd.read_excel(p),
        ".xls":     lambda: pd.read_excel(p),
    }
    loader = loaders.get(p.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file format: {p.suffix}")
    return loader()


def _save_dataframe(df: pd.DataFrame, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    writers = {
        ".csv":     lambda: df.to_csv(p, index=False),
        ".parquet": lambda: df.to_parquet(p, index=False),
        ".json":    lambda: df.to_json(p, orient="records", indent=2),
        ".xlsx":    lambda: df.to_excel(p, index=False),
    }
    writer = writers.get(p.suffix.lower())
    if writer is None:
        raise ValueError(f"Unsupported output format: {p.suffix}")
    writer()


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand handlers
# ─────────────────────────────────────────────────────────────────────────────

def cmd_run(args):
    from ml_preprocessor.pipeline import PreprocessingPipeline

    print(f"▶  Loading data from: {args.input}")
    df_train = _load_dataframe(args.input)
    print(f"   Shape: {df_train.shape}")

    print(f"▶  Building pipeline from: {args.config}")
    pipeline = PreprocessingPipeline.from_config(args.config)
    print(pipeline.summary())

    if args.test_input:
        # Fit on train, transform both
        print(f"\n▶  Fitting on train data...")
        df_train_out = pipeline.fit_transform(df_train)

        print(f"▶  Transforming test data: {args.test_input}")
        df_test = _load_dataframe(args.test_input)
        df_test_out = pipeline.transform(df_test)

        test_out_path = args.output.replace(".", "_test.")
        _save_dataframe(df_test_out, test_out_path)
        print(f"✅  Test output saved → {test_out_path} (shape={df_test_out.shape})")
    else:
        print(f"\n▶  Running fit_transform...")
        df_train_out = pipeline.fit_transform(df_train)

    _save_dataframe(df_train_out, args.output)
    print(f"✅  Output saved → {args.output} (shape={df_train_out.shape})")

    if args.save_pipeline:
        pipeline.save(args.save_pipeline)
        print(f"✅  Pipeline saved → {args.save_pipeline}")

    if args.report:
        from ml_preprocessor.reporter import generate_report
        out = generate_report(df_train, df_train_out, output_path=args.report, title=Path(args.input).stem)
        print(f"✅  HTML report saved → {out}")


def cmd_validate(args):
    from ml_preprocessor.config import load_config, build_pipeline_from_config

    print(f"▶  Validating config: {args.config}")
    try:
        cfg = load_config(args.config)
        steps = build_pipeline_from_config(cfg)
        print(f"✅  Config is valid — {len(steps)} step(s) defined.")
        for i, s in enumerate(steps):
            print(f"   {i+1}. {s.__class__.__name__}")
    except Exception as e:
        print(f"❌  Config error: {e}")
        sys.exit(1)


def cmd_inspect(args):
    from ml_preprocessor.pipeline import PreprocessingPipeline

    pipeline = PreprocessingPipeline.from_config(args.config)
    print(pipeline.summary())


def cmd_schema(_args):
    schema = """
╔══════════════════════════════════════════════════════════╗
║          ml_preprocessor — Config Schema Reference       ║
╠══════════════════════════════════════════════════════════╣

pipeline:                   # list of steps, executed in order
  - step: missing           # ── MissingValueHandler ──────────
    columns: [col1, col2]   #  optional; null = all columns
    strategy: mean          #  mean | median | most_frequent |
                            #  constant | drop_rows | drop_cols
    fill_value: 0           #  required if strategy=constant
    drop_threshold: 0.5     #  for drop_cols

  - step: encoding          # ── CategoricalEncoder ───────────
    columns: [col1]
    method: onehot          #  onehot | label | ordinal |
                            #  frequency | target
    drop_first: true        #  for onehot
    ordinal_order:          #  for ordinal
      col1: [low, mid, high]

  - step: scaling           # ── FeatureScaler ────────────────
    columns: [col1, col2]
    method: standard        #  standard | minmax | robust |
                            #  maxabs | log | quantile
    feature_range: [0, 1]   #  for minmax

  - step: features          # ── FeatureEngineer ──────────────
    interactions:
      - [col_a, col_b]      #  creates col_a * col_b
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
        prefix: price

╚══════════════════════════════════════════════════════════╝
"""
    print(schema)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml_preprocessor",
        description="Configurable ML preprocessing framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Run pipeline on data")
    p_run.add_argument("--config",        required=True,  help="Path to YAML/JSON config")
    p_run.add_argument("--input",         required=True,  help="Path to input data file")
    p_run.add_argument("--output",        required=True,  help="Path to output data file")
    p_run.add_argument("--test-input",    default=None,   help="Optional separate test set (fit on train only)")
    p_run.add_argument("--save-pipeline", default=None,   help="Save fitted pipeline to .pkl")
    p_run.add_argument("--report",        default=None,   help="Generate HTML preprocessing report")

    # validate
    p_val = sub.add_parser("validate", help="Validate a config file")
    p_val.add_argument("--config", required=True, help="Path to YAML/JSON config")

    # inspect
    p_ins = sub.add_parser("inspect", help="Show pipeline summary from config")
    p_ins.add_argument("--config", required=True, help="Path to YAML/JSON config")

    # schema
    sub.add_parser("schema", help="Print config schema reference")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "run":      cmd_run,
        "validate": cmd_validate,
        "inspect":  cmd_inspect,
        "schema":   cmd_schema,
    }
    try:
        handlers[args.command](args)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌  Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
