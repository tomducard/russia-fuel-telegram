"""Command-line interface for rft toolkit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from . import features as feat
from . import keywords
from . import model
from . import official
from . import scraping

load_dotenv()


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet"}:
        return pd.read_parquet(path)
    if suffix in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension for {path}")


def run_scrape(args: argparse.Namespace) -> None:
    channels = scraping.load_channels_csv(args.channels)
    scraper = scraping.TelegramScraper(session_name=args.session)
    
    # Handle limit
    limit_val = None if args.limit <= 0 else args.limit
    
    # Handle min_date
    min_date = None
    if args.since:
        from datetime import datetime, timezone
        # Parse YYYY-MM-DD
        dt = datetime.strptime(args.since, "%Y-%m-%d")
        min_date = dt.replace(tzinfo=timezone.utc)
        
    output = scraper.scrape_to_parquet(channels=channels, output_path=args.output, limit=limit_val, min_date=min_date)
    print(f"Wrote raw messages to {output}")


def run_features(args: argparse.Namespace) -> None:
    raw_df = _load_table(Path(args.raw))
    kw_groups = keywords.load_keyword_groups(args.keywords)
    daily = feat.build_daily_features(raw_df, kw_groups)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_path, index=False)
    print(f"Wrote daily features to {output_path}")


def run_merge(args: argparse.Namespace) -> None:
    features_df = _load_table(Path(args.features))
    official_df = official.load_official_csv(args.official_csv)
    merged = official.merge_with_official(features_df, official_df, how="left")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    print(f"Wrote merged dataset to {output_path}")


def run_train(args: argparse.Namespace) -> None:
    data = _load_table(Path(args.data))
    
    if args.mode == "classification":
        print(f"Training Classifier ({args.model_type}) on target: {args.target}")
        
        if args.model_type == "lstm":
            result = model.train_lstm(
                data,
                target_col=args.target,
                train_ratio=args.train_ratio
            )
        else:
            result = model.train_classifier(
                data, 
                target_col=args.target, 
                train_ratio=args.train_ratio,
                model_type=args.model_type
            )
            
        print(f"Accuracy: {result.accuracy:.2%}")
        print(f"ROC-AUC : {result.roc_auc:.4f} (Ranking Ability)")
        print(f"F1-Score: {result.f1_score:.4f} (Balance Precision/Recall)")
        print("\nClassification Report:")
        print(result.report)
    else:
        print(f"Training Regressor on target: {args.target}")
        result = model.train_baseline(data, target_col=args.target, train_ratio=args.train_ratio)
        print(f"Train MSE: {result.train_mse:.4f} | Test MSE: {result.test_mse:.4f}")
    
    # Extract coefficients / importances
    pipeline = result.model
    step = pipeline.named_steps["model"]
    
    # Handle different model types for feature importance
    if hasattr(step, "coef_"):
        importances = step.coef_[0] if len(step.coef_.shape) > 1 else step.coef_
        imp_name = "Coefficients"
    elif hasattr(step, "feature_importances_"):
        importances = step.feature_importances_
        imp_name = "Feature Importances"
    else:
        print("Model does not expose feature importance.")
        return
    
    # Create a nice list sorted by importance (abs value)
    feature_imp = list(zip(result.feature_cols, importances))
    feature_imp.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n{imp_name}:")
    print("-" * 40)
    for name, val in feature_imp:
         print(f"{name:<30} : {val:.4f}")
    print("-" * 40)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rft", description="Russia fuel Telegram toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scrape_p = subparsers.add_parser("scrape", help="Scrape Telegram channels to Parquet")
    scrape_p.add_argument("--channels", default="channels/channels_seed.csv", help="CSV with channel column")
    scrape_p.add_argument("--output", default="data/raw/messages.parquet", help="Output parquet path")
    scrape_p.add_argument("--limit", type=int, default=500, help="Max messages per channel (0 for unlimited)")
    scrape_p.add_argument("--since", help="Start date (YYYY-MM-DD) for deep scraping")
    scrape_p.add_argument("--session", default="rft_session", help="Telethon session name")
    scrape_p.set_defaults(func=run_scrape)

    features_p = subparsers.add_parser("features", help="Build daily features from raw messages")
    features_p.add_argument("--raw", default="data/raw/messages.parquet", help="Raw messages parquet")
    features_p.add_argument("--keywords", default="channels/keywords_ru.yaml", help="YAML keyword file")
    features_p.add_argument("--output", default="data/interim/daily_features.parquet", help="Output parquet")
    features_p.set_defaults(func=run_features)

    merge_p = subparsers.add_parser("merge-official", help="Merge features with official CSV")
    merge_p.add_argument("--features", default="data/interim/daily_features.parquet", help="Features parquet")
    merge_p.add_argument("--official-csv", required=True, help="Official CSV with date column")
    merge_p.add_argument("--output", default="data/processed/merged.parquet", help="Output merged parquet")
    merge_p.set_defaults(func=run_merge)

    train_p = subparsers.add_parser("train", help="Train model (Regression or Classification)")
    train_p.add_argument("--data", default="data/processed/merged.parquet", help="Merged dataset path")
    train_p.add_argument("--target", default="official_metric", help="Target column name")
    train_p.add_argument("--mode", choices=["regression", "classification"], default="regression", help="Training mode")
    train_p.add_argument(
        "--model-type",
        choices=["rf", "xgb", "gb", "mlp", "lstm"],
        default="rf",
        help="Classifier type (RandomForest, XGBoost, GradientBoosting, MLP, LSTM)",
    )
    train_p.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    train_p.set_defaults(func=run_train)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
