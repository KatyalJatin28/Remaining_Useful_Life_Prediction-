import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from load_data import load_data, add_rul
from features import drop_useless, add_rolling_features, cap_rul
from model import prepare_features, train_and_evaluate
from visualize import (
    plot_rul_comparison,
    plot_degradation_curve,
    plot_rmse_comparison
)
import pandas as pd
import numpy as np


def main():
    print("=" * 55)
    print("NASA C-MAPSS RUL Prediction Pipeline")
    print("=" * 55)

    print("\n[1/5] Loading data...")
    train, test, rul_test = load_data()
    print(f"Training engines:  {train['unit_id'].nunique()}")
    print(f"Training records:  {len(train)}")
    print(f"Test engines:      {test['unit_id'].nunique()}")

    print("\n[2/5] Calculating RUL labels...")
    train = add_rul(train)

    train = cap_rul(train, cap=125)
    print(f"RUL range after capping: {train['RUL'].min()} to {train['RUL'].max()}")

    print("\n[3/5] Engineering features...")
    train = drop_useless(train)
    test = drop_useless(test)
    train = add_rolling_features(train, window=30)
    test = add_rolling_features(test, window=30)
    print(f"Features after engineering: {len(train.columns)}")

    test_last = test.groupby('unit_id').last().reset_index()
    test_last['RUL'] = rul_test['RUL'].values
    test_last['RUL'] = test_last['RUL'].clip(upper=125)

    print("\n[4/5] Training models...")
    X_train, y_train, X_test, feature_cols = prepare_features(
        train, test_last
    )
    y_test = test_last['RUL']
    results = train_and_evaluate(X_train, y_train, X_test, y_test)

    print("\n[5/5] Generating visualizations...")
    os.makedirs('outputs', exist_ok=True)
    plot_rul_comparison(y_test, results)
    plot_degradation_curve(train)
    plot_rmse_comparison(results)

    print("\n" + "=" * 55)
    print("RESULTS SUMMARY")
    print("=" * 55)
    print(f"{'Model':<25} {'RMSE':>8} {'NASA Score':>12}")
    print("-" * 55)
    for name, result in results.items():
        print(f"{name:<25} {result['rmse']:>8.2f} {result['nasa_score']:>12.2f}")

    rf = results['Random Forest']
    print(f"\nBest RF parameters: {rf['best_params']}")
    print(f"Best RF CV RMSE:    {rf['cv_rmse']:.2f}")
    print("\nOutputs saved to outputs/ folder")
    print("=" * 55)


if __name__ == "__main__":
    main()