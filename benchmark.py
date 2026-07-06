"""
benchmark.py
------------
Standalone, reproducible benchmark script. Run this yourself to regenerate
every number quoted in README.md -- nothing in the README is hand-waved,
you can verify it in ~5 seconds on a laptop CPU.

Usage:
    python benchmark.py
"""

import time
import numpy as np

from model import DemandForecaster, PriceElasticityEngine, generate_synthetic_training_data


def main():
    print("=" * 60)
    print("Generating synthetic historical + supervised training data...")
    historical_series, X, y = generate_synthetic_training_data(n_samples=2000)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    split = int(0.8 * len(y))
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    forecaster = DemandForecaster()
    t0 = time.perf_counter()
    forecaster.fit(historical_series, X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"Training time (2000 samples): {train_time*1000:.1f} ms")

    # --- Accuracy metrics on held-out test set (reactive branch) ---
    X_test_scaled = forecaster.scaler.transform(X_test)
    preds = forecaster.gbr.predict(X_test_scaled)

    mae = np.mean(np.abs(preds - y_test))
    rmse = np.sqrt(np.mean((preds - y_test) ** 2))
    ss_res = np.sum((preds - y_test) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    # MASE: MAE of model / MAE of a naive one-step-lag ("tomorrow = today")
    # forecast, computed on the historical time series (this is the
    # standard MASE denominator -- it must come from a genuine time-ordered
    # sequence, not the randomly-shuffled feature rows).
    naive_mae = np.mean(np.abs(np.diff(historical_series)))
    mase = mae / naive_mae

    print("\n--- Accuracy (held-out test split, n=%d) ---" % len(y_test))
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2:  {r2:.4f}")
    print(f"MASE: {mase:.4f}  (< 1.0 means we beat a naive baseline)")

    # --- Latency benchmarks ---
    n_runs = 500

    demand_latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        forecaster.predict(
            sales_velocity=3.2, competitor_price_ratio=1.01, hour_of_day=19,
            day_of_week=5, page_views=25, add_to_carts=6, stock_level=80,
        )
        demand_latencies.append((time.perf_counter() - t0) * 1000)

    pricing_engine = PriceElasticityEngine()
    pricing_latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        pricing_engine.compute_dynamic_price(
            base_price=1299.0, demand_velocity=3.2, intent_ratio=0.24,
            stock_level=80, competitor_price_ratio=1.01,
        )
        pricing_latencies.append((time.perf_counter() - t0) * 1000)

    demand_latencies = np.array(demand_latencies)
    pricing_latencies = np.array(pricing_latencies)

    print(f"\n--- Latency over {n_runs} runs ---")
    print(f"/predict-demand   core inference: mean {demand_latencies.mean():.3f} ms, "
          f"p95 {np.percentile(demand_latencies, 95):.3f} ms")
    print(f"/get-dynamic-price core inference: mean {pricing_latencies.mean():.3f} ms, "
          f"p95 {np.percentile(pricing_latencies, 95):.3f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
