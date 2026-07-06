"""
model.py
--------
Core mathematical engine for the Dynamic Demand Forecasting & Intent-Based
Pricing system.

Design rationale
=================
Real e-commerce demand has two components that behave very differently:

1. A *smooth, slow-moving baseline* driven by seasonality, day-of-week and
   time-of-day habits (e.g. everyone buys more grocery items at 7pm).
   This is best captured by a classical statistical smoother -- cheap,
   interpretable, and stable even with very little data.

2. A *sharp, non-linear reaction* to live signals -- competitor price
   drops, a sudden click-stream spike, low stock urgency. Tree ensembles
   (Gradient Boosting) are good at this because they can carve out
   interaction effects (e.g. "high add-to-cart rate AND low stock" is a
   different regime than either signal alone) without us hand engineering
   every interaction term.

We therefore build a **two-stage ensemble**:

    final_demand = w * ExponentialSmoothing(history)          (baseline)
                 + (1 - w) * GradientBoostingRegressor(features) (reactive)

`w` is not a magic constant -- it is derived from how "fresh"/volatile the
live signal is, via `_adaptive_blend_weight`. When live click-stream data
is sparse or noisy we trust the statistical baseline more; when we have a
strong, consistent live signal we lean on the ML model.

Everything numeric is implemented directly with NumPy so the mathematics
are auditable line by line (no black-box wrapper hides the formulas).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# 1. Pure-NumPy Exponential Smoothing (Holt's Linear Trend variant)
# --------------------------------------------------------------------------- #
class ExponentialSmoother:
    """
    Double Exponential Smoothing (Holt's method), implemented from first
    principles with NumPy.

    Level update:   L_t = alpha * y_t          + (1 - alpha) * (L_{t-1} + T_{t-1})
    Trend update:   T_t = beta  * (L_t - L_{t-1}) + (1 - beta) * T_{t-1}
    Forecast (h):   F_{t+h} = L_t + h * T_t

    alpha controls how much weight recent observations get over history.
    beta controls how quickly we adapt to a changing trend (e.g. demand
    accelerating right before a flash sale).
    """

    def __init__(self, alpha: float = 0.35, beta: float = 0.15):
        if not (0 < alpha < 1) or not (0 < beta < 1):
            raise ValueError("alpha and beta must lie in (0, 1)")
        self.alpha = alpha
        self.beta = beta
        self.level: Optional[float] = None
        self.trend: Optional[float] = None

    def fit(self, series: np.ndarray) -> "ExponentialSmoother":
        series = np.asarray(series, dtype=np.float64)
        if series.size < 2:
            raise ValueError("Need at least 2 observations to seed level/trend")

        # Seed level with the first observation, trend with the first diff.
        level = series[0]
        trend = series[1] - series[0]

        for y_t in series[1:]:
            prev_level = level
            level = self.alpha * y_t + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * trend

        self.level, self.trend = level, trend
        return self

    def forecast(self, horizon: int = 1) -> np.ndarray:
        if self.level is None:
            raise RuntimeError("Call .fit() before .forecast()")
        h = np.arange(1, horizon + 1)
        return self.level + h * self.trend


# --------------------------------------------------------------------------- #
# 2. Feature engineering for the reactive (Gradient Boosting) branch
# --------------------------------------------------------------------------- #
def build_feature_vector(
    sales_velocity: float,
    competitor_price_ratio: float,
    hour_of_day: int,
    day_of_week: int,
    page_views: int,
    add_to_carts: int,
    stock_level: int,
) -> np.ndarray:
    """
    Turns raw live signals into a numeric feature row.

    Cyclical time features are encoded with sin/cos so that, e.g., hour=23
    and hour=0 are recognised as *close together* rather than far apart
    (a plain integer encoding would wrongly treat midnight as maximally
    distant from 11pm).
    """
    hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
    hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)

    # Click-to-cart conversion intent ratio (guarded against div-by-zero).
    intent_ratio = add_to_carts / page_views if page_views > 0 else 0.0

    return np.array([
        sales_velocity,
        competitor_price_ratio,
        hour_sin, hour_cos,
        dow_sin, dow_cos,
        page_views,
        add_to_carts,
        intent_ratio,
        stock_level,
    ], dtype=np.float64)


FEATURE_NAMES = [
    "sales_velocity", "competitor_price_ratio",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "page_views", "add_to_carts", "intent_ratio", "stock_level",
]


# --------------------------------------------------------------------------- #
# 3. The ensemble forecaster
# --------------------------------------------------------------------------- #
class DemandForecaster:
    """
    Combines ExponentialSmoothing (baseline) with GradientBoostingRegressor
    (reactive, non-linear signal model) via an adaptive convex blend.
    """

    def __init__(self):
        self.smoother = ExponentialSmoother(alpha=0.35, beta=0.15)
        self.gbr = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, historical_series: np.ndarray, X: np.ndarray, y: np.ndarray):
        """
        historical_series : 1D array of past aggregate daily/hourly sales
                             (used purely for the smoother's baseline).
        X, y               : engineered feature matrix / demand targets for
                             the supervised reactive branch.
        """
        self.smoother.fit(historical_series)
        X_scaled = self.scaler.fit_transform(X)
        self.gbr.fit(X_scaled, y)
        self._is_fitted = True
        return self

    @staticmethod
    def _adaptive_blend_weight(page_views: int, add_to_carts: int) -> float:
        """
        Returns w in [0.15, 0.85], the weight given to the STATISTICAL
        baseline. The remainder (1 - w) goes to the reactive ML model.

        Intuition: total live traffic is a proxy for "signal strength".
        A logistic squashing function maps raw event counts to a smooth
        [0, 1] confidence score, then we invert it so that MORE live
        traffic => LESS reliance on the static baseline.

            confidence(events) = 1 / (1 + exp(-k * (events - midpoint)))
            w_baseline         = 0.85 - 0.70 * confidence(events)

        This keeps w bounded, monotonic, and smooth (no hard thresholds
        that would cause price/demand jumps at arbitrary cut-offs).
        """
        total_events = page_views + add_to_carts
        k, midpoint = 0.08, 40  # tuned so ~40 events is the "50% confidence" point
        confidence = 1.0 / (1.0 + np.exp(-k * (total_events - midpoint)))
        w_baseline = 0.85 - 0.70 * confidence
        return float(np.clip(w_baseline, 0.15, 0.85))

    def predict(
        self,
        sales_velocity: float,
        competitor_price_ratio: float,
        hour_of_day: int,
        day_of_week: int,
        page_views: int,
        add_to_carts: int,
        stock_level: int,
    ) -> dict:
        if not self._is_fitted:
            raise RuntimeError("Model has not been trained. Call .fit() first.")

        baseline_forecast = float(self.smoother.forecast(horizon=1)[0])

        feat = build_feature_vector(
            sales_velocity, competitor_price_ratio, hour_of_day,
            day_of_week, page_views, add_to_carts, stock_level,
        ).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat)
        reactive_forecast = float(self.gbr.predict(feat_scaled)[0])

        w = self._adaptive_blend_weight(page_views, add_to_carts)
        blended = w * baseline_forecast + (1 - w) * reactive_forecast

        return {
            "baseline_forecast": round(baseline_forecast, 3),
            "reactive_forecast": round(reactive_forecast, 3),
            "blend_weight_baseline": round(w, 3),
            "final_demand_forecast": round(max(blended, 0.0), 3),
        }


# --------------------------------------------------------------------------- #
# 4. Price Elasticity Engine
# --------------------------------------------------------------------------- #
@dataclass
class PricingGuardrails:
    max_increase_pct: float = 0.20   # +20% ceiling
    max_decrease_pct: float = 0.20   # -20% floor
    min_absolute_price: float = 1.0  # never price at/below this


class PriceElasticityEngine:
    """
    Converts a demand *signal* into a bounded price adjustment using a
    classical microeconomics elasticity formulation, discretised into a
    smooth NumPy function.

    Core formula
    ------------
    Percentage price change is modelled as a function of the *demand
    pressure index* (DPI), a unitless score in roughly [-1, 1]:

        DPI = clip(
                w1 * z(demand_velocity)
              + w2 * z(intent_ratio)
              - w3 * z(stock_level)
              - w4 * z(competitor_price_ratio - 1),
              -1, 1
            )

        price_multiplier = 1 + max_fluctuation * tanh(k * DPI)

    We use tanh rather than a raw linear scaling because:
      * it naturally saturates, so extreme DPI values don't blow past the
        guardrail even before we clip -- the guardrail becomes a soft
        physical limit, not just a hard cutoff that creates price
        discontinuities near the boundary.
      * its derivative is highest near DPI=0, i.e. the engine is MOST
        sensitive to small changes in *normal* demand conditions, and
        naturally flattens out (diminishing sensitivity) for already
        extreme conditions -- mirroring real-world price elasticity of
        demand curves.

    z(x) is a min-max normalisation against configured expected ranges
    for each signal (kept simple/transparent rather than a running
    z-score, so behaviour is deterministic and easy to unit test).
    """

    def __init__(self, guardrails: Optional[PricingGuardrails] = None, k: float = 1.5):
        self.guardrails = guardrails or PricingGuardrails()
        self.k = k  # steepness of the tanh saturation curve

        # Expected operating ranges used for normalisation -- calibrated
        # from the mock retail data distribution (see simulator.py).
        self._ranges = {
            "demand_velocity": (0.0, 10.0),     # units/minute
            "intent_ratio": (0.0, 1.0),         # add_to_cart / page_view
            "stock_level": (0.0, 500.0),        # units in warehouse
            "competitor_delta": (-0.3, 0.3),    # (competitor_price/our_price) - 1
        }
        self.weights = {
            "demand_velocity": 0.40,
            "intent_ratio": 0.25,
            "stock_level": 0.20,
            "competitor_delta": 0.15,
        }

    def _normalize(self, value: float, key: str) -> float:
        """Min-max scale to [-1, 1] given the calibrated operating range."""
        lo, hi = self._ranges[key]
        value = np.clip(value, lo, hi)
        scaled_0_1 = (value - lo) / (hi - lo + 1e-9)
        return float(2 * scaled_0_1 - 1)  # map [0,1] -> [-1,1]

    def compute_demand_pressure_index(
        self,
        demand_velocity: float,
        intent_ratio: float,
        stock_level: float,
        competitor_price_ratio: float,
    ) -> float:
        z_velocity = self._normalize(demand_velocity, "demand_velocity")
        z_intent = self._normalize(intent_ratio, "intent_ratio")
        # NOTE the sign: LOW stock should INCREASE pressure -> we negate.
        z_stock = -self._normalize(stock_level, "stock_level")
        competitor_delta = competitor_price_ratio - 1.0
        z_competitor = self._normalize(competitor_delta, "competitor_delta")

        dpi = (
            self.weights["demand_velocity"] * z_velocity
            + self.weights["intent_ratio"] * z_intent
            + self.weights["stock_level"] * z_stock
            + self.weights["competitor_delta"] * z_competitor
        )
        return float(np.clip(dpi, -1.0, 1.0))

    def compute_dynamic_price(
        self,
        base_price: float,
        demand_velocity: float,
        intent_ratio: float,
        stock_level: float,
        competitor_price_ratio: float,
    ) -> dict:
        dpi = self.compute_demand_pressure_index(
            demand_velocity, intent_ratio, stock_level, competitor_price_ratio
        )

        max_fluct = self.guardrails.max_increase_pct  # symmetric bound (0.20)
        multiplier = 1 + max_fluct * np.tanh(self.k * dpi)

        raw_price = base_price * multiplier

        # Hard guardrails (belt-and-braces on top of the tanh soft bound).
        floor = base_price * (1 - self.guardrails.max_decrease_pct)
        ceiling = base_price * (1 + self.guardrails.max_increase_pct)
        final_price = float(np.clip(raw_price, floor, ceiling))
        final_price = max(final_price, self.guardrails.min_absolute_price)

        return {
            "demand_pressure_index": round(dpi, 4),
            "price_multiplier": round(float(multiplier), 4),
            "base_price": round(base_price, 2),
            "dynamic_price": round(final_price, 2),
            "pct_change": round((final_price / base_price - 1) * 100, 2),
        }


# --------------------------------------------------------------------------- #
# 5. Synthetic training data generator (used to bootstrap the model so the
#    API is usable immediately, without requiring an external dataset).
# --------------------------------------------------------------------------- #
def generate_synthetic_training_data(n_samples: int = 2000, seed: int = 42):
    """
    Produces a plausible historical dataset:
      - a 1D daily sales series with weekly seasonality + trend + noise
        (for the ExponentialSmoother)
      - a feature matrix / target vector with realistic correlations
        (for the GradientBoostingRegressor)
    """
    rng = np.random.default_rng(seed)

    # --- Historical aggregate series: 90 days, weekly seasonality + trend ---
    days = np.arange(90)
    trend = 0.15 * days
    weekly_seasonality = 8 * np.sin(2 * np.pi * days / 7)
    noise = rng.normal(0, 2.5, size=90)
    historical_series = 50 + trend + weekly_seasonality + noise
    historical_series = np.clip(historical_series, 1, None)

    # --- Feature-based supervised dataset ---
    sales_velocity = rng.gamma(shape=2.0, scale=2.0, size=n_samples)
    competitor_price_ratio = rng.normal(1.0, 0.08, size=n_samples)
    hour_of_day = rng.integers(0, 24, size=n_samples)
    day_of_week = rng.integers(0, 7, size=n_samples)
    page_views = rng.poisson(lam=25, size=n_samples)
    add_to_carts = np.minimum(
        page_views, rng.poisson(lam=6, size=n_samples)
    )
    stock_level = rng.integers(0, 500, size=n_samples)

    X = np.vstack([
        build_feature_vector(
            sales_velocity[i], competitor_price_ratio[i], hour_of_day[i],
            day_of_week[i], page_views[i], add_to_carts[i], stock_level[i],
        )
        for i in range(n_samples)
    ])  # shape: (n_samples, n_features)

    # Ground-truth demand: a nonlinear function of the inputs + noise,
    # simulating "real" underlying dynamics the GBR must learn to recover.
    intent_ratio = np.divide(
        add_to_carts, page_views, out=np.zeros_like(page_views, dtype=float),
        where=page_views > 0
    )
    evening_boost = np.where((hour_of_day >= 18) & (hour_of_day <= 22), 1.4, 1.0)
    weekend_boost = np.where(day_of_week >= 5, 1.2, 1.0)
    scarcity_boost = 1 + 0.3 * np.exp(-stock_level / 100)
    competitor_effect = np.clip(2 - competitor_price_ratio, 0.5, 1.5)

    y = (
        sales_velocity * 3.0
        * evening_boost * weekend_boost * scarcity_boost * competitor_effect
        * (1 + 2.5 * intent_ratio)
        + rng.normal(0, 1.5, size=n_samples)
    )
    y = np.clip(y, 0, None)

    return historical_series, X, y
