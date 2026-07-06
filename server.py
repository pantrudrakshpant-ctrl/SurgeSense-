"""
server.py
---------
FastAPI application exposing the demand forecasting and dynamic pricing
engine. On startup it:

  1. Generates synthetic historical training data and fits the
     DemandForecaster (ExponentialSmoothing + GradientBoostingRegressor).
  2. Spins up the background ECommerceEventGenerator thread + thread-safe
     sliding window buffer to simulate a live traffic feed.

Two endpoints are exposed:
  - GET /predict-demand?product_id=...   -> ensemble demand forecast
  - GET /get-dynamic-price?product_id=... -> elasticity-adjusted price

Both read *live* state (click-stream velocity, stock, competitor price)
from the in-memory simulator, so repeated calls will return different
numbers as the background thread ticks forward -- proving the pipeline is
actually reactive rather than a static demo.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model import DemandForecaster, PriceElasticityEngine, generate_synthetic_training_data
from simulator import (
    ECommerceEventGenerator,
    ThreadSafeSlidingWindow,
    build_initial_world_state,
)

app = FastAPI(
    title="Real-Time Dynamic Demand Forecasting & Intent-Based Pricing Engine",
    description="Local-first ML engine combining Exponential Smoothing + "
                "Gradient Boosting demand forecasts with a NumPy price "
                "elasticity function driven by live click-stream signals.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Global engine state (single-process, local-first -- no external DB/broker)
# --------------------------------------------------------------------------- #
forecaster = DemandForecaster()
pricing_engine = PriceElasticityEngine()

event_buffer = ThreadSafeSlidingWindow(capacity=100)
world_state = build_initial_world_state()
world_state_lock = threading.Lock()
event_generator: ECommerceEventGenerator | None = None


@app.on_event("startup")
def startup_event():
    global event_generator

    # 1. Train the ensemble on synthetic-but-realistic historical data.
    historical_series, X, y = generate_synthetic_training_data(n_samples=2000)
    forecaster.fit(historical_series, X, y)

    # 2. Launch the live event stream in the background.
    event_generator = ECommerceEventGenerator(
        buffer=event_buffer,
        world_state=world_state,
        world_state_lock=world_state_lock,
        tick_interval=0.4,
    )
    event_generator.start()


@app.on_event("shutdown")
def shutdown_event():
    if event_generator is not None:
        event_generator.stop()


def _get_product_state(product_id: str) -> dict:
    with world_state_lock:
        if product_id not in world_state:
            raise HTTPException(status_code=404, detail=f"Unknown product_id '{product_id}'")
        return dict(world_state[product_id])  # shallow copy for safety


@app.get("/")
def root():
    return {
        "service": "dynamic-demand-pricing-engine",
        "status": "online",
        "endpoints": ["/predict-demand", "/get-dynamic-price", "/live-buffer-stats"],
        "products": list(world_state.keys()),
    }


@app.get("/live-buffer-stats")
def live_buffer_stats():
    """Debug/observability endpoint: raw sliding-window derived signals."""
    return event_buffer.compute_demand_velocity(window_seconds=60.0)


@app.get("/predict-demand")
def predict_demand(product_id: str = "SKU_1001", hour_override: int | None = None):
    """
    Returns the blended ensemble demand forecast for a given SKU, using
    LIVE click-stream velocity pulled from the sliding window buffer.
    """
    t0 = time.perf_counter()

    product_state = _get_product_state(product_id)
    live_signals = event_buffer.compute_demand_velocity(window_seconds=60.0)

    now = datetime.now()
    hour_of_day = hour_override if hour_override is not None else now.hour
    day_of_week = now.weekday()

    result = forecaster.predict(
        sales_velocity=live_signals["demand_velocity"],
        competitor_price_ratio=product_state["competitor_price_ratio"],
        hour_of_day=hour_of_day,
        day_of_week=day_of_week,
        page_views=live_signals["page_views"],
        add_to_carts=live_signals["add_to_carts"],
        stock_level=product_state["stock_level"],
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    return {
        "product_id": product_id,
        "timestamp": now.isoformat(),
        "live_signals": live_signals,
        "forecast": result,
        "inference_latency_ms": latency_ms,
    }


@app.get("/get-dynamic-price")
def get_dynamic_price(product_id: str = "SKU_1001"):
    """
    Returns the elasticity-adjusted dynamic price for a given SKU, bounded
    by the +/-20% guardrail, driven by live demand velocity, intent ratio,
    stock level, and competitor price ratio.
    """
    t0 = time.perf_counter()

    product_state = _get_product_state(product_id)
    live_signals = event_buffer.compute_demand_velocity(window_seconds=60.0)

    page_views = live_signals["page_views"]
    add_to_carts = live_signals["add_to_carts"]
    intent_ratio = add_to_carts / page_views if page_views > 0 else 0.0

    pricing_result = pricing_engine.compute_dynamic_price(
        base_price=product_state["base_price"],
        demand_velocity=live_signals["demand_velocity"],
        intent_ratio=intent_ratio,
        stock_level=product_state["stock_level"],
        competitor_price_ratio=product_state["competitor_price_ratio"],
    )

    latency_ms = round((time.perf_counter() - t0) * 1000, 3)

    return {
        "product_id": product_id,
        "stock_level": product_state["stock_level"],
        "competitor_price_ratio": product_state["competitor_price_ratio"],
        "live_intent_ratio": round(intent_ratio, 4),
        "pricing": pricing_result,
        "inference_latency_ms": latency_ms,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
