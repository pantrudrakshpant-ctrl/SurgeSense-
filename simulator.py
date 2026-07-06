"""
simulator.py
-------------
Emulates a live e-commerce traffic feed: user click-stream events
(page_view / add_to_cart / purchase) and periodic stock/competitor-price
updates, running on a background daemon thread.

Why threading (not asyncio) here
=================================
The event generator is CPU-light and I/O-free (pure in-memory random
sampling), and it needs to run truly concurrently with FastAPI's request
handling without the generator's `time.sleep`-based pacing blocking the
event loop. A dedicated daemon `threading.Thread` decouples it completely
from the ASGI event loop used by the web server -- simplest correct
option for a single-process, local-first system.

Thread safety
=============
Both the web server's request-handling threads/coroutines and the
background generator thread touch the same rolling buffer of the last
100 events. We protect it with a `threading.Lock` around every read and
write to guarantee `deque` mutations are atomic from the perspective of
concurrent callers (CPython's GIL makes single deque ops atomic-ish, but
we still lock explicitly so buffer invariants like max-length + composite
counters stay consistent -- correctness should never depend on
implementation-specific GIL behaviour).
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


@dataclass
class UserEvent:
    event_type: str          # "page_view" | "add_to_cart" | "purchase"
    product_id: str
    timestamp: float = field(default_factory=time.time)


class ThreadSafeSlidingWindow:
    """
    Fixed-capacity, thread-safe ring buffer of the most recent user events.

    Uses collections.deque(maxlen=capacity) -- O(1) appends/evictions from
    both ends -- wrapped in a threading.Lock so producer (simulator
    thread) and consumers (FastAPI request handlers) never race on the
    underlying list.
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self._buffer: Deque[UserEvent] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, event: UserEvent) -> None:
        with self._lock:
            self._buffer.append(event)

    def snapshot(self) -> List[UserEvent]:
        """Returns a shallow copy so callers can iterate without holding the lock."""
        with self._lock:
            return list(self._buffer)

    def compute_demand_velocity(self, window_seconds: float = 60.0) -> Dict[str, float]:
        """
        Instantaneous demand signals computed directly from the live
        sliding window:

          - demand_velocity : purchases-equivalent events per minute,
                               weighting add_to_cart as a *soft* purchase
                               signal (0.3x) and purchase as full (1.0x).
          - page_views / add_to_carts : raw counts within the window,
                               fed into the pricing engine's intent ratio.

        This is intentionally O(n) over at most 100 items -- cheap enough
        to run on every API request without caching.
        """
        now = time.time()
        events = self.snapshot()
        recent = [e for e in events if now - e.timestamp <= window_seconds]

        page_views = sum(1 for e in recent if e.event_type == "page_view")
        add_to_carts = sum(1 for e in recent if e.event_type == "add_to_cart")
        purchases = sum(1 for e in recent if e.event_type == "purchase")

        weighted_events = purchases * 1.0 + add_to_carts * 0.3
        elapsed_minutes = max(window_seconds / 60.0, 1e-6)
        demand_velocity = weighted_events / elapsed_minutes

        return {
            "page_views": page_views,
            "add_to_carts": add_to_carts,
            "purchases": purchases,
            "demand_velocity": round(demand_velocity, 4),
            "window_seconds": window_seconds,
            "buffer_size": len(events),
        }


class ECommerceEventGenerator(threading.Thread):
    """
    Background daemon thread that streams synthetic-but-plausible live
    traffic into a ThreadSafeSlidingWindow, and mutates a shared mutable
    "world state" dict (stock levels, competitor prices) that the pricing
    endpoint reads from.

    Runs indefinitely until `.stop()` is called; marked as `daemon=True`
    by the caller so it never blocks process shutdown.
    """

    PRODUCT_IDS = ["SKU_1001", "SKU_1002", "SKU_1003"]

    def __init__(
        self,
        buffer: ThreadSafeSlidingWindow,
        world_state: Dict,
        world_state_lock: threading.Lock,
        tick_interval: float = 0.4,
    ):
        super().__init__(daemon=True)
        self.buffer = buffer
        self.world_state = world_state
        self.world_state_lock = world_state_lock
        self.tick_interval = tick_interval
        self._stop_flag = threading.Event()

    def stop(self) -> None:
        self._stop_flag.set()

    def _emit_click_stream_event(self) -> None:
        # Weighted random choice mirrors realistic funnel drop-off:
        # most traffic is browsing, a minority adds to cart, fewer buy.
        event_type = random.choices(
            population=["page_view", "add_to_cart", "purchase"],
            weights=[0.70, 0.22, 0.08],
            k=1,
        )[0]
        product_id = random.choice(self.PRODUCT_IDS)
        self.buffer.push(UserEvent(event_type=event_type, product_id=product_id))

    def _update_world_state(self) -> None:
        with self.world_state_lock:
            for pid in self.PRODUCT_IDS:
                state = self.world_state[pid]

                # Stock slowly depletes with purchases, occasionally
                # restocked -- a simple random walk bounded to [0, 500].
                delta = random.choice([-1, -1, 0, 0, 0, 1])
                state["stock_level"] = int(
                    max(0, min(500, state["stock_level"] + delta * random.randint(0, 3)))
                )

                # Competitor price ratio drifts around 1.0 with small
                # Gaussian jitter -- models minor real-world price wars.
                jitter = random.gauss(0, 0.01)
                state["competitor_price_ratio"] = round(
                    max(0.5, min(1.5, state["competitor_price_ratio"] + jitter)), 4
                )

    def run(self) -> None:
        while not self._stop_flag.is_set():
            self._emit_click_stream_event()
            self._update_world_state()
            time.sleep(self.tick_interval)


def build_initial_world_state() -> Dict[str, Dict]:
    """Seed state: base price + starting stock + starting competitor ratio per SKU."""
    return {
        "SKU_1001": {"base_price": 1299.0, "stock_level": 120, "competitor_price_ratio": 1.0},
        "SKU_1002": {"base_price": 799.0, "stock_level": 40, "competitor_price_ratio": 1.0},
        "SKU_1003": {"base_price": 2499.0, "stock_level": 300, "competitor_price_ratio": 1.0},
    }
