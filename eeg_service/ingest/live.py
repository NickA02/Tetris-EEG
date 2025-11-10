"""
Placeholder for live EEG ingestion.

Implement `LiveEEGSource` by integrating with the actual hardware API.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator
import time
import random


class LiveEEGSource:
    """
    Prototype live EEG source.

    This implementation simply yields random feature vectors at a fixed cadence.
    Replace `yield_features` with real device integration.
    """

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self._rng = random.Random()

    def __iter__(self) -> Iterator[Dict]:
        return self.yield_features()

    def yield_features(self) -> Iterator[Dict]:
        while True:
            features = {
                "alpha_power": self._rng.uniform(0, 1),
                "beta_power": self._rng.uniform(0, 1),
                "gamma_power": self._rng.uniform(0, 1),
                "theta_power": self._rng.uniform(0, 1),
            }
            yield features
            time.sleep(self.interval)

