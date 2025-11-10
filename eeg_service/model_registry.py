"""
Model registry and wrappers for EEG affect inference.

The goal is to present a common interface so that different models can be
swapped in without touching the streaming or ingestion code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Protocol, runtime_checkable
import json
import random
import time


@runtime_checkable
class AffectModel(Protocol):
    """Protocol describing the minimal inference interface."""

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Return affect scores given a feature dictionary."""


@dataclass
class DummyModel:
    """
    Simple baseline model used for end-to-end testing.

    It emits random valence/arousal values with optional binary encodings.
    """

    seed: int | None = None
    output_mode: str = "continuous"  # "continuous" | "binary"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        if self.output_mode == "binary":
            valence_bin = self._rng.randint(0, 1)
            arousal_bin = self._rng.randint(0, 1)
            return {
                "valence": float(valence_bin * 5),
                "arousal": float(arousal_bin * 5),
                "valence_binary": valence_bin,
                "arousal_binary": arousal_bin,
            }

        # default continuous 0-5 range
        valence = self._rng.uniform(0.0, 5.0)
        arousal = self._rng.uniform(0.0, 5.0)
        return {
            "valence": valence,
            "arousal": arousal,
            "valence_binary": 1 if valence > 2.5 else 0,
            "arousal_binary": 1 if arousal > 2.5 else 0,
        }


def load_model(name: str, config_path: str | Path | None = None) -> AffectModel:
    """
    Resolve a model identifier into an instantiated object.

    Parameters
    ----------
    name:
        Registered model name. Currently supported:
        - "dummy-continuous"
        - "dummy-binary"
        - Path to a JSON file describing a pre-trained model (future extension)
    config_path:
        Optional path to a config JSON that may provide additional parameters.
    """
    if name in {"dummy", "dummy-continuous"}:
        kwargs: Dict[str, Any] = {}
        if config_path:
            with open(config_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            kwargs.update(data.get("dummy_model", {}))
        return DummyModel(output_mode="continuous", **kwargs)

    if name == "dummy-binary":
        kwargs = {}
        if config_path:
            with open(config_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            kwargs.update(data.get("dummy_model", {}))
        return DummyModel(output_mode="binary", **kwargs)

    if Path(name).exists():
        raise NotImplementedError(
            "Loading of saved models from disk is not implemented yet. "
            "Please extend `load_model` to handle custom models."
        )

    raise ValueError(f"Unknown model specifier: {name}")

