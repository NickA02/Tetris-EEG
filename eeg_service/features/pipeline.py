"""
Feature extraction pipeline abstractions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol


class FeaturePipeline(Protocol):
    """Convert raw device readings into model-ready features."""

    def transform(self, raw_sample: Dict) -> Dict:
        ...


@dataclass
class IdentityPipeline:
    """
    Default pipeline that forwards raw samples unchanged.
    Useful while the full feature engineering stack is not implemented.
    """

    def transform(self, raw_sample: Dict) -> Dict:
        return raw_sample

