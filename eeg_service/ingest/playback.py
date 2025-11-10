"""
Offline playback of recorded affect or feature streams.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterator, Optional


class PlaybackSource:
    """
    Iterate over recorded data in JSON-lines format while preserving timing.

    Expected schema per line:
    {
        "timestamp": 1690000000.0,   # seconds since epoch
        "features": {...},           # optional raw feature dictionary
        "valence": 3.2,              # optional affect fields if already computed
        "arousal": 1.5
    }
    """

    def __init__(self, path: str | Path, speed: float = 1.0) -> None:
        self.path = Path(path)
        self.speed = speed
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def __iter__(self) -> Iterator[Dict]:
        prev_ts: Optional[float] = None
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                payload = json.loads(line)
                ts = payload.get("timestamp")
                if prev_ts is not None and ts is not None:
                    wait = max((ts - prev_ts) / self.speed, 0)
                    if wait > 0:
                        time.sleep(wait)
                prev_ts = ts
                yield payload

