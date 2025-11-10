"""
Translate valence/arousal readings into concrete game difficulty parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


def _smooth(prev: Optional[float], new: Optional[float], alpha: float) -> Optional[float]:
    if new is None:
        return prev
    if prev is None:
        return new
    return prev * (1 - alpha) + new * alpha


@dataclass
class FallSpeedConfig:
    enabled: bool = True
    easy_ms: float = 700.0
    hard_ms: float = 150.0
    smoothing: float = 0.2


@dataclass
class GarbageConfig:
    enabled: bool = False
    threshold: float = 0.25  # trigger when low affect < threshold
    interval_seconds: float = 10.0


@dataclass
class PieceBiasConfig:
    enabled: bool = False
    smoothing: float = 0.2


@dataclass
class HoldConfig:
    enabled: bool = True
    disable_threshold: float = 0.2  # disable hold if both v/a below


@dataclass
class PreviewConfig:
    enabled: bool = False
    min_preview: int = 1
    max_preview: int = 3


@dataclass
class DifficultyConfig:
    fall_speed: FallSpeedConfig = field(default_factory=FallSpeedConfig)
    garbage: GarbageConfig = field(default_factory=GarbageConfig)
    piece_bias: PieceBiasConfig = field(default_factory=PieceBiasConfig)
    hold: HoldConfig = field(default_factory=HoldConfig)
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    scale_max: float = 5.0


@dataclass
class DifficultyState:
    valence: Optional[float] = None
    arousal: Optional[float] = None
    fall_speed_ms: float = 600.0
    garbage_interval: Optional[float] = None
    piece_bias_mode: str = "normal"
    hold_allowed: bool = True
    preview_depth: int = 1


class DifficultyAdapter:
    def __init__(self, config: DifficultyConfig | None = None) -> None:
        self.config = config or DifficultyConfig()
        self.state = DifficultyState()

    def update(self, payload: Dict) -> DifficultyState:
        valence = payload.get("valence")
        arousal = payload.get("arousal")
        if valence is None:
            vb = payload.get("valence_binary")
            valence = 5.0 if vb == 1 else 0.0 if vb == 0 else None
        if arousal is None:
            ab = payload.get("arousal_binary")
            arousal = 5.0 if ab == 1 else 0.0 if ab == 0 else None

        self.state.valence = _smooth(self.state.valence, valence, 0.3)
        self.state.arousal = _smooth(self.state.arousal, arousal, 0.3)

        scale_max = self.config.scale_max
        payload_scale = payload.get("meta", {}).get("scale_max")
        if isinstance(payload_scale, (int, float)) and payload_scale > 0:
            scale_max = float(payload_scale)
        if scale_max <= 0:
            scale_max = 5.0

        val_norm = (
            max(min(self.state.valence / scale_max, 1.0), 0.0)
            if self.state.valence is not None
            else None
        )
        aro_norm = (
            max(min(self.state.arousal / scale_max, 1.0), 0.0)
            if self.state.arousal is not None
            else None
        )

        avg_norm = None
        if val_norm is not None and aro_norm is not None:
            avg_norm = (val_norm + aro_norm) / 2.0
        elif val_norm is not None:
            avg_norm = val_norm
        elif aro_norm is not None:
            avg_norm = aro_norm

        # Fall speed mapping (smaller ms -> harder)
        if self.config.fall_speed.enabled:
            easy = self.config.fall_speed.easy_ms
            hard = self.config.fall_speed.hard_ms
            if avg_norm is None:
                target = (easy + hard) / 2
            else:
                hardness = 1.0 - avg_norm  # high affect -> easy (slow fall)
                target = easy - (easy - hard) * hardness
            self.state.fall_speed_ms = _smooth(
                self.state.fall_speed_ms, target, self.config.fall_speed.smoothing
            )

        # Garbage inject interval (lower -> more frequent)
        if self.config.garbage.enabled and val_norm is not None and aro_norm is not None:
            low = min(val_norm, aro_norm)
            if low < self.config.garbage.threshold:
                self.state.garbage_interval = self.config.garbage.interval_seconds
            else:
                self.state.garbage_interval = None
        else:
            self.state.garbage_interval = None

        # Piece bias mapping
        if self.config.piece_bias.enabled and val_norm is not None and aro_norm is not None:
            if val_norm < 0.3 and aro_norm < 0.3:
                mode = "stress"
            elif val_norm < 0.3:
                mode = "challenge"
            elif val_norm > 0.7 and aro_norm > 0.7:
                mode = "recovery"
            else:
                mode = "normal"
            self.state.piece_bias_mode = mode
        else:
            self.state.piece_bias_mode = "normal"

        # Hold
        if self.config.hold.enabled and val_norm is not None and aro_norm is not None:
            low_both = val_norm < self.config.hold.disable_threshold and aro_norm < self.config.hold.disable_threshold
            self.state.hold_allowed = not low_both
        else:
            self.state.hold_allowed = True

        # Preview depth
        if self.config.preview.enabled and val_norm is not None:
            depth = self.config.preview.min_preview + int(
                round(
                    val_norm * (self.config.preview.max_preview - self.config.preview.min_preview)
                )
            )
            self.state.preview_depth = max(self.config.preview.min_preview, min(depth, self.config.preview.max_preview))
        else:
            self.state.preview_depth = self.config.preview.min_preview

        return self.state

