"""
EEG real-time inference service.

This package exposes utilities to ingest raw EEG data, run feature extraction
and inference with a pluggable model, and stream affect predictions to
consumers such as the Tetris game.
"""

__all__ = [
    "model_registry",
]

