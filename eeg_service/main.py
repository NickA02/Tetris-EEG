from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional

try:
    from .features.pipeline import IdentityPipeline
    from .ingest.live import LiveEEGSource
    from .ingest.playback import PlaybackSource
    from .model_registry import load_model
    from .stream.server import AffectServer
except ImportError:  # pragma: no cover - allows running as a script
    from features.pipeline import IdentityPipeline  # type: ignore
    from ingest.live import LiveEEGSource  # type: ignore
    from ingest.playback import PlaybackSource  # type: ignore
    from model_registry import load_model  # type: ignore
    from stream.server import AffectServer  # type: ignore


def _ensure_log_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _open_log_file(log_dir: Path, mode: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    fname = f"affect_{mode}_{ts}.jsonl"
    return log_dir / fname


def _decode_playback_sample(sample: Dict) -> Dict:
    """
    Normalise playback payloads so downstream code can treat them
    like live samples.
    """
    if "features" in sample:
        return sample
    # assume `sample` already contains feature-like keys
    return {"features": sample, "timestamp": sample.get("timestamp")}


def run_pipeline(
    iterator: Iterator[Dict],
    model_name: str,
    config: Optional[str],
    log_dir: Path,
    server: AffectServer,
    source_meta: Dict,
) -> None:
    model = load_model(model_name, config)
    pipeline = IdentityPipeline()
    log_path = _open_log_file(log_dir, source_meta.get("mode", "unknown"))

    with log_path.open("w", encoding="utf-8") as log_file:
        for raw in iterator:
            if source_meta.get("mode") == "playback":
                raw = _decode_playback_sample(raw)
            features = raw.get("features", raw)
            predictions = model.predict(features)

            valence = predictions.get("valence")
            arousal = predictions.get("arousal")
            valence_binary = predictions.get("valence_binary")
            arousal_binary = predictions.get("arousal_binary")

            # fallback binary if only continuous values exist
            if valence_binary is None and valence is not None:
                valence_binary = 1 if valence > 2.5 else 0
            if arousal_binary is None and arousal is not None:
                arousal_binary = 1 if arousal > 2.5 else 0

            payload = {
                "type": "affect",
                "timestamp": time.time(),
                "valence": valence,
                "arousal": arousal,
                "valence_binary": valence_binary,
                "arousal_binary": arousal_binary,
                "meta": {**source_meta, "model": model_name},
            }
            server.broadcast(payload)
            log_file.write(json.dumps(payload) + "\n")
            log_file.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EEG affect streaming service")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument("--model", default="dummy", help="Model identifier to load")
    parser.add_argument(
        "--model-config", default=None, help="Optional JSON config for the model"
    )
    parser.add_argument(
        "--mode",
        choices=["dummy", "live", "playback"],
        default="dummy",
        help="Data source mode",
    )
    parser.add_argument(
        "--playback-file",
        default=None,
        help="Path to JSONL playback file when mode=playback",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = real-time)",
    )
    parser.add_argument(
        "--live-interval",
        type=float,
        default=1.0,
        help="Interval between live predictions (seconds) for dummy/live mode",
    )
    parser.add_argument(
        "--log-dir", default="logs", help="Directory to store affect logs"
    )

    args = parser.parse_args(argv)

    server = AffectServer(host=args.host, port=args.port)
    server.start()

    log_dir = _ensure_log_dir(Path(args.log_dir))

    try:
        if args.mode == "playback":
            if not args.playback_file:
                parser.error("--playback-file is required when mode=playback")
            iterator = PlaybackSource(args.playback_file, speed=args.playback_speed)
            run_pipeline(
                iterator,
                args.model,
                args.model_config,
                log_dir,
                server,
                {"mode": "playback"},
            )
        elif args.mode == "live":
            iterator = LiveEEGSource(interval=args.live_interval)
            run_pipeline(
                iterator,
                args.model,
                args.model_config,
                log_dir,
                server,
                {"mode": "live"},
            )
        else:  # dummy
            iterator = LiveEEGSource(interval=args.live_interval)
            run_pipeline(
                iterator,
                args.model,
                args.model_config,
                log_dir,
                server,
                {"mode": "dummy"},
            )
    except KeyboardInterrupt:
        print("[EEG Service] Interrupted by user")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

