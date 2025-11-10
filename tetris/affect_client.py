"""
Client for receiving affect updates from the EEG service.
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class AffectClient:
    host: str = "127.0.0.1"
    port: int = 5555
    reconnect_interval: float = 2.0

    _thread: Optional[threading.Thread] = field(init=False, default=None)
    _running: bool = field(init=False, default=False)
    _queue: "queue.Queue[Dict]" = field(init=False, default_factory=queue.Queue)
    _latest: Optional[Dict] = field(init=False, default=None)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def get_latest(self) -> Optional[Dict]:
        """Return the most recent affect payload if available."""
        try:
            while True:
                self._latest = self._queue.get_nowait()
        except queue.Empty:
            pass
        return self._latest

    def _run(self) -> None:
        while self._running:
            try:
                with socket.create_connection((self.host, self.port), timeout=5.0) as sock:
                    sock_file = sock.makefile("r", encoding="utf-8")
                    print(f"[AffectClient] Connected to {self.host}:{self.port}")
                    for line in sock_file:
                        if not line.strip():
                            continue
                        try:
                            payload = json.loads(line)
                            if payload.get("type") == "affect":
                                self._queue.put(payload)
                        except json.JSONDecodeError:
                            continue
                        if not self._running:
                            break
            except (OSError, ConnectionError):
                if not self._running:
                    break
                print("[AffectClient] Connection failed, retrying…")
                time.sleep(self.reconnect_interval)

