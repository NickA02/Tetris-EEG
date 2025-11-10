"""
TCP server responsible for streaming affect predictions to clients.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def _serialize_message(message: Dict) -> bytes:
    return (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")


@dataclass
class AffectServer:
    host: str = "127.0.0.1"
    port: int = 5555
    backlog: int = 5
    heartbeat_interval: float = 5.0

    _socket: Optional[socket.socket] = field(init=False, default=None)
    _clients: List[socket.socket] = field(init=False, default_factory=list)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _accept_thread: Optional[threading.Thread] = field(init=False, default=None)
    _heartbeat_thread: Optional[threading.Thread] = field(init=False, default=None)
    _running: bool = field(init=False, default=False)

    def start(self) -> None:
        if self._running:
            return
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.listen(self.backlog)
        self._running = True
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        print(f"[AffectServer] Listening on {self.host}:{self.port}")

    def stop(self) -> None:
        self._running = False
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
        with self._lock:
            for client in self._clients:
                try:
                    client.shutdown(socket.SHUT_RDWR)
                    client.close()
                except OSError:
                    pass
            self._clients.clear()
        print("[AffectServer] Stopped")

    def broadcast(self, message: Dict) -> None:
        payload = _serialize_message(message)
        stale: List[socket.socket] = []
        with self._lock:
            for client in list(self._clients):
                try:
                    client.sendall(payload)
                except OSError:
                    stale.append(client)
            for client in stale:
                self._clients.remove(client)
                try:
                    client.close()
                except OSError:
                    pass

    def _accept_loop(self) -> None:
        assert self._socket is not None
        while self._running:
            try:
                client, addr = self._socket.accept()
                client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._lock:
                    self._clients.append(client)
                print(f"[AffectServer] Client connected from {addr}")
            except OSError:
                if self._running:
                    print("[AffectServer] Socket accept failed; continuing")
                break

    def _heartbeat_loop(self) -> None:
        while self._running:
            time.sleep(self.heartbeat_interval)
            if not self._running:
                break
            self.broadcast({"type": "heartbeat", "timestamp": time.time()})

