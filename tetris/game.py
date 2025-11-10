# game.py

import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pygame

from piece import Piece
from board import Board
from settings import *
import analytics
from difficulty_adapter import DifficultyAdapter, DifficultyConfig, DifficultyState


class Game:
    def __init__(
        self,
        difficulty_level: int = 0,
        difficulty_adapter: Optional[DifficultyAdapter] = None,
        log_dir: str = "logs",
    ):
        self.board = Board()
        self.difficulty_adapter = difficulty_adapter or DifficultyAdapter()
        self.difficulty_state = DifficultyState()
        self.overlay_enabled = SHOW_DIFFICULTY_OVERLAY
        self.game_log_path = self._open_log_file(Path(log_dir))
        self.log_file = self.game_log_path.open("w", encoding="utf-8")

        initial_bias = self.difficulty_state.piece_bias_mode
        self.max_preview_depth = max(self.difficulty_adapter.config.preview.max_preview, 1)
        self.preview_queue = [
            self._choose_piece_type(initial_bias) for _ in range(self.max_preview_depth)
        ]
        self.current_piece = self._create_piece(self._choose_piece_type(initial_bias))
        self.hold_piece = None
        self.hold_used = False
        self.score = 0
        self.lines_cleared = 0
        self.piece_count = 0
        self.game_over = False
        self.isolated_empty_sum = 0
        self.isolated_empty_count = 0
        self.difficulty_level = difficulty_level
        self._last_garbage_time = time.time()
        self._last_log_time = 0.0

    def __del__(self):
        try:
            if self.log_file:
                self.log_file.close()
        except Exception:
            pass

    def _open_log_file(self, log_dir: Path) -> Path:
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        return log_dir / f"game_session_{ts}.jsonl"

    def update_isolated_empty(self):
        val = self.board.count_isolated_empty()
        self.isolated_empty_sum += val
        self.isolated_empty_count += 1

    def get_isolated_empty_avg(self):
        if self.isolated_empty_count == 0:
            return 0
        return self.isolated_empty_sum / self.isolated_empty_count

    def hold(self):
        if not self.difficulty_state.hold_allowed:
            return
        if self.hold_used:
            return
        if self.hold_piece is None:
            self.hold_piece = self.current_piece
            self.current_piece = self._pull_next_piece()
        else:
            self.hold_piece, self.current_piece = self.current_piece, self.hold_piece
        self.current_piece.x = 3
        self.current_piece.y = 0
        if self.hold_piece:
            self.hold_piece.x = 3
            self.hold_piece.y = 0
        self.hold_used = True

    def spawn_piece(self):
        self.current_piece = self._pull_next_piece()
        self.hold_used = False
        self.piece_count += 1
        if not self.board.valid_position(self.current_piece):
            self.game_over = True
            analytics.on_game_over(self.board.get_state(), self.score, self)

    def move(self, dx, dy):
        p = self.current_piece.copy()
        p.x += dx
        p.y += dy
        if self.board.valid_position(p):
            self.current_piece = p
            analytics.log_move(self.board.grid, self.current_piece, f"move_{dx}_{dy}")
            return True
        return False

    def rotate(self):
        p = self.current_piece.copy()
        p.rotate()
        if self.board.valid_position(p):
            self.current_piece = p
            analytics.log_move(self.board.grid, self.current_piece, "rotate")
            return True
        return False

    def drop(self):
        if not self.move(0, 1):
            self.board.place_piece(self.current_piece)
            cleared = self.board.clear_lines()
            if cleared:
                self.score += [0, 40, 100, 300, 1200][cleared]
                self.lines_cleared += cleared
                analytics.log_move(self.board.grid, self.current_piece, "clear", reward=cleared)
            self.spawn_piece()

    def hard_drop(self):
        while self.move(0, 1):
            pass
        self.drop()

    def tick(self):
        if not self.game_over:
            self._maybe_add_garbage()
            self.drop()

    def set_affect_payload(self, payload):
        self.difficulty_state = self.difficulty_adapter.update(payload)

    def toggle_overlay(self):
        self.overlay_enabled = not self.overlay_enabled

    def log_state(self):
        now = time.time()
        if now - self._last_log_time < 0.5:  # log at most twice per second
            return
        self._last_log_time = now
        payload = {
            "timestamp": now,
            "valence": self.difficulty_state.valence,
            "arousal": self.difficulty_state.arousal,
            "fall_speed_ms": self.difficulty_state.fall_speed_ms,
            "garbage_interval": self.difficulty_state.garbage_interval,
            "piece_bias_mode": self.difficulty_state.piece_bias_mode,
            "hold_allowed": self.difficulty_state.hold_allowed,
            "preview_depth": self.difficulty_state.preview_depth,
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "pieces": self.piece_count,
        }
        self.log_file.write(json.dumps(payload) + "\n")
        self.log_file.flush()

    def _choose_piece_type(self, bias_mode: str) -> str:
        weights_map = {
            "normal": {"I": 1, "J": 1, "L": 1, "O": 1, "S": 1, "T": 1, "Z": 1},
            "recovery": {"I": 3, "O": 3, "T": 2, "S": 1, "Z": 1, "J": 1, "L": 1},
            "challenge": {"S": 3, "Z": 3, "T": 2, "J": 1, "L": 1, "I": 1, "O": 1},
            "stress": {"S": 4, "Z": 4, "J": 2, "L": 2, "T": 2, "I": 1, "O": 1},
        }
        weights = weights_map.get(bias_mode, weights_map["normal"])
        pieces = list(weights.keys())
        probs = [weights[p] for p in pieces]
        return random.choices(pieces, weights=probs, k=1)[0]

    def _create_piece(self, piece_type: str) -> Piece:
        piece = Piece(piece_type)
        piece.x = 3
        piece.y = 0
        return piece

    def _pull_next_piece(self) -> Piece:
        bias_mode = self.difficulty_state.piece_bias_mode
        if not self.preview_queue:
            self.preview_queue.append(self._choose_piece_type(bias_mode))
        piece_type = self.preview_queue.pop(0)
        self.preview_queue.append(self._choose_piece_type(bias_mode))
        return self._create_piece(piece_type)

    def peek_next_pieces(self, depth: int):
        depth = max(0, min(depth, len(self.preview_queue)))
        return [self._create_piece(t) for t in self.preview_queue[:depth]]

    def _maybe_add_garbage(self):
        interval = self.difficulty_state.garbage_interval
        if interval is None:
            return
        now = time.time()
        if now - self._last_garbage_time < interval:
            return
        gap = random.randint(0, GRID_WIDTH - 1)
        self.board.grid = self.board.grid[1:]
        garbage_row = np.ones(GRID_WIDTH, dtype=np.uint8)
        garbage_row[gap] = 0
        self.board.grid = np.vstack([self.board.grid, garbage_row])
        self.board.colors = self.board.colors[1:]
        color_row = np.full(GRID_WIDTH, "garbage", dtype=object)
        color_row[gap] = ""
        self.board.colors = np.vstack([self.board.colors, color_row])
        self._last_garbage_time = now