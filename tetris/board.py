# board.py

import numpy as np
from .settings import GRID_WIDTH, GRID_HEIGHT

class Board:
    def count_blocks(self):
        return int(np.sum(self.grid))

    def count_isolated_empty(self):
        # Find empty blocks not connected to the top
        visited = np.zeros_like(self.grid, dtype=bool)
        # Mark all empty blocks connected to the top
        stack = [(0, x) for x in range(self.grid.shape[1]) if self.grid[0, x] == 0]
        while stack:
            y, x = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < self.grid.shape[0] and 0 <= nx < self.grid.shape[1]:
                    if self.grid[ny, nx] == 0 and not visited[ny, nx]:
                        stack.append((ny, nx))
        # Isolated empty blocks: empty blocks not visited
        isolated = 0
        for y in range(self.grid.shape[0]):
            for x in range(self.grid.shape[1]):
                if self.grid[y, x] == 0 and not visited[y, x]:
                    isolated += 1
        return isolated
    def __init__(self):
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.uint8)
        self.colors = np.full((GRID_HEIGHT, GRID_WIDTH), fill_value="", dtype=object)

    def valid_position(self, piece):
        for x, y in piece.get_coords():
            if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
                return False
            if y >= 0 and self.grid[y, x]:
                return False
        return True

    def place_piece(self, piece):
        for x, y in piece.get_coords():
            if y >= 0:
                self.grid[y, x] = 1
                self.colors[y, x] = piece.type

    def clear_lines(self):
        lines_to_clear = [i for i, row in enumerate(self.grid) if all(row)]
        for i in lines_to_clear:
            self.grid = np.delete(self.grid, i, axis=0)
            self.grid = np.vstack([np.zeros((1, GRID_WIDTH), dtype=np.uint8), self.grid])
            self.colors = np.delete(self.colors, i, axis=0)
            self.colors = np.vstack([np.full((1, GRID_WIDTH), "", dtype=object), self.colors])
        return len(lines_to_clear)

    def is_game_over(self):
        return np.any(self.grid[0])

    def get_state(self):
        # Useful for analytics: returns a copy of the current grid
        return np.array(self.grid), np.array(self.colors)