# piece.py

import numpy as np

PIECES = {
    "I": np.array([[1, 1, 1, 1]]),
    "O": np.array([[1, 1], [1, 1]]),
    "T": np.array([[0, 1, 0], [1, 1, 1]]),
    "S": np.array([[0, 1, 1], [1, 1, 0]]),
    "Z": np.array([[1, 1, 0], [0, 1, 1]]),
    "J": np.array([[1, 0, 0], [1, 1, 1]]),
    "L": np.array([[0, 0, 1], [1, 1, 1]]),
}

import random

class Piece:
    #type: str = None

    def __init__(self, type=None):
        self.type = type if type else random.choice(list(PIECES.keys()))
        self.shape = PIECES[self.type]
        self.rotation = 0  # 0-3
        self.x = 3  # Starting x position
        self.y = 0  # Starting y position

    def rotate(self, clockwise=True):
        if clockwise:
            self.shape = np.rot90(self.shape, -1)
        else:
            self.shape = np.rot90(self.shape, 1)

    def get_blocks(self):
        return self.shape

    def get_coords(self):
        """Returns list of (x, y) tuples for the blocks occupied by this piece."""
        blocks = []
        for dy, row in enumerate(self.shape):
            for dx, cell in enumerate(row):
                if cell:
                    blocks.append((self.x + dx, self.y + dy))
        return blocks

    def copy(self):
        p = Piece(self.type)
        p.shape = np.array(self.shape)
        p.rotation = self.rotation
        p.x = self.x
        p.y = self.y
        return p