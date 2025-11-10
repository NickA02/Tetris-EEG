# settings.py
# Window size
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 800

# Board settings
GRID_WIDTH = 10
GRID_HEIGHT = 20
BLOCK_SIZE = 40



# Colors (R, G, B)
COLORS = {
    "I": (0, 255, 255),
    "J": (0, 0, 255),
    "L": (255, 127, 0),
    "O": (255, 255, 0),
    "S": (0, 255, 0),
    "T": (128, 0, 128),
    "Z": (255, 0, 0),
    "bg": (20, 20, 20),
    "grid": (50, 50, 50),
    "border": (200, 200, 200),
    "text": (255, 255, 255),
    "garbage": (120, 120, 120),
}

# Affect integration
AFFECT_HOST = "127.0.0.1"
AFFECT_PORT = 5555
AFFECT_VALUE_MAX = 5.0  # adjust to 9.0 when feeding raw DEAP ratings (1-9 scale)

# Difficulty dimension toggles
ENABLE_FALL_SPEED_ADJUST = True
ENABLE_GARBAGE = False
ENABLE_PIECE_BIAS = False
ENABLE_HOLD_CONTROL = True
ENABLE_PREVIEW_CONTROL = False

# Overlay display
SHOW_DIFFICULTY_OVERLAY = True