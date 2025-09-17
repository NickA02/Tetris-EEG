
from game import Game

def increase_difficulty_lines_cleared(game: Game, max_speed=100, min_speed=600):
    """Increase fall speed based on lines cleared."""
    lines = game.lines_cleared
    fall_speed = min_speed - (lines * 10)
    fall_speed = max(min(fall_speed, min_speed), max_speed)
    return fall_speed

def constant_difficulty(game: Game, speed=300):
    """Keep fall speed constant."""
    return speed

def increase_difficulty_adaptive(game: Game, max_speed=100, min_speed=600):
    """Adjust fall speed based on average isolated empty blocks and pieces placed, as well as lines cleared."""
    avg_isolated_empty = game.get_isolated_empty_avg()
    piece_factor = game.piece_count * 10
    lines = game.lines_cleared
    fall_speed = min_speed + avg_isolated_empty*50 - piece_factor - (lines * 3)
    fall_speed = max(min(fall_speed, min_speed), max_speed)
    return fall_speed

def increase_difficulty_blocks_placed(game: Game, max_speed=100, min_speed=600):
    """Increase fall speed based on number of pieces placed."""
    piece_factor = game.piece_count * 10
    fall_speed = min_speed - piece_factor
    fall_speed = max(min(fall_speed, min_speed), max_speed)
    return fall_speed