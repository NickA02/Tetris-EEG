# analytics.py
from piece import Piece
# This is a stub for online ML analytics integration.
# You can extend these functions to send data to your analytics system,
# or perform online learning, etc.

def log_move(board_state, piece: Piece, action, reward=0):
    """
    board_state: (np.array) the board grid after the move
    piece: (Piece) the current piece
    action: (str) the action taken ('left', 'right', 'rotate', 'drop', etc)
    reward: (float) reward for the action (e.g., lines cleared)
    """
    # Example: print for now, replace with analytics call
    print(f"[ANALYTICS] Action: {action}, Piece: {piece.type}, Reward: {reward}")

def on_game_over(board_state, score):
    print(f"[ANALYTICS] Game Over. Final Score: {score}")