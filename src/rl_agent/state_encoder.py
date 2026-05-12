from __future__ import annotations

from typing import List

from amaze_ai.env import AmazeEnv
from amaze_ai.types import State


def encode_state(env: AmazeEnv, state: State) -> List[float]:
    """
    Encode the puzle state into a flat numeric vector

    Vector layout:
    1. Board layout: 1.0 for open cell, 0.0 for wall
    2. Painted mask: 1.0 if painted, 0.0 otherwise
    3. Player position: one-hot vector over all other board cells
    """
    rows = env.rows
    cols = env.cols
    total_cells = rows * cols

    board_features: List[float] = []
    painted_features: List[float] = []
    player_features: List[float] = [0.0] * total_cells

    # Board layout
    for r in range(rows):
        for c in range(cols):
            board_features.append(1.0 if env.grid[r][c] == 1 else 0.0)

    # Painted cells
    for r in range(rows):
        for c in range(cols):
            if env.grid[r][c] == 0:
                painted_features.append(0.0)
            else:
                bit = env.cell_to_bit(r, c)
                painted = (state.painted_mask >> bit) & 1
                painted_features.append(float(painted))

    # Player one-hot
    pr, pc = state.player
    player_index = pr * cols + pc
    player_features[player_index] = 1.0

    return board_features + painted_features + player_features

def get_state_size(rows: int, cols: int) -> int:
    """
    Total encoded state size:
    board + painted + player
    """
    total_cells = rows * cols
    return total_cells * 3