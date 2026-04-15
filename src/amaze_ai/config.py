from typing import Dict, Tuple

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = [UP, DOWN, LEFT, RIGHT]

DIRS: Dict[int, Tuple[int, int]] = {
    UP: (-1,0),
    DOWN: (1,0),
    LEFT: (0,-1),
    RIGHT: (0,1),
}

ACTION_NAMES = {
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
}