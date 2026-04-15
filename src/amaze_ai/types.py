from dataclasses import dataclass
from typing import Tuple

Position = Tuple[int, int]

@dataclass(frozen=True)
class State:
    player: Position
    painted_mask: int
