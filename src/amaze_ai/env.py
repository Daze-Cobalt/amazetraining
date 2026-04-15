from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple

from amaze_ai.types import State, Position
from amaze_ai.config import ACTIONS, DIRS

Grid = List[List[int]]
# 0 = blocked
# 1 = paintable

@dataclass
class StepResult:
    state: State
    new_tiles_painted: int
    done: bool

class AmazeEnv:
    def __init__(self, grid: Grid, start: Position):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self.start = start

        self.cell_to_bit: Dict[Position, int] = {}
        self.bit_to_cell: Dict[int, Position] = {}
        self.goal_mask = 0

        self._index_cells()

    def _index_cells(self) -> None:
        bit_index = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 1:
                    self.cell_to_bit[(r, c)] = bit_index
                    self.bit_to_cell[bit_index] = (r, c)
                    self.goal_mask |= (1 << bit_index)
                    bit_index += 1

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols
    
    def _is_paintable(self, r: int, c: int) -> bool:
        return self._in_bounds(r, c) and self.grid[r][c] == 1

    def _paint_bit(self, mask: int, pos: Position) -> int:
        bit = self.cell_to_bit[pos]
        return mask | (1 << bit)
    
    def reset(self) -> State:
        mask = 0
        mask = self._paint_bit(mask, self.start)
        return State(player=self.start, painted_mask=mask)
    
    def is_goal(self, state: State) -> bool:
        return state.painted_mask == self.goal_mask
    
    def step(self, state: State, action: int) -> StepResult:
        dr, dc = DIRS[action]
        r, c = state.player
        new_mask = state.painted_mask
        before = new_mask

        while self._is_paintable(r + dr, c + dc):
            r += dr
            c += dc
            new_mask = self._paint_bit(new_mask, (r, c))

        new_state = State(player=(r, c), painted_mask=new_mask)
        new_tiles = (new_mask ^ before).bit_count()
        done = self.is_goal(new_state)
        return StepResult(state=new_state, new_tiles_painted=new_tiles, done=done)