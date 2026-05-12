from __future__ import annotations

from typing import Any

from amaze_ai.env import AmazeEnv
from amaze_ai.generator import make_random_solvable_level
from amaze_ai.types import State


"""Reinforcement-learning wrapper for the Amaze puzzle environment."""


class AmazeRLWrapper:

    @property
    def grid(self):
        grid_value = self.env.grid
        if callable(grid_value):
            return grid_value()
        return grid_value
    
    def cell_to_bit(self, r: int, c: int) -> int:
        return r * self.cols + c

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        wall_prob: float = 0.2,
        min_open_cells: int = 8,
        max_steps: int = 50,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.wall_prob = wall_prob
        self.min_open_cells = min_open_cells
        self.max_steps = max_steps

        self.env: AmazeEnv | None = None
        self.state: State | None = None
        self.steps_taken = 0

    def reset(self) -> State:
        grid, start = make_random_solvable_level(
            rows=self.rows,
            cols=self.cols,
            wall_prob=self.wall_prob,
            min_open_cells=self.min_open_cells,
        )

        self.env = AmazeEnv(grid, start)
        self.state = self.env.reset()
        self.steps_taken = 0

        return self.state

    def step(self, action: int) -> tuple[State, float, bool, dict[str, Any]]:
        if self.env is None or self.state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        previous_state = self.state
        result = self.env.step(previous_state, action)
        next_state = result.state

        self.steps_taken += 1
        self.state = next_state

        reward = self._compute_reward(previous_state, next_state, result.new_tiles_painted, result.done)

        done = result.done or self.steps_taken >= self.max_steps

        info = {
            "new_tiles_painted": result.new_tiles_painted,
            "steps_taken": self.steps_taken,
            "solved": result.done,
        }

        return next_state, reward, done, info

    def _compute_reward(
        self,
        previous_state: State,
        next_state: State,
        new_tiles_painted: int,
        solved: bool,
    ) -> float:
        reward = 0.0

        # Reward progress
        reward += float(new_tiles_painted)

        # Small step cost to discourage wasting moves
        reward -= 0.1

        # Penalize no-progress moves
        if next_state == previous_state:
            reward -= 1.0

        # Big reward for solving
        if solved:
            reward += 10.0

        return reward