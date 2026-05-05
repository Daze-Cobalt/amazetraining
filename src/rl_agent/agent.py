from __future__ import annotations

from typing import Sequence

from rl_agent.strategy import ActionSelectionStrategy, EpsilonGreedyStrategy


class QLearningAgent:
    """
    Minimal RL agent skeleton.
    For Milestone 1, this only handles action selection.
    """

    def __init__(self, strategy: ActionSelectionStrategy | None = None) -> None:
        self.strategy = strategy if strategy is not None else EpsilonGreedyStrategy(epsilon=0.1)

    def select_action(self, q_values: Sequence[float]) -> int:
        return self.strategy.select_action(q_values)