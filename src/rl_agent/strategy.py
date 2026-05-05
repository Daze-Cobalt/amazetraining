from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Sequence


"""Action-selection strategies for the RL agent."""


class ActionSelectionStrategy(ABC):
    """Base class for action-selection behavior"""

    @abstractmethod
    def select_action(self, q_values: Sequence[float]) -> int:
        """Return the chosen action index"""
        raise NotImplementedError
    
class RandomStrategy(ActionSelectionStrategy):
    """Always chooses a random action"""

    def select_action(self, q_values: Sequence[float]) -> int:
        if len(q_values) == 0:
            raise ValueError("q_values cannot be empty")
        return random.randrange(len(q_values))
    
class GreedyStrategy(ActionSelectionStrategy):
    """Always chooses the action with the highest Q-value"""

    def select_action(self, q_values: Sequence[float]) -> int:
        if len(q_values) == 0:
            raise ValueError("q_values cannot be empty")
        return max(range(len(q_values)), key=lambda i: q_values[i])
    
class EpsilonGreedyStrategy(ActionSelectionStrategy):
    """
    Chooses a random action with probability epsilon
    Otherwise chooses the greedy action
    """

    def __init__(self, epsilon: float = 0.1) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0.0 and 1.0")
        self.epsilon = epsilon
        self._random_strategy = RandomStrategy()
        self._greedy_strategy = GreedyStrategy()

    def select_action(self, q_values: Sequence[float]) -> int:
        if random.random() < self.epsilon:
            return self._random_strategy.select_action(q_values)
        return self._greedy_strategy.select_action(q_values)