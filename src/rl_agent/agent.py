from __future__ import annotations

from typing import Sequence

import torch

from rl_agent.strategy import ActionSelectionStrategy, EpsilonGreedyStrategy
from rl_agent.q_network import QNetwork


class QLearningAgent:
    """
    RL agent that can select actions from Q-values produced by a network.
    """

    def __init__(
        self,
        q_network: QNetwork | None = None,
        strategy: ActionSelectionStrategy | None = None,
        device: str = "cpu",
    ) -> None:
        self.q_network = q_network
        self.strategy = strategy if strategy is not None else EpsilonGreedyStrategy(epsilon=0.1)
        self.device = device

    def select_action(self, q_values: Sequence[float]) -> int:
        return self.strategy.select_action(q_values)

    def select_action_from_state(self, encoded_state: Sequence[float]) -> int:
        if self.q_network is None:
            raise RuntimeError("Q-network is not initialized for this agent.")

        state_tensor = torch.tensor(encoded_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0).tolist()

        return self.select_action(q_values)