from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Simple feedforward network for Q-value prediction.
    Input: encoded state vector
    Output: 4 Q-values (up, down, left, right)
    """

    def __init__(self, input_size: int, hidden_size: int = 128, output_size: int = 4) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)