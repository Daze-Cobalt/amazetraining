from __future__ import annotations

from rl_agent.env_wrapper import AmazeRLWrapper


"""Factory methods for creating RL environments with preset difficulty settings."""


class EnvironmentFactory:

    @staticmethod
    def create_easy_env() -> AmazeRLWrapper:
        return AmazeRLWrapper(
            rows=5,
            cols=5,
            wall_prob=0.18,
            min_open_cells=8,
            max_steps=40,
        )

    @staticmethod
    def create_medium_env() -> AmazeRLWrapper:
        return AmazeRLWrapper(
            rows=6,
            cols=6,
            wall_prob=0.20,
            min_open_cells=8,
            max_steps=50,
        )

    @staticmethod
    def create_hard_env() -> AmazeRLWrapper:
        return AmazeRLWrapper(
            rows=7,
            cols=7,
            wall_prob=0.22,
            min_open_cells=8,
            max_steps=60,
        )