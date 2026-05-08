import pytest

from rl_agent.strategy import (
    RandomStrategy,
    GreedyStrategy,
    EpsilonGreedyStrategy,
)


def test_random_strategy_returns_valid_index():
    strategy = RandomStrategy()
    q_values = [0.1, 0.2, 0.3, 0.4]

    action = strategy.select_action(q_values)

    assert action in range(len(q_values))


def test_random_strategy_raises_on_empty_q_values():
    strategy = RandomStrategy()

    with pytest.raises(ValueError):
        strategy.select_action([])


def test_greedy_strategy_picks_max_index():
    strategy = GreedyStrategy()
    q_values = [0.1, 0.9, 0.4, 0.2]

    action = strategy.select_action(q_values)

    assert action == 1


def test_greedy_strategy_raises_on_empty_q_values():
    strategy = GreedyStrategy()

    with pytest.raises(ValueError):
        strategy.select_action([])


def test_epsilon_greedy_rejects_invalid_epsilon():
    with pytest.raises(ValueError):
        EpsilonGreedyStrategy(epsilon=-0.1)

    with pytest.raises(ValueError):
        EpsilonGreedyStrategy(epsilon=1.1)