from rl_agent.agent import QLearningAgent
from rl_agent.strategy import EpsilonGreedyStrategy


class DummyStrategy:
    def __init__(self):
        self.called = False

    def select_action(self, q_values):
        self.called = True
        return 2


def test_agent_uses_default_strategy():
    agent = QLearningAgent()

    assert isinstance(agent.strategy, EpsilonGreedyStrategy)


def test_agent_select_action_delegates_to_strategy():
    strategy = DummyStrategy()
    agent = QLearningAgent(strategy=strategy)

    action = agent.select_action([0.0, 0.0, 0.0, 0.0])

    assert strategy.called is True
    assert action == 2