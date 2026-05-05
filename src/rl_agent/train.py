from __future__ import annotations

from rl_agent.agent import QLearningAgent
from rl_agent.factory import EnvironmentFactory


"""Initial RL training entry point for Milestone 1."""


def main() -> None:
    env = EnvironmentFactory.create_easy_env()
    agent = QLearningAgent()

    state = env.reset()
    print("Initial state:", state)

    done = False
    step_count = 0

    while not done and step_count < 5:
        # Placeholder Q-values for now
        q_values = [0.0, 0.0, 0.0, 0.0]

        action = agent.select_action(q_values)
        next_state, reward, done, info = env.step(action)

        print(f"Step {step_count + 1}")
        print("Action:", action)
        print("Next state:", next_state)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)
        print("-" * 40)

        state = next_state
        step_count += 1


if __name__ == "__main__":
    main()