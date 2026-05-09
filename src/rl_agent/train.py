from __future__ import annotations

from rl_agent.agent import QLearningAgent
from rl_agent.factory import EnvironmentFactory
from rl_agent.q_network import QNetwork
from rl_agent.state_encoder import encode_state, get_state_size


def main() -> None:
    env = EnvironmentFactory.create_easy_env()
    state = env.reset()

    input_size = get_state_size(env.env.rows, env.env.cols)
    q_network = QNetwork(input_size=input_size)
    agent = QLearningAgent(q_network=q_network)

    print("Initial state:", state)
    print("Encoded state size:", input_size)

    done = False
    step_count = 0

    while not done and step_count < 5:
        encoded_state = encode_state(env.env, env.state)
        action = agent.select_action_from_state(encoded_state)

        next_state, reward, done, info = env.step(action)

        print(f"Step {step_count + 1}")
        print("Action:", action)
        print("Next state:", next_state)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)
        print("-" * 40)

        step_count += 1


if __name__ == "__main__":
    main()