from __future__ import annotations

import os
import json
import torch

from rl_agent.agent import QLearningAgent
from rl_agent.factory import EnvironmentFactory
from rl_agent.q_network import QNetwork
from rl_agent.state_encoder import encode_state, get_state_size


def main() -> None:
    env = EnvironmentFactory.create_easy_env()
    state = env.reset()

    input_size = get_state_size(env.env.rows, env.env.cols)
    q_network = QNetwork(input_size=input_size)

    model_path = "data/rl_models/q_network.pt"

    if os.path.exists(model_path):
        q_network.load_state_dict(torch.load(model_path))
        print("Loaded existing model.")

    agent = QLearningAgent(q_network=q_network)

    print("Initial state:", state)
    print("Encoded state size:", input_size)

    episodes = 100
    max_steps = 200

    total_steps = 0

    for episode in range(episodes):
        state = env.reset()

        done = False
        step_count = 0

        while not done and step_count < max_steps:
            encoded_state = encode_state(env, env.state)

            action = agent.select_action_from_state(encoded_state)

            next_state, reward, done, info = env.step(action)

            step_count += 1
            total_steps += 1

        print(f"Episode {episode + 1}/{episodes} finished in {step_count} steps")

    os.makedirs("data/rl_models", exist_ok=True)
    os.makedirs("data/rl_logs", exist_ok=True)

    torch.save(q_network.state_dict(), "data/rl_models/q_network.pt")
    print("Saved model to data/rl_models/q_network.pt")

    results = {
        "episodes": episodes,
        "max_steps_per_episode": max_steps,
        "total_steps": total_steps,
        "last_episode_steps": step_count,
        "final_done": done,
        "board_size": f"{env.rows}x{env.cols}",
        "model_path": model_path,
    }

    with open("data/rl_logs/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Saved training results to data/rl_logs/training_results.json")


if __name__ == "__main__":
    main()