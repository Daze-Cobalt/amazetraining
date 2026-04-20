from amaze_ai.config import ACTIONS
from rl_agent.env_wrapper import AmazeRLWrapper


# Initial RL training entry point for Milestone 1.
# Current version is a runnable skeleton used to verify modular structure and environment interaction.


def main() -> None:
    env = AmazeRLWrapper(rows=5, cols=5, wall_prob=0.2, max_steps=20)

    state = env.reset()
    print("Initial state:", state)

    done = False
    step_count = 0

    while not done and step_count < 5:
        action = ACTIONS[step_count % len(ACTIONS)]
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