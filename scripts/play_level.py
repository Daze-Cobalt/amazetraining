from amaze_ai.config import ACTION_NAMES, ACTIONS
from amaze_ai.env import AmazeEnv
from amaze_ai.generator import make_test_level
from amaze_ai.renderer import render_ascii

def main() -> None:
    grid, start = make_test_level()
    env = AmazeEnv(grid, start)
    state = env.reset()

    print("Initial state:")
    print(render_ascii(env, state))
    print()

    for action in ACTIONS:
        result = env.step(state, action)
        print(f"Action: {ACTION_NAMES[action]}")
        print(render_ascii(env, result.state))
        print(f"New tiles painted: {result.new_tiles_painted}")
        print(f"Done: {result.done}")
        print("-" * 40)

if __name__ == "__main__":
    main()