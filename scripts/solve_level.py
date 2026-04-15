from amaze_ai.config import ACTION_NAMES
from amaze_ai.env import AmazeEnv
from amaze_ai.io import load_level
from amaze_ai.generator import make_test_level
from amaze_ai.renderer import render_ascii
from amaze_ai.solver import solve_bfs

def main() -> None:
    grid, start = load_level("data/levels/test_level.json")
    env = AmazeEnv(grid, start)
    state = env.reset()

    print("Initial board:")
    print(render_ascii(env, state))
    print()

    solution = solve_bfs(env)

    if solution is None:
        print("No solution found.")
        return
    
    print("Solution!")
    print([ACTION_NAMES[a] for a in solution])
    print()

    for action in solution:
        result = env.step(state, action)
        state = result.state
        print(f"Move: {ACTION_NAMES[action]}")
        print(render_ascii(env, state))
        print()

    print("Solved:", env.is_goal(state))

if __name__ == "__main__":
    main()