from amaze_ai.env import AmazeEnv
from amaze_ai.config import RIGHT, DOWN, LEFT, UP


def make_env_test_level():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    start = (0, 0)
    return grid, start


def test_reset_paints_start():
    grid, start = make_env_test_level()
    env = AmazeEnv(grid, start)
    state = env.reset()

    assert state.player == start
    assert state.painted_mask != 0


def test_step_moves_right():
    grid, start = make_env_test_level()
    env = AmazeEnv(grid, start)
    state = env.reset()

    result = env.step(state, RIGHT)
    assert result.state.player == (0, 2)


def test_full_board_can_be_solved_manually():
    grid, start = make_env_test_level()
    env = AmazeEnv(grid, start)
    state = env.reset()

    for action in [RIGHT, DOWN, LEFT, UP]:
        state = env.step(state, action).state

    assert env.is_goal(state)