import pytest

from rl_agent.env_wrapper import AmazeRLWrapper


def test_step_raises_before_reset():
    env = AmazeRLWrapper()

    with pytest.raises(RuntimeError):
        env.step(0)


def test_reset_initializes_environment_and_state():
    env = AmazeRLWrapper(rows=5, cols=5, max_steps=10)

    state = env.reset()

    assert env.env is not None
    assert env.state is not None
    assert env.steps_taken == 0
    assert state == env.state


def test_step_returns_valid_rl_tuple_and_increments_steps():
    env = AmazeRLWrapper(rows=5, cols=5, max_steps=10)
    env.reset()

    next_state, reward, done, info = env.step(0)

    assert next_state is not None
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    assert "new_tiles_painted" in info
    assert "steps_taken" in info
    assert "solved" in info

    assert env.steps_taken == 1


def test_done_becomes_true_when_max_steps_reached():
    env = AmazeRLWrapper(rows=5, cols=5, max_steps=1)
    env.reset()

    _, _, done, info = env.step(0)

    assert done is True
    assert info["steps_taken"] == 1