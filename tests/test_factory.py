from rl_agent.factory import EnvironmentFactory
from rl_agent.env_wrapper import AmazeRLWrapper


def test_create_easy_env_returns_wrapper():
    env = EnvironmentFactory.create_easy_env()

    assert isinstance(env, AmazeRLWrapper)


def test_factory_presets_have_expected_settings():
    easy_env = EnvironmentFactory.create_easy_env()
    medium_env = EnvironmentFactory.create_medium_env()
    hard_env = EnvironmentFactory.create_hard_env()

    assert easy_env.rows == 5
    assert easy_env.cols == 5
    assert easy_env.max_steps == 40

    assert medium_env.rows == 6
    assert medium_env.cols == 6
    assert medium_env.max_steps == 50

    assert hard_env.rows == 7
    assert hard_env.cols == 7
    assert hard_env.max_steps == 60