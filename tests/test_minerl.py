import json

import pytest
from pyvirtualdisplay import Display

disp = Display(backend='xvfb', visible=False).start()


@pytest.fixture
def test_env():
    import gym
    import minerl
    env = gym.make('MineRLTreechop-v0')
    env.reset()
    return env


def test_ground_state(test_env):
    next_obs, reward, done, info = test_env.step(test_env.action_space.sample())
    info = json.loads(info)
    assert 'ground_state' in info
    assert len(info['ground_state']) == 289
