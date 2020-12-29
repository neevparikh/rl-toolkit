import pytest
import torch

from rl.common.utils import tensor


def tensorify(obs, next_obs, action, reward, done):
    tensors = map(tensor, [obs, next_obs, [action], [reward], [done]])
    tensors = map(lambda t: torch.unsqueeze(t, 0), tensors)
    tensors = list(tensors)
    return tensors


@pytest.fixture
def construct_env_buffer():
    import gym
    from rl.common.async_replay_buffer import TorchReplayBuffer

    env = gym.make('Pong-v0')
    env.reset()
    shapes = (
        torch.Size(env.observation_space.shape),
        torch.Size(env.observation_space.shape),
        torch.Size([1]),
        torch.Size([
            1,
        ]),
        torch.Size([
            1,
        ]),
    )
    buf = TorchReplayBuffer(500, shapes)
    return env, buf


@pytest.fixture
def construct_buffer():
    from rl.common.async_replay_buffer import TorchReplayBuffer

    shapes = (torch.Size([
        5,
    ]),)
    buf = TorchReplayBuffer(3, shapes)
    return buf


def test_buffer_put_get(construct_env_buffer):
    env, buf = construct_env_buffer
    obs = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)

    tensors = tensorify(obs, next_obs, action, reward, done)
    buf.put(tensors)

    t_tensors = list(map(torch.zeros_like, tensors))
    buf.get(t_tensors)
    assert all([torch.equal(tt, t) for tt, t in zip(t_tensors, tensors)])
    assert len(buf) == 1


def test_buffer_put_get_stress_test(construct_env_buffer):
    env, buf = construct_env_buffer
    step = 0
    obs = env.reset()
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    tensors = tensorify(obs, next_obs, action, reward, done)

    while step < 500:
        if done:
            obs = env.reset()
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        tensors = tensorify(obs, next_obs, action, reward, done)
        buf.put(tensors)
        step += 1
        obs = next_obs

    t_tensors = map(torch.zeros_like, tensors)
    t_tensors = map(lambda t: t.expand(32, *t.shape[1:]).clone(), t_tensors)
    t_tensors = list(t_tensors)
    buf.get(t_tensors)


def test_buffer_multi_put_overflow(construct_buffer):
    buf = construct_buffer

    ones = torch.ones(2, 5)
    buf.put([ones])

    zeros = torch.zeros(2, 5)
    buf.put([zeros])

    exp = torch.ones(3, 5)
    exp[0] *= 0
    exp[1] *= 0

    ans = torch.zeros(3, 5)

    buf.get([ans])
    ans, _ = torch.sort(ans, dim=0)

    assert torch.equal(ans, exp)


def test_buffer_multi_put_overflow_bigger_than_buflen(construct_buffer):
    buf = construct_buffer

    ones = torch.ones(2, 5)
    buf.put([ones])

    zeros = torch.zeros(6, 5)
    buf.put([zeros])

    exp = torch.zeros(3, 5)
    ans = torch.ones(3, 5)

    buf.get([ans])
    assert torch.equal(ans, exp)


def test_buffer_overflow(construct_buffer):
    buf = construct_buffer

    ones = torch.ones(1, 5)
    buf.put([ones])

    twos = torch.ones(1, 5) * 2
    buf.put([twos])

    threes = torch.ones(1, 5) * 3
    buf.put([threes])

    fours = torch.ones(1, 5) * 4
    buf.put([fours])

    ans = torch.zeros(3, 5)
    buf.get([ans])

    assert 4 in ans


def test_buffer_multi_put(construct_buffer):
    buf = construct_buffer

    ones = torch.ones(3, 5)
    ones[1] *= 2
    ones[2] *= 3
    buf.put([ones])
    ans = torch.zeros(3, 5)
    buf.get([ans])
    ans, _ = torch.sort(ans, dim=0)

    assert torch.equal(ans, ones)


def test_buffer_multi_put_bigger_than_buflen(construct_buffer):
    buf = construct_buffer

    ones = torch.ones(6, 5)
    ones[-2] *= 2
    ones[-1] *= 3
    buf.put([ones])

    ans = torch.zeros(3, 5)
    buf.get([ans])
    expected = torch.ones(3, 5)
    expected[1] *= 2
    expected[2] *= 3

    ans, _ = torch.sort(ans, dim=0)
    assert torch.equal(ans, expected)


def test_buffer_random_put_get(construct_buffer):
    buf = construct_buffer
    ans = torch.zeros(3, 5)
    t = torch.rand(2, 5)
    buf.put([t])
    for _ in range(1000):
        t = torch.rand(2, 5)
        buf.put([t])
        buf.get([ans])


def test_buffer_get_more_than_buflen(construct_buffer):
    buf = construct_buffer
    ans = torch.zeros(3, 5)
    t = torch.rand(2, 5)
    buf.put([t])
    with pytest.raises(ValueError):
        buf.get([ans])
