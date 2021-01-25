import torch
import time
import timeit
import numpy as np

from rl.common.initialize_env import initialize_environment
from rl.common.utils import tensor, construct_blank_tensors
from rl.common.async_replay_buffer import TorchReplayBuffer
from rl.common.parsers import async_dqn_parser

args = async_dqn_parser.parse_args()
env, test_env = initialize_environment(args)
state_space = env.observation_space
transition_shapes = [
    torch.Size(state_space.shape),  # state
    torch.Size(state_space.shape),  # next_state
    torch.Size([1]),  # action
    torch.Size([1]),  # reward
    torch.Size([1]),  # done
]
buf = TorchReplayBuffer(args.replay_buffer_size, transition_shapes)
tmp_buf = TorchReplayBuffer(args.replay_buffer_size, transition_shapes)

tmp = construct_blank_tensors(args.replay_buffer_size, transition_shapes)
tmp_buf.put(tmp)

# step = 0
# while step < 1e3:
#     obs = env.reset()
#     done = False
#     while not done:
#         act = env.action_space.sample()
#         n_obs, rew, done, _ = env.step(act)
#         tmp_buf.put([obs.unsqueeze(0), n_obs.unsqueeze(0), tensor([act]).unsqueeze(0),
#             rew.unsqueeze(0), done.unsqueeze(0)])
#         obs = n_obs
#         step += 1

size = 512
test = construct_blank_tensors(size, transition_shapes)


def get():
    tmp_buf.get(test)


def put():
    buf.put(test)


n = 100
print(timeit.timeit('get()', globals=globals(), number=n) / n)
print(timeit.timeit('put()', globals=globals(), number=n) / n)

print(timeit.timeit('get()', globals=globals(), number=n) / (n * size))
print(timeit.timeit('put()', globals=globals(), number=n) / (n * size))
