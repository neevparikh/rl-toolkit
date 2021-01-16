import torch
import time
import numpy as np

from rl.common.initialize_env import initialize_environment
from rl.common.utils import tensor
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

step = 0
while step < 1e5:
    obs = env.reset()
    done = False
    cr = 0
    while not done:
        act = env.action_space.sample()
        n_obs, rew, done, _ = env.step(act)
        # print(n_obs, act, rew)
        # buf.put([obs.unsqueeze(0), n_obs.unsqueeze(0), tensor([act]).unsqueeze(0),
        #     rew.unsqueeze(0), done.unsqueeze(0)])
        obs = n_obs
        cr += rew
        step += 1
    print(cr)
