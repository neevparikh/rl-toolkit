import os
import json

import torch
import torch.multiprocessing as mp
import gym

from .agent import AsyncDQN_agent
from ..common.parsers import async_dqn_parser
from ..common.utils import append_timestamp, reset_seeds, get_logger
from ..common.initialize_env import initialize_environment

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    logger = get_logger()
    args = async_dqn_parser.parse_args()

    # Set seeds
    reset_seeds(args.seed)

    # Initialize envs
    env, test_env = initialize_environment(args)

    if type(env.action_space) != gym.spaces.Discrete:
        raise NotImplementedError("DQN for continuous action_spaces hasn't been implemented")

    # Check if GPU can be used and was asked for
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)

    # Logging via csv
    if args.output_path:
        base_filename = os.path.join(args.output_path, args.run_tag)
        os.makedirs(base_filename, exist_ok=True)
        log_filename = os.path.join(base_filename, 'reward.csv')
        with open(log_filename, "w") as f:
            f.write("steps,reward,runtime\n")
        with open(os.path.join(base_filename, 'params.json'), 'w') as fp:
            param_dict = vars(args).copy()
            del param_dict['output_path']
            del param_dict['model_path']
            json.dump(param_dict, fp)
    else:
        log_filename = None

    agent_args = {
        "device": device,
        "state_space": env.observation_space,
        "action_space": env.action_space,
        "num_actions": env.action_space.n,
        "lr": args.lr,
        "target_moving_average": args.target_moving_average,
        "gamma": args.gamma,
        "replay_buffer_size": args.replay_buffer_size,
        "epsilon_decay_length": args.epsilon_decay_length,
        "final_epsilon_value": args.final_epsilon_value,
        "warmup_period": args.warmup_period,
        "double_DQN": not (args.vanilla_DQN),
        "model_type": args.model_type,
        "model_shape": args.model_shape,
        "num_frames": args.num_frames,
        "num_actors": args.num_actors,
        "num_learners": args.num_learners,
        "batchsize": args.batchsize,
        "max_steps": args.max_steps,
        "min_gradient_steps": args.min_gradient_steps,
        "policy_update_steps": args.policy_update_steps,
        "num_send_transitions": args.num_send_transitions,
        "env_fn": lambda: initialize_environment(args)[0],
        "test_env_fn": lambda: initialize_environment(args)[1],
        "log_filename": log_filename,
        "episodes_per_eval": args.episodes_per_eval,
    }
    agent = AsyncDQN_agent(**agent_args)
    agent.run()
