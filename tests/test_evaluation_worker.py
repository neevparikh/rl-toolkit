import time

import torch
import torch.multiprocessing as mp

from rl.common.initialize_env import initialize_environment
from rl.common.utils import tensor, append_timestamp, reset_seeds, get_logger
from rl.common.parsers import async_dqn_parser
from rl.distributed_rl.agent import AsyncDQN_agent
from rl.distributed_rl.actor_worker import Actor, actor_run
from rl.distributed_rl.learner_worker import Learner, learner_run
from rl.distributed_rl.evaluation_worker import Evaluator, evaluator_run
from rl.common.async_replay_buffer import AsyncTorchReplayBuffer, TorchReplayBuffer,\
        async_buffer_run

if __name__ == "__main__":
    mp.set_start_method('forkserver')
    logger = get_logger()
    args = async_dqn_parser.parse_args()

    # Set seeds
    reset_seeds(args.seed)

    # Initialize envs
    env, test_env = initialize_environment(args)

    # Check if GPU can be used and was asked for
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

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
        "log_filename": None,
        "episodes_per_eval": args.episodes_per_eval,
    }
    agent = AsyncDQN_agent(**agent_args)
    ep = mp.Process(name='evaluator', target=evaluator_run, args=(agent.evaluator,))
    ep.start()
    while True:
        sd_cpu = {k: v.cpu() for k, v in agent.learner.online.state_dict().items()}
        agent.learner.evaluator_queue.put((True, 0, sd_cpu))
        time.sleep(3)
