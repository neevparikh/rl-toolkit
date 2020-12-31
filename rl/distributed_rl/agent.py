import torch
import torch.multiprocessing as mp

from ..common.async_replay_buffer import AsyncTorchReplayBuffer, TorchReplayBuffer,\
        async_buffer_run
from ..common.utils import tensor
from .actor_worker import Actor, actor_run
from .learner_worker import Learner, learner_run
from .evaluation_worker import Evaluator, evaluator_run


class AsyncDQN_agent:
    def __init__(self,
                 device,
                 state_space,
                 action_space,
                 num_actions,
                 lr,
                 target_moving_average,
                 gamma,
                 replay_buffer_size,
                 epsilon_decay_length,
                 final_epsilon_value,
                 warmup_period,
                 double_DQN,
                 num_actors,
                 batchsize,
                 max_steps,
                 min_gradient_steps,
                 policy_update_steps,
                 num_send_transitions,
                 env_fn,
                 test_env_fn,
                 log_filename,
                 episodes_per_eval,
                 num_learners,
                 model_type="mlp",
                 model_shape=None,
                 num_frames=None):
        """ Defining Async DQN agent """
        self.num_actors = num_actors
        self.num_learners = num_learners
        self.transition_shapes = [
            torch.Size(state_space.shape),  # state
            torch.Size(state_space.shape),  # next_state
            torch.Size([1]),  # action
            torch.Size([1]),  # reward
            torch.Size([1]),  # done
        ]

        learner_conn, buffer_conn = mp.Pipe()
        self.actor_done = mp.Value('i', 0)
        self.actor_steps = tensor([0] * self.num_actors)
        self.actor_steps.share_memory_()
        self.actor_steps_lock = mp.Lock()

        self.learner = Learner(
            num_workers=num_learners,
            buffer_conn=buffer_conn,
            device=device,
            batchsize=batchsize,
            transition_shapes=self.transition_shapes,
            state_space=state_space,
            action_space=action_space,
            num_actions=num_actions,
            min_gradient_steps=min_gradient_steps,
            lr=lr,
            target_moving_average=target_moving_average,
            gamma=gamma,
            double_DQN=double_DQN,
            model_type=model_type,
            model_shape=model_shape,
            num_frames=num_frames,
            num_actors_per_learner=num_actors,
        )
        self.evaluator = Evaluator(
            evaluator_id=0,
            env_fn=test_env_fn,
            policy_fn=self.learner.get_policy_fn(torch.device('cpu')),
            policy_queue=self.learner.evaluator_queue,
            log_filename=log_filename,
            episodes_per_eval=episodes_per_eval,
        )
        self.actors = [
            Actor(
                actor_id=i,
                policy_update_steps=policy_update_steps,
                num_send_transitions=num_send_transitions,
                actor_steps=self.actor_steps,
                steps_lock=self.actor_steps_lock,
                env_fn=env_fn,
                actor_done=self.actor_done,
                max_steps_per_actor=max_steps // num_actors,
                policy_fn=self.learner.get_policy_fn(torch.device('cpu')),
                buffer_fn=TorchReplayBuffer,
                policy_queue=self.learner.actor_queues[i],
                transition_shapes=self.transition_shapes,
                epsilon_decay_length=epsilon_decay_length,
                final_epsilon_value=final_epsilon_value,
                warmup_period=warmup_period,
            ) for i in range(self.num_actors)
        ]

        self.replay_buffer = AsyncTorchReplayBuffer(
            actor_queues=[actor.buffer_queue for actor in self.actors],
            learner_conn=learner_conn,
            max_size=replay_buffer_size,
            transition_shapes=self.transition_shapes,
        )

    def run(self):
        lp = mp.Process(name='learner', target=learner_run, args=(self.learner,))
        aps = [
            mp.Process(name='actor_{}'.format(i), target=actor_run, args=(actor,)) for i,
            actor in enumerate(self.actors)
        ]
        bp = mp.Process(name='buffer', target=async_buffer_run, args=(self.replay_buffer,))
        ep = mp.Process(name='evaluator', target=evaluator_run, args=(self.evaluator,))
        lp.start()
        for ap in aps:
            ap.start()
        bp.start()
        ep.start()
        while self.actor_done.value != self.num_actors:
            continue
        self.learner.shutdown()
