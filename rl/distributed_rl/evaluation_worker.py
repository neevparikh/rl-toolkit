import logging
import time
from queue import Empty
from random import random

import torch
import torch.multiprocessing as mp

from ..common.utils import construct_blank_tensors, tensor, get_logger


class Evaluator:
    def __init__(
        self,
        evaluator_id,
        env_fn,
        policy_fn,
        policy_queue,
        log_filename,
        episodes_per_eval,
        writer=None,
    ):

        self.evaluator_id = evaluator_id
        self.exit = mp.Event()
        self.writer = writer

        self.log_filename = log_filename
        self.st = None

        # Construct env, policy, and local buffer
        self.env = env_fn()
        self.policy_net = policy_fn()
        self.episodes_per_eval = episodes_per_eval

        # Queue to receive updated policy parameters
        self.policy_queue = policy_queue

    def test_policy(self, learner_steps):
        with torch.no_grad():
            # Reset environment
            cumulative_reward = 0

            for _ in range(self.episodes_per_eval):
                state = self.env.reset()

                done = False

                # Test episode loop
                while not done:
                    action = self.policy_net.act(state, 0)

                    state, reward, done, _ = self.env.step(action)

                    # Update reward
                    cumulative_reward += reward

            eval_reward = cumulative_reward / self.episodes_per_eval

            self.logger.info("{} | reward - {}".format(learner_steps, eval_reward.item()))

            # Logging
            if self.writer:
                self.writer.add_scalar('evaluator_{}/policy_reward'.format(self.evaluator_id),
                                       learner_steps,
                                       eval_reward.item())
            if self.log_filename:
                with open(self.log_filename, "a") as f:
                    f.write("{},{},{}\n".format(learner_steps,
                                                eval_reward.item(),
                                                time.time() - self.st))

    def shutdown(self):
        self.exit.set()


def evaluator_run(evaluator):
    evaluator.logger = get_logger()
    evaluator.logger.info("Waiting for eval request")
    evaluator.st = time.time()

    while not evaluator.exit.is_set():
        try:
            learner_running, steps, state_dict = evaluator.policy_queue.get(timeout=30)
            if learner_running:
                evaluator.policy_net.load_state_dict(state_dict)
                del state_dict
                evaluator.test_policy(steps)
            else:
                evaluator.shutdown()
        except Empty:
            evaluator.logger.warning('No learner eval request in 30s')
