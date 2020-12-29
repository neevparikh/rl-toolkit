from random import random
import logging
import time

import torch
import torch.multiprocessing as mp

from ..common.utils import construct_blank_tensors, tensor, get_logger


class Actor:
    def __init__(
        self,
        actor_id,
        actor_done,
        max_steps_per_actor,
        epsilon_decay_length,
        final_epsilon_value,
        warmup_period,
        policy_update_steps,
        num_send_transitions,
        env_fn,
        policy_fn,
        buffer_fn,
        policy_queue,
        transition_shapes,
        writer=None,
    ):

        self.actor_id = actor_id
        self.exit = mp.Event()

        self.writer = writer
        self.actor_done = actor_done

        # Construct env, policy, and local buffer
        self.env = env_fn()
        self.max_steps_per_actor = max_steps_per_actor
        self.policy_net = policy_fn()
        self.local_buffer = buffer_fn(num_send_transitions, transition_shapes)
        self.transition_shapes = transition_shapes

        self.epsilon_decay_length = epsilon_decay_length
        self.final_epsilon_value = final_epsilon_value
        self.warmup_period = warmup_period
        self.epsilon = 1

        # queries for policy network update every these many steps
        self.policy_update_steps = policy_update_steps
        # send transitions to central buffer once these many have been collected
        self.num_send_transitions = num_send_transitions

        # Queue to receive updated policy parameters
        self.policy_queue = policy_queue

        # Queue to send transitions to central buffer
        self.buffer_queue = mp.Queue()
        self.transition_tensors = construct_blank_tensors(self.num_send_transitions,
                                                          self.transition_shapes)

        # Set local state variables
        self.local_steps = 0
        self.state = self.env.reset()

    def act(self):
        # Compute action, step in env, store locally
        action = self.policy_net.act(self.state, self.epsilon)
        next_state, reward, done, info = self.env.step(action)

        # Store the experience locally
        transition_tensors = [self.state, next_state, tensor([action]), reward, done]
        transition_tensors = list(map(lambda t: t.unsqueeze(0), transition_tensors))
        self.local_buffer.put(transition_tensors)
        self.state = next_state

        # reset environment if episode terminates
        if done:
            self.state = self.env.reset()

        # Update local steps
        self.local_steps += 1
        self.update_epsilon()

    def update_epsilon(self):
        if self.local_steps <= self.warmup_period:
            self.epsilon = 1
        else:
            current_epsilon_decay = 1 - (1 - self.final_epsilon_value) * (
                self.local_steps - self.warmup_period) / self.epsilon_decay_length

            self.epsilon = max(self.final_epsilon_value, current_epsilon_decay)

        if self.local_steps % 10000 == 0:
            self.logger.info('{} | epsilon - {}'.format(self.local_steps, self.epsilon))
        if self.writer is not None:
            self.writer.add_scalar('actor_{}/epsilon'.format(self.actor_id),
                                   self.epsilon,
                                   self.local_steps)

    def send_transitions(self):
        # Get the latest transitions
        self.local_buffer.get(self.transition_tensors)
        self.buffer_queue.put((True, self.transition_tensors))

    def get_policy_params(self):
        learner_running, state_dict = self.policy_queue.get()
        if learner_running:
            self.policy_net.load_state_dict(state_dict)
            del state_dict
        else:
            self.shutdown()

    def shutdown(self):
        self.buffer_queue.put((False, None))
        self.logger.info('{} | epsilon - {}'.format(self.local_steps, self.epsilon))
        self.exit.set()


def actor_run(actor):
    actor.logger = get_logger()
    actor.logger.info("Getting initial policy")
    actor.get_policy_params()

    while actor.local_steps < actor.max_steps_per_actor:
        actor.act()
        if len(actor.local_buffer) >= actor.num_send_transitions:
            actor.send_transitions()
            actor.local_buffer.reset()
        if actor.local_steps % actor.policy_update_steps == 0:
            actor.get_policy_params()
    with actor.actor_done.get_lock():
        actor.actor_done.value += 1
