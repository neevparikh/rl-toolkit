import logging
import queue
import time

import torch
import torch.multiprocessing as mp

from ..common.utils import construct_blank_tensors, get_logger


class TorchReplayBuffer:
    def __init__(self, max_size, transition_shapes):
        self.max_size = max_size
        self._next_idx = 0
        self._cur_len = 0

        # Shape of tensors without batch dim
        # Example: [next_obs, obs, action, reward, done]
        # We should have: [(84,84), (84,84), (1,), (1,), (1,)]
        self.transition_shapes = transition_shapes
        self._transition_storage = construct_blank_tensors(self.max_size, self.transition_shapes)

    def reset(self):
        self._next_idx = 0
        self._cur_len = 0
        self._transition_storage = construct_blank_tensors(self.max_size, self.transition_shapes)

    def __len__(self):
        return self._cur_len

    def _put_chunk(self, transition_tensors):
        # Size of batch
        size = transition_tensors[0].shape[0]

        # For each 'item' (see definition)
        for i, tensor in enumerate(transition_tensors):
            assert size == tensor.shape[0], "{}".format(tensor.shape)
            assert self.transition_shapes[i] == tensor.shape[1:], "{} != {}".format(
                    self.transition_shapes[i], tensor.shape[1:])

            # if it won't fit, it'll wrap around
            if self._next_idx + size > self.max_size:
                at_end = self.max_size - self._next_idx
                self._transition_storage[i][self._next_idx:self.max_size] = tensor[:at_end].clone()
                self._transition_storage[i][:(size - at_end)] = tensor[at_end:].clone()
            else:
                self._transition_storage[i][self._next_idx:self._next_idx + size] = tensor.clone()

        # Update index and length
        self._next_idx = (self._next_idx + size) % self.max_size
        self._cur_len = min(self.max_size, size + self._cur_len)

    def put(self, transition_tensors):
        assert len(transition_tensors) == len(self.transition_shapes), "{} != {}".format(
                len(transition_tensors), len(self.transition_shapes))

        if transition_tensors[0].shape[0] > self.max_size:
            # Split into chunks if putting more than capacity at a time
            # This is [(state_chunk1, state_chunk2), (action_chunk1, action_chunk2)]
            chunks = list(map(lambda t: torch.split(t, self.max_size, dim=0), transition_tensors))
            # This is [(state_chunk1, action_chunk1), (state_chunk2, action_chunk2)]
            chunks = list(zip(*chunks))
            for chunk in chunks:
                self._put_chunk(chunk)
        else:
            self._put_chunk(transition_tensors)

    def get(self, transition_tensors):
        assert len(transition_tensors) == len(self.transition_shapes), "{}".format(
                len(transition_tensors))

        size = transition_tensors[0].shape[0]

        if size > self._cur_len:
            raise ValueError("Batchsize requested too large, requested {}, buffer has {}".format(
                size, self._cur_len))

        # Shuffle indices, get requested
        indices = torch.randperm(self._cur_len)[:size]

        for i in range(len(self.transition_shapes)):
            transition_tensors[i].copy_(self._transition_storage[i][indices])


class AsyncTorchReplayBuffer(TorchReplayBuffer):
    def __init__(self, actor_queues, learner_conn, max_size, transition_shapes):
        super(AsyncTorchReplayBuffer, self).__init__(max_size, transition_shapes)
        self.actor_queues = actor_queues
        self.learner_conn = learner_conn
        self.exit = mp.Event()

    def shutdown(self):
        self.exit.set()


def async_buffer_run(buf):
    buf.logger = get_logger()
    buf.actor_queues = [[False, aq] for aq in buf.actor_queues]
    while not buf.exit.is_set():
        if buf.learner_conn.poll(timeout=0.5):
            learner_running, transition_tensors = buf.learner_conn.recv()

            if learner_running:
                if len(buf) >= transition_tensors[0].shape[0]:
                    buf.logger.debug("Learner request, sending batch")
                    buf.get(transition_tensors)
                    del transition_tensors
                    buf.learner_conn.send(True)
                else:
                    del transition_tensors
                    buf.learner_conn.send(False)
            else:
                buf.shutdown()
        else:
            buf.logger.info("No learner request")

        for i in range(len(buf.actor_queues)):
            actor_exited, actor_queue = buf.actor_queues[i]
            if not actor_exited:
                try:
                    actor_running, transition_tensors = actor_queue.get_nowait()
                    if actor_running:
                        buf.put(transition_tensors)
                        del transition_tensors
                    else:
                        buf.actor_queues[i][0] = True
                except queue.Empty:
                    buf.logger.debug("No actor_{} data".format(i))
