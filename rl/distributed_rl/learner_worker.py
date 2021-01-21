import time
import logging

import torch
import torch.multiprocessing as mp

from ..model_free.dqn.model import DQN_MLP_model, DQN_CNN_model
from ..common.utils import construct_blank_tensors, soft_update, tensor, get_logger


class Learner:
    def __init__(
        self,
        buffer_conn,
        device,
        batchsize,
        transition_shapes,
        state_space,
        action_space,
        num_actions,
        lr,
        target_moving_average,
        gamma,
        min_gradient_steps,
        double_DQN,
        model_type,
        model_shape,
        num_frames,
        num_actors_per_learner,
        num_workers,
        writer=None,
        gradient_clip=None,
    ):

        self.writer = writer
        self.device = device
        self.lr = lr

        self.num_frames = num_frames
        self.model_type = model_type
        self.model_shape = model_shape
        self.state_space = state_space
        self.action_space = action_space
        self.double_DQN = double_DQN
        self.batchsize = batchsize
        self.num_frames = num_frames
        self.num_actions = num_actions
        self.num_workers = num_workers

        policy_fn = self.get_policy_fn(self.device)
        self.online = policy_fn()
        self.target = policy_fn()

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.gamma = gamma
        self.gradient_clip = gradient_clip
        self.target_moving_average = target_moving_average

        self.transition_shapes = transition_shapes
        self.num_actors_per_learner = num_actors_per_learner
        self.buffer_conn = buffer_conn
        self.evaluator_queue = mp.Queue()
        self.actor_queues = [mp.Queue() for _ in range(self.num_actors_per_learner)]

        self.minibatch = construct_blank_tensors(self.batchsize, self.transition_shapes)

        self.steps = tensor([0] * self.num_workers)
        self.steps.share_memory_()
        self.fps_alpha = 1
        self.min_gradient_steps = min_gradient_steps

        self.lock = mp.Lock()
        self.steps_lock = mp.Lock()
        self.exit = mp.Event()
        self.ready = mp.Event()

    def get_policy_fn(self, device):
        if self.model_type == "mlp":
            policy_fn = lambda: DQN_MLP_model(
                device, self.state_space, self.action_space, self.num_actions, self.model_shape)
        elif self.model_type == "cnn":
            assert self.num_frames
            policy_fn = lambda: DQN_CNN_model(device,
                                              self.state_space,
                                              self.action_space,
                                              self.num_actions,
                                              num_frames=self.num_frames)
        else:
            raise NotImplementedError(model_type)
        return policy_fn

    def loss_func(self, minibatch):
        state_tensor, next_state_tensor, action_tensor, reward_tensor, done_tensor = minibatch

        # Get q value predictions
        q_pred_batch = self.online(state_tensor)
        q_pred_batch = q_pred_batch.gather(dim=1, index=action_tensor.long())

        with torch.no_grad():
            if self.double_DQN:
                selected_actions = self.online.argmax_over_actions(next_state_tensor).unsqueeze(1)
                q_target = self.target(next_state_tensor)
                q_target = q_target.gather(dim=1, index=selected_actions.long())
            else:
                q_target = self.target.max_over_actions(next_state_tensor.detach()).values

            q_label_batch = reward_tensor + (self.gamma) * (1 - done_tensor) * q_target

        loss = torch.nn.functional.smooth_l1_loss(q_label_batch, q_pred_batch)

        return loss

    def shutdown(self):
        self.exit.set()

    def train_batch(self, optimizer, minibatch):
        optimizer.zero_grad()

        loss = self.loss_func(minibatch)
        loss.backward()

        if self.gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.gradient_clip)

        # Update parameters
        optimizer.step()
        soft_update(self.target, self.online, self.target_moving_average)

        return loss


def train_worker(rank, learner):
    learner.online = learner.online.to(learner.device)
    learner.target = learner.target.to(learner.device)
    learner.online.share_memory()
    learner.target.share_memory()

    learner.logger = get_logger()
    local_steps = 0
    fps = 0
    if rank == 0:
        prev_total = 0
    learner.minibatch = list(map(lambda t: t.to(learner.device), learner.minibatch))
    optimizer = torch.optim.Adam(learner.online.parameters(), lr=learner.lr)
    st = time.time()

    min_steps_per_worker = learner.min_gradient_steps // learner.num_workers

    while local_steps < min_steps_per_worker or not learner.exit.is_set():
        # Send policy to actors
        if rank == 0:
            sd_cpu = {k: v.cpu() for k, v in learner.online.state_dict().items()}
            for actor_queue in learner.actor_queues:
                if actor_queue.empty():
                    actor_queue.put((True, sd_cpu))

        learner.lock.acquire()
        learner.buffer_conn.send((True, learner.minibatch))
        get_possible = learner.buffer_conn.recv()
        learner.lock.release()

        if not get_possible:
            learner.logger.warning("No batch available")
            continue

        loss = learner.train_batch(optimizer, learner.minibatch)

        local_steps += 1

        en = time.time()

        learner.logger.debug('grabbing steps_lock')
        learner.steps_lock.acquire()
        learner.steps[rank] = local_steps
        if rank == 0:
            if local_steps % 100 == 0 or local_steps % (1000 // learner.num_workers) == 0:
                new_total = torch.sum(learner.steps).item()
                learner.steps_lock.release()
                fps = learner.fps_alpha * ((new_total - prev_total) / (en - st)) \
                    + (1 - learner.fps_alpha) * fps
                prev_total = new_total
                learner.logger.info('{} (local - {})| fps - {}'.format(
                    prev_total, local_steps, fps))
                st = en
            else:
                learner.steps_lock.release()

            if local_steps % (1000 // learner.num_workers) == 0:
                sd_cpu = {k: v.cpu() for k, v in learner.online.state_dict().items()}
                learner.evaluator_queue.put((True, prev_total, sd_cpu))

        else:
            learner.steps_lock.release()
            learner.logger.debug('releasing steps_lock')

    if rank == 0:
        learner.evaluator_queue.put((False, None, None))
        learner.buffer_conn.send((False, None))
        time.sleep(1)


def learner_run(learner):
    learner.logger = get_logger()

    procs = []

    for i in range(learner.num_workers):
        p = mp.Process(name='learner_{}'.format(i), target=train_worker, args=(
            i,
            learner,
        ))
        learner.logger.info('Launching learner_{}'.format(i))
        p.start()
        procs.append(p)

    learner.ready.set()
    del learner

    for p in procs:
        p.join()
