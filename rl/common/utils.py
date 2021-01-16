import pdb
from datetime import datetime
import random
import math
import sys
import os
import logging

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.multiprocessing as mp
import gym


class ForkablePdb(pdb.Pdb):

    _original_stdin_fd = sys.stdin.fileno()
    _original_stdin = None

    def __init__(self):
        pdb.Pdb.__init__(self, nosigint=True)

    def _cmdloop(self):
        current_stdin = sys.stdin
        try:
            if not self._original_stdin:
                self._original_stdin = os.fdopen(self._original_stdin_fd)
            sys.stdin = self._original_stdin
            self.cmdloop()
        finally:
            sys.stdin = current_stdin


numpy_to_torch_dtype_dict = {
    np.dtype('bool'): torch.bool,
    np.dtype('uint8'): torch.uint8,
    np.dtype('int8'): torch.int8,
    np.dtype('int16'): torch.int16,
    np.dtype('int32'): torch.int32,
    np.dtype('int64'): torch.int64,
    np.dtype('float16'): torch.float16,
    np.dtype('float32'): torch.float32,
    np.dtype('float64'): torch.float64,
    np.dtype('complex64'): torch.complex64,
    np.complex128: torch.complex128,
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


def construct_blank_tensors(size, transition_shapes):
    return [torch.zeros(size, *shp) for shp in transition_shapes]


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x)
    return x


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.5, 0.5)
        torch.nn.init.uniform_(m.bias, -1, 1)


def conv2d_size_out(size, kernel_size, stride):
    ''' Adapted from pytorch tutorials: 
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 
    '''
    return ((size[0] - (kernel_size[0] - 1) - 1) // stride + 1,
            (size[1] - (kernel_size[1] - 1) - 1) // stride + 1)


def deque_to_tensor(last_num_frames):
    """ Convert deque of n frames to tensor """
    return torch.cat(list(last_num_frames), dim=0)


def plot_grad_flow(named_parameters):
    '''
    Thanks to RoshanRane - Pytorch forums (Dec 2018)
        - (https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10)

    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow

    writer.add_figure('training/gradient_flow', plot_grad_flow(agent.online.named_parameters(), 
        episode), global_step=episode)

    '''
    plt.clf()
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=4, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=0, top=2.5)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([
        Line2D([0], [0], color="c", lw=4),
        Line2D([0], [0], color="b", lw=4),
        Line2D([0], [0], color="k", lw=4)
    ], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    return plt.gcf()


def reset_seeds(seed):
    # Setting cuda seeds
    if torch.cuda.is_available():
        torch.backends.cuda.deterministic = True
        torch.backends.cuda.benchmark = False

    # Setting random seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def append_timestamp(string, fmt_string=None):
    now = datetime.now()
    if fmt_string:
        return string + "_" + now.strftime(fmt_string)
    else:
        return string + "_" + str(now).replace(" ", "_")


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def weights_init_(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def get_logger():
    logger = mp.get_logger()
    lvl = logging.INFO
    h = logging.StreamHandler(stream=sys.stderr)
    h.setLevel(lvl)
    f = logging.Formatter(
        fmt='%(asctime)s | %(process)s - [%(levelname)s/%(processName)s] %(message)s', style='%')
    h.setFormatter(f)
    logger.addHandler(h)
    logger.setLevel(lvl)
    return logger
