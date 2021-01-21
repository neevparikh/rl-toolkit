import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from atariari.benchmark.wrapper import AtariARIWrapper

from .gym_wrappers import *


def make_atari(env, num_frames, action_stack=False):
    """ Wrap env in atari processed env """
    try:
        noop_action = env.get_action_meanings().index("NOOP")
    except ValueError:
        print("Cannot find NOOP in env, defaulting to 0")
        noop_action = 0
    if action_stack:
        num_frames *= 2
    env = WarpFrame(env)
    env = MaxAndSkipEnv(env, 4)
    env = TorchWrap(env)
    env = TorchFrameStack(env, num_frames, action_stack=action_stack, reset_action=noop_action)
    return env


def make_ari(env):
    """ Wrap env in reset to match observation """
    return ResetARI(AtariARIWrapper(env))


def make_visual(env, shape):
    """ Wrap env to return pixel observations """
    env = PixelObservationWrapper(env, pixels_only=False, pixel_keys=("pixels",))
    env = ObservationDictToInfo(env, "pixels")
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape)
    return env


def get_wrapped_env(env_string, seed, wrapper_func, fake_display=False, **kwargs):
    if fake_display:
        from pyvirtualdisplay import Display
        _ = Display(visible=False, backend='xvfb').start()
    env = gym.make(env_string)
    test_env = gym.make(env_string)
    env = wrapper_func(env, **kwargs)
    test_env = wrapper_func(test_env, **kwargs)
    env.seed(seed)
    test_env.seed(seed + 1000)
    env.reset()
    test_env.reset()
    return env, test_env


def initialize_environment(args):
    # Initialize environment
    visual_cartpole_shape = (80, 120)
    visual_pendulum_shape = (120, 120)

    if args.env == "VisualCartPole-v0":
        env, test_env = get_wrapped_env("CartPole-v0", args.seed, make_visual, shape=visual_cartpole_shape)
    elif args.env == "VisualCartPole-v1":
        env, test_env = get_wrapped_env("CartPole-v1", args.seed, make_visual, shape=visual_cartpole_shape)
    elif args.env == "VisualPendulum-v0":
        env, test_env = get_wrapped_env("Pendulum-v0", args.seed, make_visual, shape=visual_pendulum_shape)
    elif args.env == "CartPole-PosAngle-v0":
        env, test_env = get_wrapped_env("CartPole-v0", args.seed,
                IndexedObservation, indices=[0, 1], fake_display=False)
    elif args.env == "CartPole-PosAngle-v1":
        env, test_env = get_wrapped_env("CartPole-v1", args.seed,
                IndexedObservation, indices=[0, 1], fake_display=False)
    elif args.env == "VisualMountainCar-v0":
        env, test_env = get_wrapped_env("MountainCar-v0", args.seed, make_visual,
                shape=visual_cartpole_shape)
    elif args.env == "VisualAcrobot-v1":
        env, test_env = get_wrapped_env("Acrobot-v1", args.seed, make_visual, shape=visual_pendulum_shape)
    elif args.env[:6] == 'Visual':
        env, test_env = get_wrapped_env(args.env[6:], make_visual, shape=(64, 64))
    elif not args.no_atari:
        env, test_env = get_wrapped_env(args.env, args.seed, make_atari,
                num_frames=args.num_frames, action_stack=args.action_stack)
    elif args.ari:
        env, test_env = get_wrapped_env(args.env, args.seed, make_ari)
    else:
        env, test_env = get_wrapped_env(args.env, args.seed, TorchWrap)

    return env, test_env
