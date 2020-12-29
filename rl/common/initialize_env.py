import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from atariari.benchmark.wrapper import AtariARIWrapper

from .gym_wrappers import AtariPreprocess, MaxAndSkipEnv, FrameStack, ResetARI, \
        ObservationDictToInfo, ResizeObservation, IndexedObservation, TorchWrap, TorchFrameStack


def make_atari(env, num_frames, action_stack=False):
    """ Wrap env in atari processed env """
    try:
        noop_action = env.get_action_meanings().index("NOOP")
    except ValueError:
        print("Cannot find NOOP in env, defaulting to 0")
        noop_action = 0
    return TorchFrameStack(TorchWrap(MaxAndSkipEnv(AtariPreprocess(env), 4)),
                           num_frames,
                           action_stack=action_stack,
                           reset_action=noop_action)


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


def get_wrapped_env(env_string, wrapper_func, fake_display=True, **kwargs):
    if fake_display:
        from pyvirtualdisplay import Display
        _ = Display(visible=False, backend='xvfb').start()
    env = gym.make(env_string)
    test_env = gym.make(env_string)
    env.reset()
    test_env.reset()
    env = wrapper_func(env, **kwargs)
    test_env = wrapper_func(test_env, **kwargs)
    return env, test_env


def initialize_environment(args):
    # Initialize environment
    visual_cartpole_shape = (80, 120)
    visual_pendulum_shape = (120, 120)
    if args.env == "VisualCartPole-v0":
        env, test_env = get_wrapped_env("CartPole-v0", make_visual, shape=visual_cartpole_shape)
    elif args.env == "VisualCartPole-v1":
        env, test_env = get_wrapped_env("CartPole-v1", make_visual, shape=visual_cartpole_shape)
    elif args.env == "VisualPendulum-v0":
        env, test_env = get_wrapped_env("Pendulum-v0", make_visual, shape=visual_pendulum_shape)
    elif args.env == "CartPole-PosAngle-v0":
        env, test_env = get_wrapped_env("CartPole-v0", IndexedObservation, indices=[0, 1],
                fake_display=False)
    elif args.env == "CartPole-PosAngle-v1":
        env, test_env = get_wrapped_env("CartPole-v1", IndexedObservation, indices=[0, 1],
                fake_display=False)
    elif args.env == "VisualMountainCar-v0":
        env, test_env = get_wrapped_env("MountainCar-v0", make_visual, shape=visual_cartpole_shape)
    elif args.env == "VisualAcrobot-v1":
        env, test_env = get_wrapped_env("Acrobot-v1", make_visual, shape=visual_pendulum_shape)
    elif args.env[:6] == 'Visual':
        env, test_env = get_wrapped_env(args.env[6:], make_visual, shape=(64, 64))
    else:
        env = TorchWrap(gym.make(args.env))
        test_env = TorchWrap(gym.make(args.env))

    if args.model_type == 'cnn':
        assert args.num_frames

        if args.action_stack:
            print("Stacking actions")
            args.num_frames *= 2

        if not args.no_atari:
            print("Using atari preprocessing")
            env = make_atari(env, args.num_frames, action_stack=args.action_stack)
            test_env = make_atari(test_env, args.num_frames, action_stack=args.action_stack)
        else:
            print("FrameStacking with {}".format(args.num_frames))
            env = TorchFrameStack(env,
                                  args.num_frames,
                                  action_stack=args.action_stack,
                                  reset_action=0)
            test_env = TorchFrameStack(test_env,
                                       args.num_frames,
                                       action_stack=args.action_stack,
                                       reset_action=0)

    if args.ari:
        print("Using ARI")
        env = make_ari(env)
        test_env = make_ari(test_env)

    env.seed(args.seed)
    test_env.seed(args.seed + 1000)
    return env, test_env
