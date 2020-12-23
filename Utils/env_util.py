import gym
from gym.spaces import Discrete

__all__ = ['get_env_info', 'get_env_space']


def get_env_space(env_id):
    env = gym.make(env_id)
    # 解除环境限制
    # env = env.unwrapped
    num_states = env.observation_space.shape[0]
    if type(env.action_space) == Discrete:
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]

    return env, num_states, num_actions


def get_env_info(env_id, unwrap=False):
    env = gym.make(env_id)
    if unwrap:  # 解除环境限制
        env = env.unwrapped
    num_states = env.observation_space.shape[0]
    env_continuous = False
    if type(env.action_space) == Discrete:
        num_actions = env.action_space.n
    else:
        num_actions = env.action_space.shape[0]
        env_continuous = True

    return env, env_continuous, num_states, num_actions
