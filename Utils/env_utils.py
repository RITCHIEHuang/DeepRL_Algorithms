import gym


def get_env_space(env_id):
    env = gym.make(env_id)
    # 解除环境限制
    env = env.unwrapped
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    return env, num_states, num_actions
