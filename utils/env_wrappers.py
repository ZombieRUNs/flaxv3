import ale_py
import gymnasium
import numpy as np
from collections import deque
from einops import rearrange
import copy
import cv2


class LifeLossInfo(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives_info = None

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        current_lives_info = info["lives"]
        if current_lives_info < self.lives_info:
            info["life_loss"] = True
            self.lives_info = info["lives"]
        else:
            info["life_loss"] = False

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.lives_info = info["lives"]
        info["life_loss"] = False
        return observation, info


class SeedEnvWrapper(gymnasium.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.seed = seed
        self.env.action_space.seed(seed)

    def reset(self, **kwargs):
        kwargs["seed"] = self.seed
        obs, _ = self.env.reset(**kwargs)
        return obs, _

    def step(self, action):
        return self.env.step(action)


class MaxLast2FrameSkipWrapper(gymnasium.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs, _

    def step(self, action):
        total_reward = 0
        self.obs_buffer = deque(maxlen=2)
        for _ in range(self.skip):
            obs, reward, done, truncated, info = self.env.step(action)
            self.obs_buffer.append(obs)
            total_reward += reward
            if done or truncated:
                break
        if len(self.obs_buffer) == 1:
            obs = self.obs_buffer[0]
        else:
            obs = np.max(np.stack(self.obs_buffer), axis=0)
        return obs, total_reward, done, truncated, info

def build_single_env(env_name, image_size, seed):
    env_name = f'ALE/{env_name}-v5'
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = SeedEnvWrapper(env, seed=seed)
    env = MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = LifeLossInfo(env)
    return env

def build_vec_env(env_name, image_size, num_envs, seed):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, image_size):
        return lambda: build_single_env(env_name, image_size, seed)
    env_fns = []
    env_fns = [lambda_generator(env_name, image_size) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env

def build_eval_env(env_name, image_size):
    env_name = f'ALE/{env_name}-v5'
    env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
    env = MaxLast2FrameSkipWrapper(env, skip=4)
    env = gymnasium.wrappers.ResizeObservation(env, shape=image_size)
    env = LifeLossInfo(env)
    return env

def build_eval_vec_env(env_name, image_size, num_envs):
    def lambda_generator(env_name, image_size):
        return lambda: build_eval_env(env_name, image_size)
    env_fns = [lambda_generator(env_name, image_size) for _ in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env
    

if __name__ == "__main__":
    # vec_env = build_vec_env('ALE/Pong-v5',(64,64),4,1)
    # current_obs, _ = vec_env.reset()
    # print(vec_env.observation_space.dtype)
    # print(vec_env.action_space.dtype)
    pass