import gym
import numpy as np

from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC

from env import OsmoEnv

if __name__ == "__main__":
    env = DummyVecEnv([lambda : OsmoEnv()])

    model = SAC(MlpPolicy, env, verbose=1,learning_rate=1e-4)
    model.learn(total_timesteps=30000)
    model.save("SAC_baselines")

    env = OsmoEnv()
    for i in range(10):
        observation = env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation)
            observation, _, done, info = env.step(action)
        else:
            print(info)