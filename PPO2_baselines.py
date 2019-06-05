import gym
import os

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

from env import OsmoEnv

if __name__ == '__main__':
    env = SubprocVecEnv([lambda: OsmoEnv() for i in range(os.cpu_count())])
    # model = PPO2(MlpPolicy, env, verbose=1, learning_rate=1e-4)
    # model.learn(total_timesteps=25000)
    # model.save('PPO2_baselines')
    model = PPO2.load('PPO2_baselines')
    model.set_env(env)
    model.learning_rate = 1e-5
    model.learn(total_timesteps=30000)
    model.save('PP02_baselines')

    env = OsmoEnv()
    for i in range(10):
        observation = env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation)
            observation, _, done, info = env.step(action)
        else:
            print(info)