import gym
import numpy as np
import os
import tensorflow.nn

# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import A2C

from env import OsmoEnv, NUMCONC

if __name__=='__main__':
    # multiprocessing
    env = SubprocVecEnv([lambda: OsmoEnv() for i in range(os.cpu_count())])
    # policy_kwargs = dict(act_fun=tensorflow.nn.tanh, net_arch=[64,32])
    # model = A2C('MlpPolicy', env, policy_kwargs=policy_kwargs, n_steps=30, learning_rate=0.003, verbose=1)

    model = A2C.load('A2C_baselines')
    model.set_env(env)
    model.n_steps = 30
    model.learning_rate=0.002
    model.learn(total_timesteps=20000)
    model.save('A2C_baselines')

    # play test
    env=OsmoEnv()
    for i in range(10):
        observation = env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation)
            observation, _, done, info = env.step(action)
        else:
            print(info)