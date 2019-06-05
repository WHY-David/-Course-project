import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1

from env import OsmoEnv, NUMCONC

if __name__ == "__main__":
    env = DummyVecEnv([lambda: OsmoEnv()])
    model = PPO1(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("PPO1_baselines")

    for i in range(10):
        observation = env.reset()
        done = False
        while not done:
            action, _ = model.predict(observation)
            observation, _, done, info = env.step(action)
        else:
            print(info)
