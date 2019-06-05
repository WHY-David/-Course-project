import gym
import numpy as np

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

from env import OsmoEnv, NUMCONC

# callback
# best_mean_reward, n_steps = -np.inf, 0
# def callback(_locals, _globals):
#     global n_steps, best_mean_reward
#     mean_reward = 0
#     if(n_steps+1)%1000 == 0:
#         x, y = ts2xy(load_results(log_dir), 'timesteps')
#         if len(x) > 0:
#             mean_reward = np.mean(y[-100:])
#         print(x[-1], 'timesteps')
#         print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
#         if mean_reward > best_mean_reward:
#             best_mean_reward = mean_reward
#         # Example for saving best model
#         print("Saving new best model")
#         _locals['self'].save(log_dir + 'best_model.pkl')
#         n_steps += 1
#         return False
#
# # Create log dir
# log_dir = "/tmp/"
# os.makedirs(log_dir, exist_ok=True)

# environment
# env = OsmoEnv()
# env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: OsmoEnv()])

# parameters(for training)
tau = 0.1         # update rate for target model
gamma = 0.95        # discount rate for q value.
# batch_size = NUMCONC*5+3    # size of batch
batch_size = 10
alr = 0.003        # actor learning rate
clr = 0.003        # critic learning rate

# noise(to better exploration)
n_actions = env.action_space.shape[-1]
param_noise = AdaptiveParamNoiseSpec()
# action_noise = None
# param_noise = None
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions)) # A gaussian action noise

# model(DDPG)
#        Deep Deterministic Policy Gradient Algorithms.
#        DDPG is the combination of Nature DQN、Actor-Critic and DPG, it is designed to tackle continuous action space problems.
#        Policy-learning
#        The policy function(actor) takes state as input and is updated according to policy gradient.
#        Q-learning
#        The value function(critic) take state and action as input and is adjusted to minimize the loss.
#        Q-learning algorithm for function approximator is largely based on minimizing this MSBE loss function, with two main tricks, viz replay buffer and targrt network.
#        The replay buffer is used to store experience, because DDPG is an off-policy algorithm.
#        A target network is designed to minimize MSBE loss.
#        A target policy network to compute an action which approximately maximizes Q_{\phi_{\text{targ}}}.
#        Ornstein-Uhlenbeck process is applied to add exploration noise during training to make DDPG policies explore better.

model = DDPG(MlpPolicy, env, verbose=1, tau=tau, gamma=gamma, batch_size=batch_size, actor_lr=alr,
             critic_lr=clr, param_noise=param_noise, action_noise=action_noise)

if __name__ == '__main__':
    # train
    model.learn(total_timesteps=10000)
    model.save("DDPG_baselines")

    # play
    env=OsmoEnv()
    for i in range(10):
        observation = env.reset()
        done=False
        while not done:
            action, _ = model.predict(observation)
            observation, reward, done, info = env.step(action)
            # print(reward)
        print(info)