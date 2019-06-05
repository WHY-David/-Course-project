import numpy as np
import random
import os
import gym

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, Input, concatenate
from keras.optimizers import Adam
from keras.losses import logcosh, mean_squared_error
from keras.metrics import mae
from keras.engine.saving import save_model

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from env import OsmoEnv

# Constants
HIDDEN1_UNITS = 30
HIDDEN2_UNITS = 10

# Environment
env = OsmoEnv()
# np.random.seed(123)
# env.seed(123)
nb_actions = env.action_space.shape[0]

# Actor Network
input = Input(shape=(1,) + env.observation_space.shape, name='actor_observation_input')
flattened_input = Flatten()(input)
x = Dense(HIDDEN1_UNITS, activation='sigmoid')(flattened_input)
x = Dense(HIDDEN2_UNITS, activation='sigmoid')(x)
outputA = Dense(nb_actions, activation='linear')(x)
actor = Model(inputs=input, outputs=outputA)
print(actor.summary())


# Critic Network
action_input = Input(shape=(nb_actions,), name='critic_action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='critic_observation_input')
flattened_observation = Flatten()(observation_input)
y = concatenate([action_input, flattened_observation])
y = Dense(HIDDEN1_UNITS, activation='sigmoid')(y)
y = Dense(HIDDEN2_UNITS, activation='sigmoid')(y)
outputC = Dense(1, activation='linear')(y)
critic = Model(inputs=[observation_input, action_input], outputs=outputC)
print(critic.summary())


# Add noise, build agent and compile
memory = SequentialMemory(limit=1000000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=300, nb_steps_warmup_actor=300,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(optimizer=Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))

if __name__ == '__main__':
    # Load
    # agent.load_weights('OsmoEnv.hdf5')

    # Train
    agent.fit(env, nb_steps=20000, verbose=1, nb_max_episode_steps=200)

    # Weights
    agent.save_weights('OsmoEnv.hdf5',overwrite=True)

    # Test
    # agent.test(env, visualize=False, nb_episodes=50, nb_max_episode_steps=200)

    #Play
    for _ in range(6):
        observation = env.reset()
        done=False
        while not done:
            action = agent.forward(observation)
            observation, reward, done, info = env.step(action)
        else:
            print(info)