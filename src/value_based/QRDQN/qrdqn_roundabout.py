import gym
import highway_env
import numpy as np



from gym.spaces import Box
import gym
import highway_env
from d3rlpy.algos import DQN
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy

import torch.nn as nn
import torch
from d3rlpy.models.q_functions import QRQFunctionFactory

import gym.spaces as spaces
from gym import ObservationWrapper


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""
    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        return spaces.flatten(self.env.observation_space, observation)


params = {
    "environment": "roundabout-v0",
    "model_name": "QRDQN",
    "train_steps": 200000,
    "buffer_size": 10000,
    "batch_size": 32,
    "gamma": 0.99,
    "target_update_interval": 15000,
    "train_freq": 256,
    "gradient_steps": -1,
    "exploration_fraction": 0.0943543691594673,
    "exploration_final_eps": 0.170467871171168,
    "learning_rate": 0.000403307327179189,
    "learning_starts": 10000,
    "policy": "MlpPolicy",
    "test_episodes": 10000,
    "n_quantiles": 192,
}

env = gym.make('highway-v0')
eval_env = gym.make('highway-v0')
env = FlattenObservation(env)
eval_env = FlattenObservation(env)
exp_name = params.get("model_name") + "_train_" + params.get("environment")
log_dir = '../../../logs/' + exp_name


def train(params):
    # setup algorithm
    dqn = DQN(batch_size=params.get("batch_size"),
              learning_rate=params.get("learning_rate"),
              target_update_interval=params.get("target_update_interval"),
              q_func_factory=QRQFunctionFactory(n_quantiles=params.get("n_quantiles")),
              n_steps= params.get("train_freq"),
              gamma= params.get("gamma"),
              n_critics= 1,
              target_reduction_type= "min",
              use_gpu=True)

    # setup replay buffer
    buffer = ReplayBuffer(maxlen=params.get("buffer_size"), env=env)

    # setup explorers
    explorer = LinearDecayEpsilonGreedy(start_epsilon=1.0,
                                        end_epsilon=params.get("exploration_final_eps"),
                                        duration=100000)

    # start training
    dqn.fit_online(env,
                   buffer,
                   n_steps=params.get("train_steps"),
                   explorer=explorer, # you don't need this with probablistic policy algorithms
                   tensorboard_dir= log_dir,
                   eval_env=eval_env)

    dqn.save_model(exp_name)
train(params)
