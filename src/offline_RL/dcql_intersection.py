import gym
import highway_env
import numpy as np

import gym
from d3rlpy.algos import DQN
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy

from d3rlpy.wrappers.sb3 import to_mdp_dataset
import torch.nn as nn
import torch
from d3rlpy.algos import DiscreteCQL
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split
from d3rlpy.dataset import MDPDataset
from d3rlpy.models.q_functions import QRQFunctionFactory
import d3rlpy
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
    "environment": "intersection-v0",
    "model_name": "CQL-QRDQN",
    "train_steps": 200000,
    "buffer_size": 100000,
    "batch_size": 64,
    "gamma": 0.95,
    "target_update_interval": 1000,
    "train_freq": 1,
    "gradient_steps": -1,
    "exploration_fraction": 0.084170222643946,
    "exploration_final_eps": 0.080869221015041,
    "learning_rate": 0.000557346630876,
    "learning_starts": 10000,
    "policy": "MlpPolicy",
    "test_episodes": 10000,
    "n_quantiles": 193,
}

env = gym.make(params.get("environment"))
eval_env = gym.make(params.get("environment"))
env = FlattenObservation(env)
eval_env = FlattenObservation(env)
exp_name = params.get("model_name") + "_online_" + params.get("environment")
log_dir = '../../../logs/' + exp_name
pretrain = False

def train(params):
    # setup algorithm
    # setup algorithm
    if pretrain:

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

        print("Saving Model")
        dqn.save_model(exp_name)

        print("convert buffer to dataset")
        dataset = buffer.to_mdp_dataset()
        # save MDPDataset
        dataset.dump('{0}.h5'.format(exp_name))

    print("Loading Dataset for Offline Training")
    dataset = d3rlpy.dataset.MDPDataset.load('{0}.h5'.format(exp_name))
    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)
    # The dataset can then be used to train a d3rlpy model

    cql = DiscreteCQL(
        learning_rate=6.25e-05,
        encoder_factory='default',
        q_func_factory='mean',
        batch_size=32,
        n_frames=1,
        n_steps=1,
        gamma=0.99,
        n_critics=1,
        bootstrap=False,
        share_encoder=False,
        target_reduction_type='min',
        target_update_interval=8000,
        use_gpu=True,
        scaler=None,
        augmentation=None,
        generator=None,
        impl=None)

    cql_exp = params.get("model_name") + "_offline_" + params.get("environment")
    cql_log = '../../../logs/' + cql_exp

    cql.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_epochs=1000,
            scorers={
                'environment': evaluate_on_environment(env, epsilon=0.05),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer,
                'value_scale': average_value_estimation_scorer,

            },
            tensorboard_dir= cql_log)

    cql.save_model(cql_exp)

train(params)
