import gym
import highway_env
import numpy as np
import argparse


from stable_baselines.common import set_global_seeds
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import logger
from stable_baselines import DQN

params = {
    "environment": "merge-v0",
    "model_name": "DQN-EXT",
    "train_steps": 200000,
    "buffer_size": 100000,
    "batch_size": 32,
    "gamma": 0.99,
    "target_update_interval": 5000,
    "train_freq": 1,
    "gradient_steps": 1,
    "exploration_fraction": 0.307843104782809,
    "exploration_final_eps": 0.07844201883172,
    "learning_rate": 0,
    "learning_starts": 0,
    "exploration_initial_eps": 0.995,
    "policy": "mlp",
    "test_episodes": 10000
}

CNN_config = {
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75
        }
    }

policy_kwargs = dict(dueling=True)

def train(params):

    # setup config
    if params.get("policy") == 'mlp':
        policy = MlpPolicy
        env = gym.make(params.get("environment"))
    else:
        policy = CnnPolicy
        env = gym.make(params.get("environment"))
        env.configure(CNN_config)
        env.reset()

    exp_name = ("{0}_{1}_{2}".format(
        params.get("model_name"),
        params.get("policy"),
        params.get("environment")))

    log_dir = './logs/' + exp_name

    # create model
    model = DQN(policy, env,
                verbose=1,
                tensorboard_log=log_dir,
                buffer_size=params.get("buffer_size"),
                learning_rate=params.get("learning_rate"),
                gamma=params.get("gamma"),
                target_network_update_freq=params.get("target_update_interval"),
                exploration_fraction=params.get("exploration_fraction"),
                exploration_final_eps=params.get("exploration_final_eps"),
                learning_starts=params.get("learning_starts"),
                batch_size=params.get("batch_size"),
                exploration_initial_eps = params.get("exploration_initial_eps"),
                double_q = True,
                prioritized_replay= True,
                prioritized_replay_alpha = 0.6,
                prioritized_replay_beta0 = 0.4,
                prioritized_replay_beta_iters = None,
                prioritized_replay_eps = 1e-06,
                train_freq = params.get("train_freq"),
                policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=params.get("train_steps"), log_interval=10)
    model.save(exp_name)
    env.close()
    del env


if __name__ == '__main__':
    train(params)
