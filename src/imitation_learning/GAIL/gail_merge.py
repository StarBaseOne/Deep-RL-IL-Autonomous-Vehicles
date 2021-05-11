import gym
import highway_env
import numpy as np
from stable_baselines import GAIL, PPO1, TRPO
from stable_baselines.gail import ExpertDataset, generate_expert_traj
import gym.spaces as spaces
from gym import ObservationWrapper
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from gym.spaces import Box

EXP_MODEL = 'TRPO'

class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        return spaces.flatten(self.env.observation_space, observation)


params = {
    "environment": "merge-v0",
    "model_name": "GAIL_TRPO",
    "expert_timesteps": 100000,
    "n_episodes": 10,
    "train_steps": 200000,
    "batch_size": 256,
    "clip_range": 0.1,
    "ent_coef": 1.78387038356393E-07,
    "gae_lambda": 1,
    "gamma": 0.95,
    "learning_rate": 0.000874783914199,
    "max_grad_norm": 0.9,
    "n_epochs": 20,
    "n_steps": 128,
    "sde_sample_freq": 64,
    "vf_coef": 0.557588101099478,
    "seed": 0,
    "expert_name": "TRPO",
    "pre_train": True
}

def train(params):

    env = FlattenObservation(gym.make(params.get("environment")))
    exp_name = params.get("model_name") + "_train_" + params.get("environment")
    log_dir = './logs/' + exp_name
    expert_name = 'expert_{0}'.format(exp_name)

    if params.get("expert_name") == 'TRPO':
        print("Loading TRPO Model")
        model = TRPO(MlpPolicy, env,
                     verbose=1,
                     tensorboard_log=log_dir
                     )

    if params.get("expert_name") == 'PPO':
        print("Loading PPO Model")
        model = PPO1(MlpPolicy, env,
                     verbose=1,
                     tensorboard_log=log_dir,
                     entcoeff=params.get("ent_coef"),
                     gamma=params.get("gamma"),
                     optim_batchsize=params.get("batch_size"),
                     clip_param=params.get("clip_range"),
                     lam=params.get("gae_lambda")
                     )
    if params.get("pre_train") is False:
        print("Training expert trajectories")
        # Train expert controller (if needed) and record expert trajectories.
        generate_expert_traj(model, expert_name, n_timesteps=params.get("expert_timesteps"),
                             n_episodes=params.get("n_episodes"))

    dataset = ExpertDataset(expert_path='{0}.npz'.format(expert_name),
                            traj_limitation=-1,
                            randomize=True,  # if the dataset should be shuffled
                            verbose=1)

    model = GAIL('MlpPolicy', env, dataset, verbose=1, tensorboard_log=log_dir)  # Check out for defaults
    if params.get("pre_train") is True:
        print("Pretraining Dataset with Behavioural Cloning")
        model.pretrain(dataset, n_epochs=10000)

    print("Executing GAIL Learning")
    model.learn(total_timesteps=params.get("train_steps"))
    model.save(exp_name)

    env.close()
    del env


if __name__ == '__main__':
    train(params)


