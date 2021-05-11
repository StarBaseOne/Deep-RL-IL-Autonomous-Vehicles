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


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)
        self.observation_space = spaces.flatten_space(env.observation_space)

    def observation(self, observation):
        return spaces.flatten(self.env.observation_space, observation)


params = {
    "environment": "two-way-v0",
    "model_name": "GAIL_TRPO",
    "expert_timesteps": 100000,
    "n_episodes": 10,
    "train_steps": 200000,
    "batch_size": 256,
    "clip_range": 0.3,
    "ent_coef": 0.000280185342608,
    "gae_lambda": 0.9,
    "gamma": 0.98,
    "learning_rate": 5.73452493905307E-05,
    "max_grad_norm": 0.3,
    "n_epochs": 20,
    "n_steps": 512,
    "sde_sample_freq": -1,
    "vf_coef": 0.810626018991842,
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


