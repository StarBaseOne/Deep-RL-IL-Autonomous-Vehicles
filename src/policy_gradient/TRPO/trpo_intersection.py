import gym
import highway_env
import numpy as np
import argparse

from mpi4py import MPI

from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy, CnnPolicy
from stable_baselines import logger
from stable_baselines.trpo_mpi import TRPO

CnnNet = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75
    }
}

params = {
    "environment": "intersection-v0",
    "model_name": "TRPO",
    "train_steps": 200000,
    "timesteps_per_batch": 512,
    "max_kl": 0.001,
    "cg_iters": 10,
    "cg_damping": 1e-3,
    "gamma": 0.98,
    "entcoeff": 0.00087,
    "lam": 0.9,
    "vf_iters": 3,
    "vf_stepsize": 1e-4,
    "seed": 0,
    "policy": "mlp"
}
policy_kwargs = dict(net_arch=[128, 128])


def train(params):
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])

    # setup config
    if params.get("policy") == 'mlp':
        policy = MlpPolicy
        env = gym.make(params.get("environment"))
    else:
        policy = CnnPolicy
        env = gym.make(params.get("environment"))
        env.configure(CnnNet)
        env.reset()

    exp_name = ("{0}_{1}_{2}".format(
        params.get("model_name"),
        params.get("policy"),
        params.get("environment")))

    log_dir = './logs/' + exp_name

    if params.get("seed") > 0:
        workerseed = params.get("seed"), + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)

    # create model
    model = TRPO(policy, env,
                 verbose=1,
                 tensorboard_log=log_dir,
                 timesteps_per_batch=params.get("timesteps_per_batch"),
                 max_kl=params.get("max_kl"),
                 cg_iters=params.get("cg_iters"),
                 cg_damping=params.get("cg_damping"),
                 entcoeff=params.get("entcoeff"),
                 gamma=params.get("gamma"),
                 lam=params.get("lam"),
                 vf_iters=params.get("vf_iters"),
                 vf_stepsize=params.get("vf_stepsize")
                 # ,policy_kwargs=policy_kwargs
                 )

    model.learn(total_timesteps=params.get("train_steps"))
    model.save(exp_name)
    env.close()
    del env


if __name__ == '__main__':
    train(params)
