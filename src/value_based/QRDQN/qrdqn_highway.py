import gym
import highway_env
import numpy as np
import argparse
from sb3_contrib import QRDQN
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter




CNN_observation = {
        "lanes_count": 3,
        "vehicles_count": 15,
        "observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (128, 64),
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
            "scaling": 1.75,
        },
        "policy_frequency": 2,
        "duration": 40,
    }

params = {
    "environment": "highway-v0",
    "model_name": "QRDQN",
    "train_steps": 200000,
    "batch_size": 128,
    "buffer_size": 100000,
    "exploration_final_eps": 0.032125724283431,
    "exploration_fraction": 0.186206036265873,
    "gamma": 0.98,
    "learning_starts": 10000,
    "learning_rate": 0.000118089190936,
    "n_quantiles": 172,
    "subsample_steps": 4,
    "gradient_steps": 1000,
    "target_update_interval": 1,
    "train_freq": 256,
    "policy": "CnnPolicy",
    "test_episodes": 10000,
}
env = gym.make(params.get("environment"))

env.configure(CNN_observation)
env.reset()
exp_name = params.get("model_name") + "_train_" + params.get("environment")
log_dir = '../../../logs/' + exp_name

def train(params):

    model = QRDQN(params.get("policy"), env,
                verbose=1,
                buffer_size=params.get("buffer_size"),
                learning_rate=params.get("learning_rate"),
                tensorboard_log=log_dir,
                   batch_size=params.get("batch_size"),
                policy_kwargs=dict(net_arch=[32, 32], n_quantiles=params.get("n_quantiles")))

    # Train for 1e5 steps
    model.learn(total_timesteps=params.get("train_steps"), log_interval=4)
    # Save the trained agent
    model.save(exp_name)

def evaluate(params):

    # Load saved model
    model = QRDQN.load(exp_name, env=env)
    results = np.zeros(shape=(0,0))
    obs = env.reset()

    # Evaluate the agent
    episode_reward = 0
    for _ in range(params.get("test_episodes")):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done or info.get('is_success', False):
            episode_reward = 0.0
            obs = env.reset()

        result = ("Reward:", episode_reward, "Success?", info.get('is_success', True))
        results = np.append(results, result, axis=None)


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_dir, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()


def sb3_plot():
    results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, exp_name)


train(params)
#evaluate(params)
